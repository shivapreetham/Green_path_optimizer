import os, time, requests, numpy as np, polyline
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from shapely.geometry import LineString, Point
from shapely.ops import transform
import pyproj
import aiohttp
import asyncio
from haversine import haversine


# Load .env
load_dotenv()

GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY")
# OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

EMISSION_RATES = {"EV":0, "Petrol":120, "Diesel":180}
WEIGHTS        = {"aqi":0.4, "zones":0.3, "emissions":0.2, "time":0.1}

app = Flask(__name__)

# ─── TIME LOGGER ──────────────────────────────────────────────
def log_time(label, start):
    end = time.time()
    print(f"{label}: {end - start:.2f}s")

# ─── Geocode an address to lat/lng ─────────────────────────────
def geocode(addr):
    start = time.time()
    j = requests.get(
        "https://maps.googleapis.com/maps/api/geocode/json",
        params={"address": addr, "key": GOOGLE_API_KEY}
    ).json()
    loc = j["results"][0]["geometry"]["location"]
    log_time("📍 Geocode API", start)
    return {"lat": loc["lat"], "lng": loc["lng"]}

# ─── Cluster orders spatially ──────────────────────────────────
def cluster_orders(ords, eps_km=1.0, min_samples=1):
    coords = np.array([[o['lat'], o['lng']] for o in ords])
    labels = DBSCAN(eps=eps_km/111, min_samples=min_samples).fit(coords).labels_
    clusters = {}
    for o, lab in zip(ords, labels):
        if lab < 0: continue
        clusters.setdefault(lab, []).append(o)
    return list(clusters.values())

# ─── Fetch time & distance with traffic ────────────────────────
def fetch_matrix(nodes):
    start = time.time()
    coords = [f"{n['lat']},{n['lng']}" for n in nodes]
    resp = requests.get(
        "https://maps.googleapis.com/maps/api/distancematrix/json",
        params={
            "origins": "|".join(coords),
            "destinations": "|".join(coords),
            "departure_time": "now",
            "traffic_model": "best_guess",
            "key": GOOGLE_API_KEY
        }
    ).json()
    N = len(nodes)
    t = np.zeros((N, N)); d = np.zeros((N, N))
    for i, row in enumerate(resp["rows"]):
        for j, el in enumerate(row["elements"]):
            t[i, j] = el["duration_in_traffic"]["value"]
            d[i, j] = el["distance"]["value"]
    log_time("🧭 Distance Matrix API", start)
    return t, d

# ─── Get route polyline ────────────────────────────────────────
def fetch_geometry(a, b):
    start = time.time()
    res = requests.get(
        "https://maps.googleapis.com/maps/api/directions/json",
        params={
            "origin": f"{a['lat']},{a['lng']}",
            "destination": f"{b['lat']},{b['lng']}",
            "departure_time": "now",
            "traffic_model": "best_guess",
            "key": GOOGLE_API_KEY
        }
    ).json()
    log_time("🛣️ Directions API", start)
    return polyline.decode(res["routes"][0]["overview_polyline"]["points"])



# ─── Sample AQI along route (Open-Meteo, Async Optimized) ──────────────────────────
async def fetch_aqi(session, lat, lng):
    t1 = time.time()
    async with session.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude":  lat,
            "longitude": lng,
            "hourly":    "pm2_5,pm10",
            "timezone":  "UTC"
        }
    ) as resp:
        j = await resp.json()
        pm25 = j["hourly"]["pm2_5"][0]
        pm10 = j["hourly"]["pm10"][0]
        log_time("🌫️ Open-Meteo AQI (one point)", t1)
        return (pm25 + pm10) / 2

def sample_aqi(a, b, samples=5):
    start = time.time()
    # 1. Get full route geometry
    pts = fetch_geometry(a, b)

    # 2. Pick up to `samples` evenly spaced points
    step = max(1, len(pts) // samples)
    sample_pts = pts[0::step]

    # 3. Async fetch all AQI readings concurrently
    async def gather_aqi():
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_aqi(session, lat, lng) for lat, lng in sample_pts]
            return await asyncio.gather(*tasks)

    vals = asyncio.run(gather_aqi())

    # 4. Log & return the mean
    log_time("🌫️ Sample AQI (full, Open-Meteo, Async)", start)
    return np.mean(vals)


# def sample_aqi(a, b, samples=5):
#     start = time.time()
#     pts = fetch_geometry(a, b)
#     step = max(1, len(pts)//samples)
#     vals = []
#     for lat, lng in pts[0::step]:
#         t1 = time.time()
#         j = requests.get(
#             "http://api.openweathermap.org/data/2.5/air_pollution",
#             params={"lat": lat, "lon": lng, "appid": OPENWEATHER_API_KEY}
#         ).json()
#         vals.append(j["list"][0]["main"]["aqi"])
#         time.sleep(0.1)
#         log_time("🌫️ AQI API (one point)", t1)
#     log_time("🌫️ Sample AQI (full)", start)
#     return np.mean(vals)




# ─── Count sensitive POIs along route ──────────────────────────

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def buffer_route(route_pts, buffer_m=100):
    """
    route_pts: list of (lat, lng)
    buffer_m: buffer distance in meters
    Returns a Shapely polygon (in WGS84) covering route ± buffer_m.
    """
    # 1) Build a lon/lat LineString
    line = LineString([(lng, lat) for lat, lng in route_pts])

    # 2) Project to a local Azimuthal Equidistant CRS (meters)
    wgs84 = pyproj.CRS("EPSG:4326")
    aeqd  = pyproj.CRS.from_proj4(
        f"+proj=aeqd +lat_0={route_pts[0][0]} +lon_0={route_pts[0][1]} +units=m"
    )
    proj_to_aeqd = pyproj.Transformer.from_crs(wgs84, aeqd, always_xy=True).transform
    proj_to_wgs84 = pyproj.Transformer.from_crs(aeqd, wgs84, always_xy=True).transform

    line_m = transform(proj_to_aeqd, line)
    buff_m = line_m.buffer(buffer_m)
    return transform(proj_to_wgs84, buff_m)

def count_zones_overpass(route_pts, buffer_m=100, types=None):
    """
    Counts unique POIs of given amenities within buffer_m meters of the route.
    route_pts: list of (lat, lng)
    types: list of OSM amenity tags (defaults to your 4 types)
    """
    if types is None:
        types = ["school", "hospital", "shopping_mall", "place_of_worship"]

    # 1) Build the buffer polygon
    poly = buffer_route(route_pts, buffer_m)
    minx, miny, maxx, maxy = poly.bounds  # lon_min, lat_min, lon_max, lat_max

    # 2) Construct Overpass QL to fetch nodes/ways/rels in that bbox
    tag_filter = "".join(f'["amenity"="{t}"]' for t in types)
    query = f"""
      [out:json][timeout:25];
      (
        node{tag_filter}({miny},{minx},{maxy},{maxx});
        way{tag_filter}({miny},{minx},{maxy},{maxx});
        rel{tag_filter}({miny},{minx},{maxy},{maxx});
      );
      out center;
    """

    # 3) Send single Overpass request
    resp = requests.get(OVERPASS_URL, params={"data": query}).json()

    # 4) Filter elements to those truly inside the buffer and dedupe
    seen = set()
    for el in resp.get("elements", []):
        # pick a coordinate to test: node → (lat,lon); way/rel → center
        if el["type"] == "node":
            lat, lon = el["lat"], el["lon"]
        else:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        # shapely Point expects (x=lon, y=lat)
        if poly.contains(Point(lon, lat)):
            seen.add((el["type"], el["id"]))
    return len(seen)

def count_zones(a, b, samples=5):
    # 1) get full route geometry
    route_pts = fetch_geometry(a, b)
    # 2) count POIs in buffer corridor (100 m each side)
    start = time.time()
    tot = count_zones_overpass(route_pts, buffer_m=100)
    log_time("☣️ Overpass Sensitive Zone Scan", start)
    return tot


# def count_zones(a, b, samples=5):
#     start = time.time()
#     pts = fetch_geometry(a, b)
#     step = max(1, len(pts)//samples)
#     tot = 0
#     for lat, lng in pts[0::step]:
#         for t in ["school", "hospital", "shopping_mall", "place_of_worship"]:
#             t1 = time.time()
#             j = requests.get(
#                 "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
#                 params={"location": f"{lat},{lng}", "radius": 100, "type": t, "key": GOOGLE_API_KEY}
#             ).json()
#             tot += len(j.get("results", []))
#             time.sleep(0.1)
#             log_time(f"🏥 Places API ({t})", t1)
#     log_time("☣️ Total Sensitive Zone Scan", start)
#     return tot



# ─── Build weighted cost matrix ────────────────────────────────
def build_cost(t, d, aqi, zones, em):
    start = time.time()
    def norm(m):
        mn, mx = m.min(), m.max()
        return (m - mn) / (mx - mn + 1e-9)
    nt, na, nz, ne = norm(t), norm(aqi), norm(zones), norm((d/1000)*em)
    w = WEIGHTS
    log_time("💰 Build Cost Matrix", start)
    return w["time"]*nt + w["aqi"]*na + w["zones"]*nz + w["emissions"]*ne

# ─── Solve TSP. Please dont ask me , i dont know whats hapenning ─────
def solve_tsp(cost):
    start = time.time()
    N = cost.shape[0]
    mgr = pywrapcp.RoutingIndexManager(N, 1, 0)
    r   = pywrapcp.RoutingModel(mgr)
    def cb(i, j):
        return int(cost[mgr.IndexToNode(i), mgr.IndexToNode(j)] * 1000)
    idx = r.RegisterTransitCallback(cb)
    r.SetArcCostEvaluatorOfAllVehicles(idx)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sol = r.SolveWithParameters(params)
    route, cur = [], r.Start(0)
    while not r.IsEnd(cur):
        route.append(mgr.IndexToNode(cur))
        cur = sol.Value(r.NextVar(cur))
    route.append(mgr.IndexToNode(cur))
    log_time("🧠 TSP Solve Time", start)
    return route


# ─── API Endpoint ────────────────────────────────────────────────
@app.route("/api/eco_route", methods=["POST"])
def eco_route():
    total_start = time.time()
    data = request.json

    src   = data["source"]           # dict {lat, lng}
    dests = data["destinations"]     # list of dicts
    veh   = data.get("vehicle", "Petrol")
    emission_rate = EMISSION_RATES.get(veh, EMISSION_RATES["Petrol"])

    # 2. Cluster destinations into spatial batches (min_samples=2 by default)
    clusters = cluster_orders(dests)
    print("Clusters:", clusters)

    # 3. If no cluster (e.g. only 1 destination), just use all dests as a single batch
    if clusters:
        batch = clusters[0]
    else:
        batch = dests

    # 4. Build the list of nodes (depot + batch)
    nodes = [src] + batch

    # 5. Fetch real-time travel time & distance matrix
    tmat, dmat = fetch_matrix(nodes)

    # 6. Sample AQI & count sensitive zones for each pair
    N = len(nodes)
    aqi_mat  = np.zeros((N, N))
    zone_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            aqi_mat[i, j]  = sample_aqi(nodes[i], nodes[j])
            zone_mat[i, j] = count_zones(nodes[i], nodes[j])


    # 7. Build composite cost matrix
    cost_mat = build_cost(tmat, dmat, aqi_mat, zone_mat, emission_rate)

    # 8. Solve the one‑vehicle TSP (returns a list of indices into `nodes`)
    visit_order = solve_tsp(cost_mat)
    print("Visit order:", visit_order)

    # 9. Map indices back to lat/lng objects
    route = [nodes[idx] for idx in visit_order]

    log_time("🚚 /api/eco_route Total Time", total_start)
    return jsonify(route), 200

# ─── Frontend ──────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           api_url="/api/eco_route",
                           gmaps_key=GOOGLE_API_KEY)

if __name__ == "__main__":
    app.run(debug=True)
