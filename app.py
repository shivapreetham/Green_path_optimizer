import os, time, requests, numpy as np, polyline, json
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
import redis

# Load .env
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_TTL = 86400  # 1 day

EMISSION_RATES = {"EV":0, "Petrol":120, "Diesel":180}
WEIGHTS        = {"aqi":0.4, "zones":0.3, "emissions":0.2, "time":0.1}

app = Flask(__name__)
rdb = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def log_time(label, start):
    end = time.time()
    print(f"{label}: {end - start:.2f}s")

def geocode(addr):
    start = time.time()
    j = requests.get(
        "https://maps.googleapis.com/maps/api/geocode/json",
        params={"address": addr, "key": GOOGLE_API_KEY}
    ).json()
    loc = j["results"][0]["geometry"]["location"]
    log_time("📍 Geocode API", start)
    return {"lat": loc["lat"], "lng": loc["lng"]}

def cluster_orders(ords, eps_km=1.0, min_samples=1):
    coords = np.array([[o['lat'], o['lng']] for o in ords])
    labels = DBSCAN(eps=eps_km/111, min_samples=min_samples).fit(coords).labels_
    clusters = {}
    for o, lab in zip(ords, labels):
        if lab < 0: continue
        clusters.setdefault(lab, []).append(o)
    return list(clusters.values())

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

def fetch_geometry(a, b):
    key = f"geom:{a['lat']},{a['lng']}-{b['lat']},{b['lng']}"
    if cached := rdb.get(key):
        return json.loads(cached)

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

    pts = polyline.decode(res["routes"][0]["overview_polyline"]["points"])
    rdb.setex(key, REDIS_TTL, json.dumps(pts))
    log_time("🛣️ Directions API (with cache)", start)
    return pts



async def fetch_aqi(session, lat, lng):
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
        return (pm25 + pm10) / 2

async def async_sample_aqi(a, b, samples=5):
    key = f"aqi:{a['lat']},{a['lng']}-{b['lat']},{b['lng']}"
    if cached := rdb.get(key):
        return float(cached)

    start = time.time()
    pts = fetch_geometry(a, b)
    step = max(1, len(pts) // samples)
    sample_pts = pts[0::step]

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_aqi(session, lat, lng) for lat, lng in sample_pts]
        vals = await asyncio.gather(*tasks)

    avg = float(np.mean(vals))
    rdb.setex(key, REDIS_TTL, avg)
    log_time("🌫️ Sample AQI (async, cached)", start)
    return avg


OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def buffer_route(route_pts, buffer_m=100):
    line = LineString([(lng, lat) for lat, lng in route_pts])
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
    if types is None:
        types = ["school", "hospital", "shopping_mall", "place_of_worship"]

    poly = buffer_route(route_pts, buffer_m)
    minx, miny, maxx, maxy = poly.bounds

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

    resp = requests.get(OVERPASS_URL, params={"data": query}).json()

    seen = set()
    for el in resp.get("elements", []):
        if el["type"] == "node":
            lat, lon = el["lat"], el["lon"]
        else:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        if poly.contains(Point(lon, lat)):
            seen.add((el["type"], el["id"]))
    return len(seen)

async def async_count_zones(a, b):
    key = f"zones:{a['lat']},{a['lng']}-{b['lat']},{b['lng']}"
    if cached := rdb.get(key):
        return int(cached)

    start = time.time()
    route_pts = fetch_geometry(a, b)
    total = count_zones_overpass(route_pts, buffer_m=100)
    rdb.setex(key, REDIS_TTL, total)
    log_time("☣️ Overpass Zone Count (async, cached)", start)
    return total


def build_cost(t, d, aqi, zones, em):
    start = time.time()
    def norm(m):
        mn, mx = m.min(), m.max()
        return (m - mn) / (mx - mn + 1e-9)
    nt, na, nz, ne = norm(t), norm(aqi), norm(zones), norm((d/1000)*em)
    w = WEIGHTS
    log_time("💰 Build Cost Matrix", start)
    return w["time"]*nt + w["aqi"]*na + w["zones"]*nz + w["emissions"]*ne

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

async def compute_all_pairs(nodes):
    N = len(nodes)
    aqi_mat  = np.zeros((N, N))
    zone_mat = np.zeros((N, N))

    sem = asyncio.Semaphore(5)  # Limit concurrency

    async def compute(i, j):
        if i == j: return
        async with sem:
            aqi = await async_sample_aqi(nodes[i], nodes[j])
            zones = await async_count_zones(nodes[i], nodes[j])
            aqi_mat[i][j] = aqi
            zone_mat[i][j] = zones

    await asyncio.gather(*(compute(i, j) for i in range(N) for j in range(N)))
    return aqi_mat, zone_mat


@app.route("/api/eco_route", methods=["POST"])
def eco_route():
    total_start = time.time()
    data = request.json

    src   = data["source"]
    dests = data["destinations"]
    veh   = data.get("vehicle", "Petrol")
    emission_rate = EMISSION_RATES.get(veh, EMISSION_RATES["Petrol"])

    clusters = cluster_orders(dests)
    if not clusters:
        clusters = [dests]  # fallback

    all_routes = []

    for cluster in clusters:
        nodes = [src] + cluster

        tmat, dmat = fetch_matrix(nodes)
        aqi_mat, zone_mat = asyncio.run(compute_all_pairs(nodes))
        cost_mat = build_cost(tmat, dmat, aqi_mat, zone_mat, emission_rate)
        visit_order = solve_tsp(cost_mat)

        optimized_route = [nodes[idx] for idx in visit_order]
        all_routes.append(optimized_route)

    log_time("🚚 /api/eco_route Total Time", total_start)
    return jsonify(all_routes), 200


@app.route("/")
def index():
    return render_template("index.html",
                           api_url="/api/eco_route",
                           gmaps_key=GOOGLE_API_KEY)

if __name__ == "__main__":
    app.run(debug=True)
