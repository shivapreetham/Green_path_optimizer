import os, time, requests, numpy as np, polyline
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Load .env
load_dotenv()
#get it for yourself, it's free 😘
GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

EMISSION_RATES = {"EV":0, "Petrol":120, "Diesel":180}
WEIGHTS        = {"aqi":0.4, "zones":0.3, "emissions":0.2, "time":0.1}

app = Flask(__name__)

# Geocode an address to lat/lng
def geocode(addr):
    j = requests.get(
      "https://maps.googleapis.com/maps/api/geocode/json",
      params={"address": addr, "key": GOOGLE_API_KEY}
    ).json()
    loc = j["results"][0]["geometry"]["location"]
    return {"lat": loc["lat"], "lng": loc["lng"]}

# Cluster orders spatially
def cluster_orders(ords, eps_km=1.0, min_samples=2):
    coords = np.array([[o['lat'],o['lng']] for o in ords])
    labels = DBSCAN(eps=eps_km/111, min_samples=min_samples).fit(coords).labels_
    clusters = {}
    for o,lab in zip(ords, labels):
        if lab<0: continue
        clusters.setdefault(lab, []).append(o)
    return list(clusters.values())

# Fetch time & distance with traffic
def fetch_matrix(nodes):
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
    t = np.zeros((N,N)); d = np.zeros((N,N))
    for i,row in enumerate(resp["rows"]):
        for j,el in enumerate(row["elements"]):
            t[i,j] = el["duration_in_traffic"]["value"]
            d[i,j] = el["distance"]["value"]
    return t, d

# Get route polyline
def fetch_geometry(a, b):
    res = requests.get(
      "https://maps.googleapis.com/maps/api/directions/json",
      params={
        "origin": f"{a['lat']},{a['lng']}",
        "destination": f"{b['lat']},{b['lng']}",
        "departure_time":"now",
        "traffic_model":"best_guess",
        "key": GOOGLE_API_KEY
      }
    ).json()
    return polyline.decode(res["routes"][0]["overview_polyline"]["points"])

# Sample AQI along route
def sample_aqi(a, b, samples=5):
    pts = fetch_geometry(a, b)
    step = max(1, len(pts)//samples)
    vals = []
    for lat,lng in pts[0::step]:
        j = requests.get(
          "http://api.openweathermap.org/data/2.5/air_pollution",
          params={"lat":lat, "lon":lng, "appid": OPENWEATHER_API_KEY}
        ).json()
        vals.append(j["list"][0]["main"]["aqi"])
        time.sleep(0.1)
    return np.mean(vals)

# Count sensitive POIs along route
def count_zones(a, b, samples=5):
    pts = fetch_geometry(a, b)
    step = max(1, len(pts)//samples)
    tot = 0
    for lat,lng in pts[0::step]:
        for t in ["school","hospital","shopping_mall","place_of_worship"]:
            j = requests.get(
              "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
              params={"location":f"{lat},{lng}", "radius":100, "type":t, "key": GOOGLE_API_KEY}
            ).json()
            tot += len(j.get("results", []))
        time.sleep(0.1)
    return tot

# Build weighted cost matrix
def build_cost(t, d, aqi, zones, em):
    def norm(m):
        mn,mx = m.min(), m.max()
        return (m - mn) / (mx - mn + 1e-9)
    nt, na, nz, ne = norm(t), norm(aqi), norm(zones), norm((d/1000)*em)
    w = WEIGHTS
    return w["time"]*nt + w["aqi"]*na + w["zones"]*nz + w["emissions"]*ne

# Solve TSP. Please dont ask me , i dont know whats hapenning
def solve_tsp(cost):
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
    return route

@app.route("/api/eco_route", methods=["POST"])
def eco_route():
    data = request.json

    # 1. Read source, destinations, and vehicle type
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
    cost_mat = build_cost(
        tmat,
        dmat,
        aqi_mat,
        zone_mat,
        emission_rate
    )

    # 8. Solve the one‑vehicle TSP (returns a list of indices into `nodes`)
    visit_order = solve_tsp(cost_mat)
    print("Visit order:", visit_order)
    # 9. Map indices back to lat/lng objects
    route = [nodes[idx] for idx in visit_order]

    return jsonify(route), 200

# Front‑end
@app.route("/")
def index():
    return render_template("index.html",
                           api_url="/api/eco_route",
                           gmaps_key=GOOGLE_API_KEY)

if __name__=="__main__":
    app.run(debug=True)
