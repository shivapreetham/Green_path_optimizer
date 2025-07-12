import os
import time
import numpy as np
import requests
import polyline
from sklearn.cluster import DBSCAN
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

EMISSION_RATES = {
    "EV":     0,
    "Petrol": 120,
    "Diesel": 180,
}

WEIGHTS = {
    "aqi":       0.4,
    "zones":     0.3,
    "emissions": 0.2,
    "time":      0.1,
}

# Logging helper
def log_time(label, start_time):
    print(f"{label} took {time.time() - start_time:.2f} seconds")

# ─── 1) CLUSTER ORDERS ─────────────────────────────────────────────
def cluster_orders(orders, eps_km=1.0, min_samples=2):
    start = time.time()
    coords = np.array([[o['lat'], o['lng']] for o in orders])
    db = DBSCAN(eps=eps_km/111, min_samples=min_samples).fit(coords)
    labels = db.labels_
    clusters = {}
    for o, lab in zip(orders, labels):
        if lab < 0: continue
        clusters.setdefault(lab, []).append(o)
    log_time("🧠 Clustering Orders", start)
    print("Clusters:", list(clusters.values()))
    return list(clusters.values())

# ─── 2) DISTANCE & TIME MATRIX ─────────────────────────────────────
def fetch_google_matrix(nodes):
    start = time.time()
    coords = [f"{n['lat']},{n['lng']}" for n in nodes]
    params = {
        "origins":       "|".join(coords),
        "destinations":  "|".join(coords),
        "departure_time":"now",
        "traffic_model": "best_guess",
        "key":           GOOGLE_API_KEY
    }
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    resp = requests.get(url, params=params).json()
    N = len(nodes)
    time_mat = np.zeros((N, N))
    dist_mat = np.zeros((N, N))
    for i, row in enumerate(resp['rows']):
        for j, elem in enumerate(row['elements']):
            time_mat[i,j] = elem['duration_in_traffic']['value']
            dist_mat[i,j] = elem['distance']['value']
    log_time("📏 Distance Matrix API", start)
    return time_mat, dist_mat

# ─── 3) FETCH ROUTE GEOMETRY ───────────────────────────────────────
def fetch_route_geometry(orig, dest):
    start = time.time()
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin":        f"{orig['lat']},{orig['lng']}",
        "destination":   f"{dest['lat']},{dest['lng']}",
        "departure_time":"now",
        "traffic_model": "best_guess",
        "alternatives":  "false",
        "key":           GOOGLE_API_KEY
    }
    res = requests.get(url, params=params).json()
    poly = res["routes"][0]["overview_polyline"]["points"]
    log_time("🛣️ Directions API (fetch_geometry)", start)
    return polyline.decode(poly)

# ─── 4) SAMPLE AQI ─────────────────────────────────────────────────
def sample_aqi_along_points(points):
    aqi_vals = []
    for lat, lng in points:
        params = {"lat": lat, "lon": lng, "appid": OPENWEATHER_API_KEY}
        j = requests.get("http://api.openweathermap.org/data/2.5/air_pollution", params=params).json()
        aqi_vals.append(j['list'][0]['main']['aqi'])
        time.sleep(0.1)
    return np.mean(aqi_vals)

def sample_aqi_for_pair(orig, dest, samples=5):
    start = time.time()
    geom = fetch_route_geometry(orig, dest)
    step = max(1, len(geom)//samples)
    sample_pts = geom[0::step]
    aqi = sample_aqi_along_points(sample_pts)
    log_time("🌫️ Sample AQI", start)
    return aqi

# ─── 5) COUNT SENSITIVE ZONES ──────────────────────────────────────
def count_pois_near(lat, lng, radius_m=100, types=None):
    if types is None:
        types = ["school","hospital","shopping_mall","place_of_worship"]
    total = 0
    for t in types:
        params = {
            "location": f"{lat},{lng}",
            "radius":   radius_m,
            "type":     t,
            "key":      GOOGLE_API_KEY
        }
        j = requests.get("https://maps.googleapis.com/maps/api/place/nearbysearch/json", params=params).json()
        total += len(j.get("results", []))
        time.sleep(0.1)
    return total

def count_zones_for_pair(orig, dest, samples=5):
    start = time.time()
    geom = fetch_route_geometry(orig, dest)
    step = max(1, len(geom)//samples)
    sample_pts = geom[0::step]
    count = sum(count_pois_near(lat, lng) for lat, lng in sample_pts)
    log_time("☣️ Count Sensitive Zones", start)
    return count

# ─── 6) BUILD COST MATRIX ──────────────────────────────────────────
def build_cost_matrix(time_mat, dist_mat, aqi_mat, zone_mat, emission_rate):
    start = time.time()
    def normalize(m):
        mn, mx = np.nanmin(m), np.nanmax(m)
        return (m - mn) / (mx - mn + 1e-9)
    nt = normalize(time_mat)
    na = normalize(aqi_mat)
    nz = normalize(zone_mat)
    ne = normalize((dist_mat/1000) * emission_rate)
    w = WEIGHTS
    log_time("🧮 Build Cost Matrix", start)
    return w['time']*nt + w['aqi']*na + w['zones']*nz + w['emissions']*ne

# ─── 7) TSP SOLVER ─────────────────────────────────────────────────
def solve_tsp(cost_matrix, return_to_depot=True):
    start = time.time()
    N = cost_matrix.shape[0]
    mgr = pywrapcp.RoutingIndexManager(N, 1, 0)
    routing = pywrapcp.RoutingModel(mgr)
    def cb(i, j):
        return int(cost_matrix[mgr.IndexToNode(i), mgr.IndexToNode(j)] * 1000)
    idx = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(idx)
    if not return_to_depot:
        routing.SetFixedCostOfAllVehicles(0)
        routing.AddDimension(idx, 0, 1, False, "NoReturn")
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sol = routing.SolveWithParameters(params)
    route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        route.append(mgr.IndexToNode(index))
        index = sol.Value(routing.NextVar(index))
    route.append(mgr.IndexToNode(index))
    log_time("📦 TSP Solve", start)
    return route

# ─── 8) MAIN PLANNING FUNCTION ─────────────────────────────────────
def plan_eco_batch(orders, depot, vehicle_type="Petrol"):
    total_start = time.time()
    clusters = cluster_orders(orders)
    if not clusters:
        return []
    batch = clusters[0]
    nodes = [depot] + batch

    time_mat, dist_mat = fetch_google_matrix(nodes)

    print("⚙️ Starting AQI + Zone matrix computation")
    aqi_zone_start = time.time()
    N = len(nodes)
    aqi_mat  = np.zeros((N,N))
    zone_mat = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i == j: continue
            aqi_mat[i,j]  = sample_aqi_for_pair(nodes[i], nodes[j])
            zone_mat[i,j] = count_zones_for_pair(nodes[i], nodes[j])
    log_time("📊 AQI + Zone matrix computed", aqi_zone_start)

    cost_mat = build_cost_matrix(
        time_mat, dist_mat, aqi_mat, zone_mat,
        EMISSION_RATES.get(vehicle_type, 0)
    )

    order = solve_tsp(cost_mat, return_to_depot=True)
    log_time("🚀 Total route optimization", total_start)
    print("Visit order:", order)
    return [nodes[i] for i in order]

# ─── 9) DEMO ───────────────────────────────────────────────────────
if __name__ == "__main__":
    depot = {"lat": 12.9716, "lng": 77.5946}
    orders = [
        {"order_id":1, "lat": 12.9721, "lng": 77.5950},
        {"order_id":2, "lat": 12.9702, "lng": 77.5935},
        {"order_id":3, "lat": 12.9698, "lng": 77.5961},
    ]
    route = plan_eco_batch(orders, depot, vehicle_type="Petrol")
    print("📍 Eco‑friendly delivery sequence:")
    for stop in route:
        print(stop)
