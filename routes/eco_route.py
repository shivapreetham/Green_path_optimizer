import os, time, requests, numpy as np, polyline, json, aiohttp, asyncio
from flask import Blueprint, request, jsonify
from sklearn.cluster import DBSCAN
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from shapely.geometry import LineString
import redis
from dotenv import load_dotenv

eco_route_bp = Blueprint("eco_route", __name__)

# Load .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
rdb = redis.Redis.from_url(os.environ["REDIS_URL"])
REDIS_TTL = 86400
EMISSION_RATES = {"EV": 0, "Petrol": 120, "Diesel": 180}
WEIGHTS = {"aqi": 0.4, "zones": 0.3, "emissions": 0.2, "time": 0.1}

def log_duration(label):
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def wrapper(*args, **kwargs):
                print(f"🚀 Starting: {label}")
                start = time.time()
                result = await func(*args, **kwargs)
                end = time.time()
                print(f"✅ Finished: {label} | ⏱️ Duration: {end - start:.2f}s")
                return result
            return wrapper
        else:
            def wrapper(*args, **kwargs):
                print(f"🚀 Starting: {label}")
                start = time.time()
                result = func(*args, **kwargs)
                end = time.time()
                print(f"✅ Finished: {label} | ⏱️ Duration: {end - start:.2f}s")
                return result
            return wrapper
    return decorator

@log_duration("🧭 fetch_matrix")
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
    t, d = np.zeros((N, N)), np.zeros((N, N))
    for i, row in enumerate(resp["rows"]):
        for j, el in enumerate(row["elements"]):
            t[i][j] = el["duration_in_traffic"]["value"]
            d[i][j] = el["distance"]["value"]
    return t, d

@log_duration("🛣️ fetch_geometry")
def fetch_geometry(a, b):
    key = f"geom:{a['lat']},{a['lng']}-{b['lat']},{b['lng']}"
    if cached := rdb.get(key): return json.loads(cached)
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
    return pts

@log_duration("🌫️ fetch_aqi")
async def fetch_aqi(session, lat, lng):
    try:
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
            if "hourly" not in j or "pm2_5" not in j["hourly"] or "pm10" not in j["hourly"]:
                return 50.0
            pm25 = j["hourly"]["pm2_5"][0]
            pm10 = j["hourly"]["pm10"][0]
            return (pm25 + pm10) / 2
    except:
        return 50.0

@log_duration("🌫️ async_sample_aqi")
async def async_sample_aqi(a, b, samples=5):
    key = f"aqi:{a['lat']},{a['lng']}-{b['lat']},{b['lng']}"
    if cached := rdb.get(key): return float(cached)
    pts = fetch_geometry(a, b)
    step = max(1, len(pts) // samples)
    sample_pts = pts[0::step]
    async with aiohttp.ClientSession() as session:
        vals = await asyncio.gather(*[fetch_aqi(session, lat, lng) for lat, lng in sample_pts])
    avg = float(np.mean(vals))
    rdb.setex(key, REDIS_TTL, avg)
    return avg

@log_duration("☣️ async_count_zones")
async def async_count_zones(a, b):
    key = f"zones:{a['lat']},{a['lng']}-{b['lat']},{b['lng']}"
    if cached := rdb.get(key): return int(cached)
    pts = fetch_geometry(a, b)
    poly = LineString([(lng, lat) for lat, lng in pts]).buffer(0.0045)
    minx, miny, maxx, maxy = poly.bounds
    query = f"""
    [out:json][timeout:25];
    (
        node["amenity"]({miny},{minx},{maxy},{maxx});
        way["amenity"]({miny},{minx},{maxy},{maxx});
        rel["amenity"]({miny},{minx},{maxy},{maxx});
    ); out center;
    """
    res = requests.get("https://overpass-api.de/api/interpreter", params={"data": query}).json()
    count = len({(el["type"], el["id"]) for el in res.get("elements", [])})
    rdb.setex(key, REDIS_TTL, count)
    return count

@log_duration("🧠 compute_all_pairs")
async def compute_all_pairs(nodes):
    N = len(nodes)
    aqi_mat, zone_mat = np.zeros((N, N)), np.zeros((N, N))
    sem = asyncio.Semaphore(5)
    async def compute(i, j):
        if i == j: return
        async with sem:
            aqi = await async_sample_aqi(nodes[i], nodes[j])
            zones = await async_count_zones(nodes[i], nodes[j])
            aqi_mat[i][j], zone_mat[i][j] = aqi, zones
    await asyncio.gather(*(compute(i, j) for i in range(N) for j in range(N)))
    return aqi_mat, zone_mat

@log_duration("💰 build_cost")
def build_cost(t, d, aqi, zones, emission_rate):
    def norm(m): mn, mx = m.min(), m.max(); return (m - mn) / (mx - mn + 1e-9)
    nt, na, nz, ne = norm(t), norm(aqi), norm(zones), norm((d / 1000) * emission_rate)
    return WEIGHTS["time"] * nt + WEIGHTS["aqi"] * na + WEIGHTS["zones"] * nz + WEIGHTS["emissions"] * ne

@log_duration("🧩 solve_tsp")
def solve_tsp(cost):
    N = cost.shape[0]
    mgr = pywrapcp.RoutingIndexManager(N, 1, 0)
    r = pywrapcp.RoutingModel(mgr)
    cb = lambda i, j: int(cost[mgr.IndexToNode(i), mgr.IndexToNode(j)] * 1000)
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

@log_duration("🪜 naive_nearest_neighbor")
def naive_nearest_neighbor(dmat):
    N = dmat.shape[0]
    visited = [False] * N
    path = [0]  # Start from warehouse
    visited[0] = True

    for _ in range(N - 1):
        last = path[-1]
        nearest = None
        nearest_dist = float("inf")
        for j in range(N):
            if not visited[j] and dmat[last][j] < nearest_dist:
                nearest = j
                nearest_dist = dmat[last][j]
        if nearest is not None:
            path.append(nearest)
            visited[nearest] = True

    path.append(0)  # Return to warehouse
    return path



@log_duration("🪜 naive_round_trip_each_order")
def naive_round_trip_each_order(src, dests, dmat, tmat):
    path = []
    order_ids = []
    total_dist = 0
    total_time = 0
    steps = []

    for i, dest in enumerate(dests):
        # From warehouse to order
        dist_to = dmat[0][i+1] / 1000
        time_to = tmat[0][i+1]
        # From order back to warehouse
        dist_back = dmat[i+1][0] / 1000
        time_back = tmat[i+1][0]

        total_dist += dist_to + dist_back
        total_time += time_to + time_back

        # Forward step
        steps.append({
            "from": src,
            "to": dest,
            "fromOrderId": None,
            "toOrderId": dest.get("orderId"),
            "distanceKm": round(dist_to, 2),
            "durationSec": int(time_to),
            "polyline": fetch_geometry(src, dest)
        })
        # Return step
        steps.append({
            "from": dest,
            "to": src,
            "fromOrderId": dest.get("orderId"),
            "toOrderId": None,
            "distanceKm": round(dist_back, 2),
            "durationSec": int(time_back),
            "polyline": fetch_geometry(dest, src)
        })

        path.append(src)  # Starting point
        path.append(dest)  # Destination
        path.append(src)  # Return
        order_ids.extend([None, dest.get("orderId"), None])

    return path, order_ids, steps, total_dist, total_time


@eco_route_bp.route("/api/eco_route", methods=["POST"])
@log_duration("📦 /api/eco_route handler")
def eco_route():
    data = request.json
    src = data["warehouse"]
    dests = data["orders"]
    vehicle_type = data.get("vehicle", "Petrol")
    emission = EMISSION_RATES.get(vehicle_type, 120)

    nodes = [src] + dests
    order_ids = [None] + [o.get("orderId") for o in dests]

    tmat, dmat = fetch_matrix(nodes)
    aqi_mat, zone_mat = asyncio.run(compute_all_pairs(nodes))

    # 🌱 Eco route computation
    cost_eco = build_cost(tmat, dmat, aqi_mat, zone_mat, emission)
    order_eco = solve_tsp(cost_eco)
    route_eco = [nodes[i] for i in order_eco]
    route_order_ids = [order_ids[i] for i in order_eco]

    steps = []
    total_dist = total_time = total_co2 = total_zones = 0
    total_aqi = []

    for i in range(len(route_eco) - 1):
        a, b = route_eco[i], route_eco[i + 1]
        idx_a, idx_b = order_eco[i], order_eco[i + 1]
        dist = dmat[idx_a][idx_b] / 1000
        time_sec = tmat[idx_a][idx_b]
        aqi = aqi_mat[idx_a][idx_b]
        zones = zone_mat[idx_a][idx_b]
        co2 = dist * emission

        steps.append({
            "from": a,
            "to": b,
            "fromOrderId": route_order_ids[i],
            "toOrderId": route_order_ids[i + 1],
            "distanceKm": round(dist, 2),
            "durationSec": int(time_sec),
            "aqi": round(aqi, 2),
            "zoneCount": int(zones),
            "co2g": round(co2, 2),
            "polyline": fetch_geometry(a, b)
        })

        total_dist += dist
        total_time += time_sec
        total_co2 += co2
        total_zones += zones
        total_aqi.append(aqi)

    # 🚚 Naive round-trip route
    # 🚚 Naive round-trip route
    naive_path, naive_order_ids, naive_steps, naive_distance_km, naive_duration_sec = naive_round_trip_each_order(
        src, dests, dmat, tmat
    )

    # ➕ Add AQI and Zone data to naive steps
    async def enrich_naive_steps():
        sem = asyncio.Semaphore(5)
        async with aiohttp.ClientSession() as session:
            tasks = []
            for step in naive_steps:
                a = step["from"]
                b = step["to"]
                async def enrich_step(step=step, a=a, b=b):
                    async with sem:
                        try:
                            aqi = await fetch_aqi(session, a["lat"], a["lng"])
                            zones = await async_count_zones(a, b)
                            step["aqi"] = round(aqi, 2)
                            step["zoneCount"] = zones
                        except Exception as e:
                            step["aqi"] = 50.0
                            step["zoneCount"] = 0
                tasks.append(enrich_step())
            await asyncio.gather(*tasks)

    asyncio.run(enrich_naive_steps())

    naive_co2 = naive_distance_km * emission
    naive_zones = sum(step.get("zoneCount", 0) for step in naive_steps if "zoneCount" in step)

    avg_aqi = round(sum(total_aqi) / len(total_aqi), 2)
    aqi_rating = (
        "Good" if avg_aqi <= 50 else
        "Moderate" if avg_aqi <= 100 else
        "Unhealthy" if avg_aqi <= 150 else
        "Very Unhealthy" if avg_aqi <= 200 else
        "Hazardous"
    )

    # 🔥 Final impact summary
    impact_summary = {
        "ecoCO2g": round(total_co2, 2),
        "naiveCO2g": round(naive_co2, 2),
        "ecoZones": int(total_zones),
        "naiveZones": int(naive_zones),
        "avgAQI": avg_aqi,
        "AQIRating": aqi_rating
    }

    response_data = [{
        "eco": {
            "route": [[p["lat"], p["lng"]] for p in route_eco],
            "stepOrderIds": route_order_ids,
            "steps": steps
        },
        "naive": {
            "route": [[p["lat"], p["lng"]] for p in naive_path],
            "stepOrderIds": naive_order_ids,
            "steps": naive_steps
        },
        "impactSummary": impact_summary,
        "efficiency": {
            "optimizedVsUnoptimizedKm": {
                "optimized": round(total_dist, 2),
                "naive": round(naive_distance_km, 2),
                "savedKm": round(naive_distance_km - total_dist, 2),
                "savedPercent": round(100 * (naive_distance_km - total_dist) / max(naive_distance_km, 0.01), 2)
            },
            "optimizedVsUnoptimizedTime": {
                "optimized": int(total_time),
                "naive": int(naive_duration_sec),
                "savedMin": round((naive_duration_sec - total_time) / 60, 2)
            }
        },
        "meta": {
            "vehicleType": vehicle_type,
            "emissionRate": emission,
            "engine": "Google Maps + OR-Tools + AQI Zones",
            "aqiProvider": "Open-Meteo",
            "mapsCreditsUsed": len(nodes)**2 + len(nodes)
        }
    }]

    # 🖨️ Console log for debugging
    print("📊 Impact Summary:")
    print(json.dumps(impact_summary, indent=2))

    return jsonify(response_data), 200