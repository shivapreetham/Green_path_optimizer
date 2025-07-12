from flask import Blueprint, request, jsonify, abort
from statistics import mean
from haversine import haversine

store_bp = Blueprint("find_store", __name__)

def compute_center(cluster):
    lat = mean(p["lat"] for p in cluster)
    lng = mean(p["lng"] for p in cluster)
    return {"lat": lat, "lng": lng}

@store_bp.route("/api/find_nearest_store", methods=["POST"])
def find_nearest_store_route():
    data = request.json
    cluster = data.get("cluster")
    stores = data.get("stores")

    if not cluster or not stores:
        return abort(400, "Missing 'cluster' or 'stores' in request body.")

    center = compute_center(cluster)
    nearest = min(stores, key=lambda s: haversine(
        (center["lat"], center["lng"]),
        (s["lat"], s["lng"])
    ))
    return jsonify(nearest), 200
