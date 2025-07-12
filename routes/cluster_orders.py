from flask import Blueprint, request, jsonify, abort
import numpy as np
from sklearn.cluster import DBSCAN

cluster_bp = Blueprint("cluster_orders", __name__)

def cluster_orders(ords, eps_km=1.0, min_samples=1):
    coords = np.array([[o['lat'], o['lng']] for o in ords])
    labels = DBSCAN(eps=eps_km/111, min_samples=min_samples).fit(coords).labels_
    clusters = {}
    for o, lab in zip(ords, labels):
        if lab < 0: continue
        clusters.setdefault(lab, []).append(o)
    return list(clusters.values())

@cluster_bp.route("/api/cluster_orders", methods=["POST"])
def cluster_orders_route():
    data = request.json
    orders = data.get("orders")
    if not orders:
        return abort(400, "Missing 'orders' in request body.")
    clusters = cluster_orders(orders)
    return jsonify(clusters), 200
