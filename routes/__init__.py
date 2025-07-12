from .eco_route import eco_route_bp
from .cluster_orders import cluster_bp
from .find_store import store_bp

def register_routes(app):
    app.register_blueprint(eco_route_bp)
    app.register_blueprint(cluster_bp)
    app.register_blueprint(store_bp)
