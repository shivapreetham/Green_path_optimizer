from flask import Flask, render_template
from flask_cors import CORS               # ⬅️ Add this line
from routes import register_routes
import os
from dotenv import load_dotenv

load_dotenv()  # Load env here

app = Flask(__name__)
CORS(app)      # ⬅️ Enable CORS for all routes (allow all origins)
# CORS(app, origins=["http://localhost:3000"])  # ⬅️ Safer: restrict to frontend origin

register_routes(app)

@app.route("/")
def index():
    return render_template("index.html", gmaps_key=os.getenv("GOOGLE_API_KEY"))

if __name__ == "__main__":
    app.run(debug=True)
