# ⚙️ Green_Path_Optimizer

A FastAPI microservice that powers the eco‐route calculations and batching logic for the Green‑Path platform. It ingests warehouse and order data, applies a multi‑factor optimization (distance, emissions, AQI, sensitive‑zone avoidance), and returns both **eco** and **naïve** routes with comparative metrics, plus suggested delivery batches.

---

![PyPI Version](https://img.shields.io/pypi/v/fastapi) ![License](https://img.shields.io/badge/license-MIT-blue)

---

## 🎯 Problem Statement

Modern e‑commerce and logistics systems often:

- **Ignore environmental cost** → high CO₂ emissions  
- **Lack customer awareness** → no incentive for green choices  
- **Use non‑optimized routing** → longer distances, higher fuel use  
- **Fail to batch intelligently** → redundant trips  

**Green_Path_Optimizer** solves this by providing:

1. **Eco‑optimized routes** minimizing emissions, AQI exposure, and sensitive‑zone traversal.  
2. **Naïve baseline routes** for direct distance/time comparison.  
3. **Smart batching suggestions** grouping spatially/temporally close orders.

---

## 💡 Key Features

- **Multi‑Factor Eco Route**  
  - Real‑time traffic & distance (Google Distance Matrix API)  
  - Vehicle emission profiles (EV, Petrol, Diesel in g/km)  
  - AQI sampling (PM2.5 + PM10 via Open‑Meteo)  
  - Sensitive‑zone avoidance (hospitals, schools via OSM Overpass)  
  - Congestion & route‑geometry penalties  
  - Solved with Google OR‑Tools TSP solver  

- **Naïve Baseline Route**  
  - Standard shortest‑path round‑trip for baseline metrics  

- **Batching Engine**  
  - Clusters orders by proximity & time window  
  - Fallback to individual routes when beneficial  

- **Comprehensive Metrics**  
  - CO₂ saved (kg)  
  - AQI exposure reduction  
  - Distance & time comparisons  
  - Sensitive‑zone count avoided  

---

## 🛠 Tech Stack

- **Framework:** FastAPI & Uvicorn  
- **Language:** Python 3.10+  
- **Optimization:** Google OR‑Tools (TSP)  
- **Data Sources:**  
  - Google Maps APIs (Distance, Directions)  
  - Open‑Meteo AQI API  
  - OpenStreetMap Overpass API  
- **Concurrency:** Asyncio + Aiohttp  
- **Cache:** Redis (via aioredis)  
- **Containerization:** Docker & Docker Compose  
- **Testing:** Pytest  

---

## 🚀 Getting Started

### 1. Clone & Setup

```bash
git clone https://github.com/HiiiiiPritam/Green_path_optimizer.git
cd Green_path_optimizer
