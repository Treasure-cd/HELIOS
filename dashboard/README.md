# Helios Local Dashboard

A lightweight, purely local dashboard. Backend uses FastAPI/Pandas to serve computed JSON. Frontend is a tiny React/Vite app generating pure SVG visualisations. 

## Requirements
- Node.js (v18+)
- Python 3.10+

## 1. Start the Server
Open a terminal, navigate to the `dashboard/server` folder, install the python dependencies, and run:
```bash
pip install -r requirements.txt
# Specify where your results folder is if it's not the default `../data/results`
export RESULTS_DIR="../../data/results"
export RESULTS_V1_DIR="../../data/results_v1"

uvicorn main:app --reload --port 8000
