import os
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))

POSSIBLE_MAIN_DIRS = [
    os.getenv("RESULTS_DIR"),
    os.path.abspath(os.path.join(SERVER_DIR, "../../data/results")),
    os.path.abspath(os.path.join(SERVER_DIR, "../data/results")),
    os.path.abspath("data/results"),
]
RESULTS_DIR = next((p for p in POSSIBLE_MAIN_DIRS if p and os.path.isdir(p)), POSSIBLE_MAIN_DIRS[1])

POSSIBLE_V1_DIRS = [
    os.getenv("RESULTS_V1_DIR"),
    os.path.abspath(os.path.join(SERVER_DIR, "../../data/results_v1")),
    os.path.abspath(os.path.join(SERVER_DIR, "../data/results_v1")),
    os.path.abspath("data/results_v1"),
]
RESULTS_V1_DIR = next((p for p in POSSIBLE_V1_DIRS if p and os.path.isdir(p)), POSSIBLE_V1_DIRS[1])

print(f"[INFO] Using RESULTS_DIR: {RESULTS_DIR}")
print(f"[INFO] Using RESULTS_V1_DIR: {RESULTS_V1_DIR}")

def sanitize_records(df: pd.DataFrame):
    """Safely converts NaN/Inf to None so FastAPI emits valid JSON null."""
    if df.empty:
        return []
    clean_df = df.astype(object).where(pd.notnull(df), None)
    return clean_df.to_dict(orient="records")

def load_csv(filename: str) -> pd.DataFrame:
    main_path = os.path.join(RESULTS_DIR, filename)
    v1_path = os.path.join(RESULTS_V1_DIR, filename)

    df_main = pd.read_csv(main_path) if os.path.exists(main_path) else pd.DataFrame()
    df_v1 = pd.read_csv(v1_path) if os.path.exists(v1_path) else pd.DataFrame()

    if df_main.empty and df_v1.empty:
        return pd.DataFrame()
    if df_main.empty: return df_v1
    if df_v1.empty: return df_main

    # Merge v1 rows where strategy is absent from main
    main_strategies = set(df_main.get('strategy', []))
    if 'strategy' in df_v1.columns:
        df_v1 = df_v1[~df_v1['strategy'].isin(main_strategies)]

    return pd.concat([df_main, df_v1], ignore_index=True)

@app.get("/api/meta")
def get_meta():
    meta = {"patterns": ["uniform", "evening"], "scenarios": [], "pv_kws": [], "battery_kwhs": [], "strategies": []}
    for pattern in meta["patterns"]:
        df = load_csv(f"sweep_{pattern}_summary.csv")
        if not df.empty:
            meta["scenarios"].extend(df["scenario"].dropna().unique())
            meta["pv_kws"].extend(df["pv_kw"].dropna().unique())
            meta["battery_kwhs"].extend(df["battery_kwh"].dropna().unique())
            meta["strategies"].extend(df["strategy"].dropna().unique())

    return {k: sorted(list(set(v))) for k, v in meta.items()}

@app.get("/api/summary")
def get_summary(source: str, pattern: str, scenario: str, pv_kw: float, battery_kwh: float):
    df = load_csv(f"{source}_{pattern}_summary.csv")
    if df.empty: return []
    filtered = df[(df.scenario == scenario) & (np.isclose(df.pv_kw.astype(float), float(pv_kw))) & (np.isclose(df.battery_kwh.astype(float), float(battery_kwh)))]
    return sanitize_records(filtered)

@app.get("/api/split")
def get_split(pattern: str, scenario: str, pv_kw: float, battery_kwh: float):
    df = load_csv(f"sweep_{pattern}_runs.csv")
    if df.empty: return []
    filtered = df[(df.scenario == scenario) & (np.isclose(df.pv_kw.astype(float), float(pv_kw))) & (np.isclose(df.battery_kwh.astype(float), float(battery_kwh)))]
    if filtered.empty or "reactive" not in filtered["strategy"].values: return []

    pivot = filtered.pivot_table(index='seed', columns='strategy', values='total_cost_ngn')
    if 'reactive' not in pivot.columns: return []

    saved_pct = pd.DataFrame(index=pivot.index)
    for col in pivot.columns:
        saved_pct[col] = (pivot['reactive'] - pivot[col]) / pivot['reactive'] * 100

    tune = saved_pct[saved_pct.index < 10].agg(['mean', 'std']).T
    test = saved_pct[saved_pct.index >= 10].agg(['mean', 'std']).T

    res = []
    for strat in pivot.columns:
        res.append({
            "strategy": strat,
            "tune_mean": tune.loc[strat, 'mean'] if strat in tune.index else None,
            "tune_std": tune.loc[strat, 'std'] if strat in tune.index else None,
            "test_mean": test.loc[strat, 'mean'] if strat in test.index else None,
            "test_std": test.loc[strat, 'std'] if test.index.notna().any() and strat in test.index else None
        })
    return sanitize_records(pd.DataFrame(res))

@app.get("/api/grid")
def get_grid(pattern: str, scenario: str, strategy: str, metric: str = "saved_pct_mean"):
    df = load_csv(f"sweep_{pattern}_summary.csv")
    if df.empty: return []
    filtered = df[(df.scenario == scenario) & (df.strategy == strategy)]
    if filtered.empty or metric not in filtered.columns: return []

    grid = filtered[['pv_kw', 'battery_kwh', metric]].rename(columns={metric: 'value'})
    return sanitize_records(grid)
