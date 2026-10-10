from pathlib import Path

import pandas as pd

from simulation.engine import run_simulation
from simulation.strategies import STRATEGIES
from simulation.metrics import avoided_vs, summarize_run
from simulation.oracle import make_oracle_strategy


PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("data/results")
GRID_COLS = ["grid_mild", "grid_moderate", "grid_severe", "grid_extreme"]


if __name__ == "__main__":
    state = pd.read_csv(PROCESSED_DIR / "state_table.csv", parse_dates=["timestamp"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = {}
    for grid_col in GRID_COLS:
        strategies = {
            **STRATEGIES,
            "oracle": make_oracle_strategy(state, grid_col),
            "oracle_no_precharge": make_oracle_strategy(state, grid_col, precharge_h=0),
        }
        for name, strategy in strategies.items():
            res = run_simulation(state, strategy, grid_col=grid_col)
            res.to_csv(RESULTS_DIR / f"{name}__{grid_col}.csv", index=False)
            rows[(grid_col.replace("grid_", ""), name)] = summarize_run(res)

    summary = pd.DataFrame(rows).T
    summary.index.names = ["scenario", "strategy"]
    summary.to_csv(RESULTS_DIR / "summary.csv")

    show = [
        "gen_hours",
        "fuel_l",
        "total_cost_ngn",
        "battery_cycles",
        "unserved_kwh",
        "solar_utilization_pct",
        "renewable_share_pct",
    ]
    print(summary[show].astype(float).round(1).to_string())
    print("\nAvoided vs reactive (positive = better than reactive):")
    print(avoided_vs(summary.astype(float), "reactive").round(0).to_string())
    print("\nSaved hourly results and summary.csv to", RESULTS_DIR)
