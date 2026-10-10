import argparse
import itertools
import os
from pathlib import Path
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from simulation.grid import SCENARIOS, generate_grid_availability
from simulation import config as cfg
from simulation.engine import Battery, run_simulation
from simulation.metrics import summarize_run
from simulation.oracle import make_oracle_strategy
from simulation.strategies import STRATEGIES
from simulation.risk import load_model
from simulation.strategies import STRATEGIES, make_helios_predictive_strategy

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("data/results")

# Fixed in advance (do not edit after seeing results)
SWEEP_PV_KW = (2.0, 3.0, 5.0)
SWEEP_BATTERY_KWH = (2.5, 5.0)
PREDICTIVE = {"helios_p10": (0.10, True), "helios_p13": (0.13, True), "helios_q10": (0.10, False)}
SWEEP_STRATEGIES = ("reactive", "helios_w50", "helios_p10", "helios_q10")
BASELINE = "reactive"

PATTERNS = {
    "uniform": None,
    "evening": [3.0 if 17 <= h <= 22 else 1.0 for h in range(24)],
}

_CTX = {}


def _init_worker(state, names, weights):
    global _CTX
    _CTX = {"state": state, "names": names, "weights": weights}


def _run_task(task):
    """One (PV, battery, scenario, seed) combination, all strategies."""
    pv_kw, battery_kwh, scenario, seed = task
    state, names, weights = _CTX["state"], _CTX["names"], _CTX["weights"]

    solar = (state["irradiance_wh_m2"] / 1000 * pv_kw * cfg.SYSTEM_EFFICIENCY).to_numpy()
    grid = generate_grid_availability(
        state["timestamp"], seed=seed, start_hour_weights=weights, **SCENARIOS[scenario]
    )["grid"].to_numpy()
    run_state = state.assign(grid_run=grid, solar_run=solar)

    rows = []
    for name in names:
        if name == "oracle":
            strategy = make_oracle_strategy(run_state, "grid_run", solar_col="solar_run")
        elif name in PREDICTIVE:
            pattern = "evening" if weights is not None else "uniform"
            thr, pre = PREDICTIVE[name]
            strategy = make_helios_predictive_strategy(load_model(pattern), thr, precharge=pre)
        else:
            strategy = STRATEGIES[name]

        res = run_simulation(
            run_state,
            strategy,
            grid_col="grid_run",
            solar_col="solar_run",
            battery=Battery(capacity_kwh=battery_kwh),
            check_each_hour=False,
        )
        m = summarize_run(res)
        rows.append({
            "pv_kw": pv_kw,
            "battery_kwh": battery_kwh,
            "scenario": scenario,
            "seed": seed,
            "strategy": name,
            "gen_hours": m["gen_hours"],
            "fuel_cost_ngn": m["fuel_cost_ngn"],
            "total_cost_ngn": m["total_cost_ngn"],
            "unserved_kwh": m["unserved_kwh"],
            "grid_availability_pct": 100 * grid.mean(),
        })
    return rows


def run_all(state, pv_kws, battery_kwhs, seeds, strategy_names, start_hour_weights=None):
    tasks = list(itertools.product(pv_kws, battery_kwhs, SCENARIOS.keys(), seeds))
    workers = max(1, (os.cpu_count() or 1) - 1)
    print(f"{len(tasks)} tasks x {len(strategy_names)} strategies on {workers} worker(s)")

    rows, done = [], 0
    with Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(state, tuple(strategy_names), start_hour_weights),
    ) as pool:
        for chunk in pool.imap_unordered(_run_task, tasks):
            rows.extend(chunk)
            done += 1
            if done % 4 == 0 or done == len(tasks):
                print(f"\r  {done}/{len(tasks)} tasks done", end="", flush=True)
    print()

    runs = pd.DataFrame(rows)
    scen_order = {k: i for i, k in enumerate(SCENARIOS)}
    strat_order = {k: i for i, k in enumerate(strategy_names)}
    runs["_s"] = runs["scenario"].map(scen_order)
    runs["_k"] = runs["strategy"].map(strat_order)
    runs = runs.sort_values(["pv_kw", "battery_kwh", "_s", "seed", "_k"]).drop(columns=["_s", "_k"])
    return runs.reset_index(drop=True)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Paired comparison against the baseline, seed by seed."""
    keys = ["pv_kw", "battery_kwh", "scenario", "seed"]
    base = runs[runs.strategy == BASELINE].set_index(keys)[["gen_hours", "total_cost_ngn"]]
    runs = runs.join(base, on=keys, rsuffix="_base")
    runs["cost_saved_ngn"] = runs["total_cost_ngn_base"] - runs["total_cost_ngn"]
    runs["cost_saved_pct"] = 100 * runs["cost_saved_ngn"] / runs["total_cost_ngn_base"]
    runs["gen_hours_avoided"] = runs["gen_hours_base"] - runs["gen_hours"]
    runs["beats_baseline"] = runs["cost_saved_ngn"] > 0

    g = runs.groupby(["pv_kw", "battery_kwh", "scenario", "strategy"], sort=False)
    out = g.agg(
        cost_ngn=("total_cost_ngn", "mean"),
        gen_hours=("gen_hours", "mean"),
        gen_avoided=("gen_hours_avoided", "mean"),
        saved_pct_mean=("cost_saved_pct", "mean"),
        saved_pct_std=("cost_saved_pct", "std"),
        seeds_beating_baseline=("beats_baseline", "mean"),
        unserved_kwh=("unserved_kwh", "sum"),
    )
    out["seeds_beating_baseline"] = (100 * out["seeds_beating_baseline"]).round(0)
    return out.round(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--sweep", action="store_true", help="PV x battery sizing sweep")
    ap.add_argument(
        "--pattern",
        choices=list(PATTERNS),
        default="uniform",
        help="when outages tend to start",
    )
    args = ap.parse_args()

    state = pd.read_csv(PROCESSED_DIR / "state_table.csv", parse_dates=["timestamp"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        pv, bat, names = SWEEP_PV_KW, SWEEP_BATTERY_KWH, SWEEP_STRATEGIES
        n_seeds, tag = args.seeds or 10, "sweep"
    else:
        pv, bat, names = (
            (cfg.PV_CAPACITY_KW,),
            (cfg.BATTERY_CAPACITY_KWH,),
            tuple(STRATEGIES) + ("oracle",),
        )
        n_seeds, tag = args.seeds or 30, "seeds"

    t0 = time.time()
    runs = run_all(state, pv, bat, range(n_seeds), names, PATTERNS[args.pattern])
    tag = f"{tag}_{args.pattern}"
    runs.to_csv(RESULTS_DIR / f"{tag}_runs.csv", index=False)
    summary = summarize(runs)
    summary.to_csv(RESULTS_DIR / f"{tag}_summary.csv")

    pd.set_option("display.width", 200)
    print(summary.to_string())
    print(f"\n{len(runs)} runs in {time.time() - t0:.0f}s. Saved to {RESULTS_DIR}/{tag}_*.csv")
    if (runs["unserved_kwh"] > 0).any():
        print("WARNING: some runs left energy unserved, check unserved_kwh")
