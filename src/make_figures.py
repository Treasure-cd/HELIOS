"""
Make every figure for the documentation, saved as PNG files (no display needed).

    python src/make_figures.py

Needs data/processed/state_table.csv. Figures that depend on the multi-seed and
sizing-sweep results (python src/run_sweep.py ...) are made only if those files exist.
Everything is written to figures/.
"""
import matplotlib

matplotlib.use("Agg")   # draw to files, not to a window (works on servers/WSL/SSH)

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simulation.engine import run_simulation
from simulation.metrics import summarize_run
from simulation.oracle import make_oracle_strategy
from simulation.strategies import STRATEGIES

PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "data" / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 170, "axes.grid": True, "grid.alpha": 0.3})


def save(fig, name):
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("saved", path.relative_to(ROOT))


# ---------------------------------------------------------------- input data
def fig_solar(state):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    hourly = state.groupby("hour")["solar_generation_kwh"].mean()
    axes[0].plot(hourly.index, hourly.values, marker="o")
    axes[0].set(title="Average solar generation by hour of day (5 kW PV)",
                xlabel="hour (local clock time)", ylabel="kWh", xticks=range(0, 24, 2))
    monthly = state.groupby(state["timestamp"].dt.month)["solar_generation_kwh"].sum()
    axes[1].bar(monthly.index, monthly.values)
    axes[1].set(title="Total solar generation by month", xlabel="month", ylabel="kWh", xticks=range(1, 13))
    save(fig, "fig01_solar_generation.png")


def fig_demand(state):
    profile = state.groupby(["is_weekend", "hour"])["demand_kwh"].mean().unstack(0)
    profile.columns = ["Weekday", "Weekend"]
    fig, ax = plt.subplots(figsize=(8, 4))
    profile.plot(ax=ax, marker="o")
    ax.set(title="Synthetic household demand: average hourly profile",
           xlabel="hour", ylabel="kWh", xticks=range(0, 24, 2))
    save(fig, "fig02_household_demand.png")


def fig_grid(state, grid_cols):
    fig, axes = plt.subplots(len(grid_cols), 1, figsize=(11, 1.6 * len(grid_cols) + 1), sharex=True)
    first = state.head(24 * 14)
    for ax, col in zip(np.atleast_1d(axes), grid_cols):
        ax.step(first["timestamp"], first[col], where="post")
        ax.set_yticks([0, 1], ["off", "on"])
        ax.set_ylabel(f"{col.replace('grid_', '')}\n{100 * state[col].mean():.1f}% up", fontsize=9)
        ax.grid(False)
    axes[0].set_title("Grid availability by scenario, first 14 days")
    plt.setp(axes[-1].get_xticklabels(), rotation=30)
    save(fig, "fig03_grid_scenarios.png")


# ------------------------------------------------------------ single-year runs
def run_everything(state, grid_cols):
    runs, rows = {}, {}
    for grid_col in grid_cols:
        scenario = grid_col.replace("grid_", "")
        strategies = {**STRATEGIES, "oracle": make_oracle_strategy(state, grid_col)}
        for name, strategy in strategies.items():
            res = run_simulation(state, strategy, grid_col=grid_col, check_each_hour=False)
            runs[(scenario, name)] = res
            rows[(scenario, name)] = summarize_run(res)
    summary = pd.DataFrame(rows).T.astype(float)
    summary.index.names = ["scenario", "strategy"]
    return runs, summary


def fig_results(summary, scenario_order):
    strategy_order = list(dict.fromkeys(summary.index.get_level_values("strategy")))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, metric, title, unit in [
        (axes[0], "total_cost_ngn", "Annual operating cost", "NGN"),
        (axes[1], "gen_hours", "Generator running hours", "hours"),
    ]:
        table = summary[metric].unstack("strategy").loc[scenario_order, strategy_order]
        table.plot.bar(ax=ax, width=0.85, rot=0)
        ax.set(title=title, xlabel="grid scenario", ylabel=unit)
        ax.legend(fontsize=7, ncol=2)
    save(fig, "fig04_cost_and_generator_hours.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    table = summary["total_cost_ngn"].unstack("strategy").loc[scenario_order, [s for s in strategy_order if s != "naive"]]
    table.plot.bar(ax=ax, width=0.85, rot=0)
    ax.set(title="Annual operating cost without the naive baseline (zoomed)",
           xlabel="grid scenario", ylabel="NGN")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "fig04b_cost_zoomed.png")


def plot_day(ax, res, day, title):
    d = res[res["timestamp"].dt.date == day].reset_index(drop=True)
    bottom = np.zeros(len(d))
    for col, label in [("solar_to_load_kwh", "solar"), ("battery_to_load_kwh", "battery"),
                       ("grid_to_load_kwh", "grid"), ("generator_to_load_kwh", "generator")]:
        ax.bar(d["hour"], d[col], bottom=bottom, label=label, width=0.85)
        bottom += d[col].to_numpy()
    for h in d.loc[d["grid_available"] == 0, "hour"]:
        ax.axvspan(h - 0.5, h + 0.5, color="red", alpha=0.08, lw=0)
    ax.set(title=title, xlabel="hour of day (red shading = grid outage)", ylabel="kWh served",
           xticks=range(0, 24, 2))
    ax2 = ax.twinx()
    ax2.plot(d["hour"], d["battery_soc"], color="black", marker=".", lw=1)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("battery state of charge")
    ax2.grid(False)


def fig_worst_day(runs):
    reactive = runs[("extreme", "reactive")]
    day = reactive.groupby(reactive["timestamp"].dt.date)["generator_running"].sum().idxmax()
    peak = reactive[reactive["timestamp"].dt.date == day]["demand_kwh"].max()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    plot_day(axes[0], runs[("extreme", "reactive")], day, f"reactive, {day}")
    plot_day(axes[1], runs[("extreme", "oracle")], day, f"oracle, {day}")
    axes[0].set_ylim(0, peak * 1.2)
    axes[0].legend(loc="upper left", fontsize=8)
    save(fig, "fig05_worst_day.png")


# ------------------------------------------------------- robustness / sweep CSVs
def fig_seeds(scenario_order):
    available = {}
    for p in ["uniform", "evening"]:
        path = RESULTS_DIR / f"seeds_{p}_summary.csv"
        if path.exists():
            available[p] = pd.read_csv(path)
    if not available:
        print("skipped fig06 (no seeds_*_summary.csv, run: python src/run_sweep.py)")
        return
    keys = ["helios_r50", "helios_r70", "oracle"]
    fig, axes = plt.subplots(1, len(available), figsize=(7 * len(available), 4.5), sharey=True, squeeze=False)
    for ax, (pattern, df) in zip(axes[0], available.items()):
        for i, s in enumerate(keys):
            sub = df[df["strategy"] == s].set_index("scenario").loc[scenario_order]
            x = np.arange(len(scenario_order)) + (i - 1) * 0.27
            ax.bar(x, sub["saved_pct_mean"], width=0.27, yerr=sub["saved_pct_std"], capsize=3, label=s)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(scenario_order)), scenario_order)
        ax.set_title(f"Cost saving vs reactive across outage years ({pattern})")
        ax.set_ylabel("% cheaper than reactive (negative = costs more)")
        ax.legend(fontsize=8)
    save(fig, "fig06_seeds_cost_saving.png")


def fig_sweep():
    made = False
    for pattern in ["uniform", "evening"]:
        path = RESULTS_DIR / f"sweep_{pattern}_summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        ext = df[df["scenario"] == "extreme"]
        panels = [("reactive", "gen_hours", "reactive: generator hours", "Reds"),
                  ("helios_r50", "saved_pct_mean", "helios_r50: % cost saved vs reactive", "RdYlGn"),
                  ("oracle", "saved_pct_mean", "oracle: % cost saved vs reactive", "RdYlGn")]
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
        for ax, (strategy, metric, title, cmap) in zip(axes, panels):
            t = ext[ext["strategy"] == strategy].pivot(index="pv_kw", columns="battery_kwh", values=metric)
            lim = max(abs(t.values).max(), 1e-9)
            im = ax.imshow(t.values, cmap=cmap, aspect="auto",
                           **({"vmin": -lim, "vmax": lim} if cmap == "RdYlGn" else {}))
            ax.set_xticks(range(len(t.columns)), t.columns)
            ax.set_yticks(range(len(t.index)), t.index)
            for (i, j), v in np.ndenumerate(t.values):
                dark = (metric == "gen_hours" and v > 0.6 * t.values.max()) or (metric != "gen_hours" and abs(v) > 0.6 * lim)
                ax.text(j, i, f"{v:.0f}" if metric == "gen_hours" else f"{v:.1f}",
                        ha="center", va="center", color="white" if dark else "black")
            ax.set(title=title, xlabel="battery (kWh)", ylabel="PV (kW)")
            ax.grid(False)
        fig.suptitle(f"System-size sweep, extreme grid scenario (outage pattern: {pattern})")
        save(fig, f"fig07_sizing_sweep_{pattern}.png")
        made = True
    if not made:
        print("skipped fig07 (no sweep_*_summary.csv, run: python src/run_sweep.py --sweep)")


if __name__ == "__main__":
    state = pd.read_csv(PROCESSED_DIR / "state_table.csv", parse_dates=["timestamp"])
    grid_cols = [c for c in state.columns if c.startswith("grid_")]
    scenario_order = [c.replace("grid_", "") for c in grid_cols]

    fig_solar(state)
    fig_demand(state)
    fig_grid(state, grid_cols)

    print("running all strategies (about a minute)...")
    runs, summary = run_everything(state, grid_cols)
    summary.to_csv(RESULTS_DIR / "single_year_summary.csv")
    fig_results(summary, scenario_order)
    fig_worst_day(runs)

    fig_seeds(scenario_order)
    fig_sweep()
    print(f"\nDone. Open the files in {FIGURES_DIR}")
