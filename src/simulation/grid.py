import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

SCENARIOS = {
    "mild":     {"events_per_year": 60,  "mean_duration_h": 2.0},
    "moderate": {"events_per_year": 120, "mean_duration_h": 3.0},
    "severe":   {"events_per_year": 240, "mean_duration_h": 4.0},
    "extreme":  {"events_per_year": 700, "mean_duration_h": 4.0},
}


def generate_grid_availability(
    timestamps,
    events_per_year: int = 120,
    mean_duration_h: float = 3.0,
    duration_sigma: float = 0.6,
    max_duration_h: int = 24,
    start_hour_weights=None,
    seed: int = 7,
) -> pd.DataFrame:
    """Return a DataFrame with 'timestamp' and 'grid' (1 = available, 0 = outage)."""
    rng = np.random.default_rng(seed)
    n = len(timestamps)
    hours = pd.DatetimeIndex(timestamps).hour

    # Number of events: Poisson around the yearly average
    n_events = rng.poisson(events_per_year)

    # Start positions: uniform or weighted by hour of day
    if start_hour_weights is None:
        p = None
    else:
        w = np.asarray(start_hour_weights, dtype=float)[hours]
        p = w / w.sum()
    starts = rng.choice(n, size=n_events, p=p)

    # Durations: lognormal (mostly short, occasionally long), at least 1 h
    mu = np.log(mean_duration_h) - duration_sigma**2 / 2
    durations = rng.lognormal(mu, duration_sigma, size=n_events)
    durations = np.clip(np.ceil(durations), 1, max_duration_h).astype(int)

    grid = np.ones(n, dtype=int)
    for s, d in zip(starts, durations):
        grid[s : s + d] = 0

    return pd.DataFrame({"timestamp": timestamps, "grid": grid})


def generate_all_scenarios(timestamps, scenarios=None, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Generate grid availability DataFrames for all configured scenarios."""
    scenarios = scenarios or SCENARIOS
    return {
        name: generate_grid_availability(timestamps, seed=seed, **params)
        for name, params in scenarios.items()
    }


def outage_summary(grid_df: pd.DataFrame) -> dict:
    """Calculate key reliability metrics from a grid availability series."""
    off = grid_df["grid"] == 0
    episode = (off & ~off.shift(fill_value=False)).cumsum()
    lengths = off.groupby(episode[off]).sum()


    return {
        "availability_%": round(100 * grid_df["grid"].mean(), 1),
        "outage_hours": int(off.sum()),
        "episodes": len(lengths),
        "mean_len_h": round(lengths.mean(), 2) if len(lengths) > 0 else 0.0,
        "longest_h": int(lengths.max()) if len(lengths) > 0 else 0,
    }

if __name__ == "__main__":
    PROCESSED_DIR = Path("data/processed")
    demand_path = PROCESSED_DIR / "household_demand.csv"

    if not demand_path.exists():
        raise FileNotFoundError(
            "household_demand.csv not found. Run generate_demand.py first."
        )

    # Reuse demand's timestamps so the grid column lines up exactly
    timestamps = pd.read_csv(demand_path, parse_dates=["timestamp"])["timestamp"]

    scenarios = generate_all_scenarios(timestamps)

    combined = pd.DataFrame({"timestamp": timestamps})
    for name, g in scenarios.items():
        combined[f"grid_{name}"] = g["grid"].values
        print(f"{name:<9}", outage_summary(g))

    combined.to_csv(PROCESSED_DIR / "grid_availability.csv", index=False)
    print("Saved:", PROCESSED_DIR / "grid_availability.csv")
