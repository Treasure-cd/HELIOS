from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")
GRID_COLS = ["grid_mild", "grid_moderate", "grid_severe", "grid_extreme"]


def build_state_table() -> pd.DataFrame:
    """Merge solar, demand and grid into one hourly table (one row = one hour)."""
    solar = pd.read_csv(
        PROCESSED_DIR / "solar_combined.csv",
        parse_dates=["timestamp"],
        usecols=["timestamp", "ALLSKY_SFC_SW_DWN", "solar_generation_kwh"],
    ).rename(columns={"ALLSKY_SFC_SW_DWN": "irradiance_wh_m2"})

    demand = pd.read_csv(PROCESSED_DIR / "household_demand.csv", parse_dates=["timestamp"])
    grid = pd.read_csv(PROCESSED_DIR / "grid_availability.csv", parse_dates=["timestamp"])

    state = (
        demand.merge(solar, on="timestamp", how="inner", validate="one_to_one")
        .merge(grid, on="timestamp", how="inner", validate="one_to_one")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # An inner merge silently drops unmatched hours, so verify nothing was lost
    assert len(state) == len(solar) == len(demand) == len(grid), (
        f"Row mismatch: solar={len(solar)}, demand={len(demand)}, "
        f"grid={len(grid)}, merged={len(state)}"
    )
    assert state.isna().sum().sum() == 0, "NaNs in merged table"
    assert (state["solar_generation_kwh"] >= 0).all(), "negative solar generation"
    assert (state["demand_kwh"] > 0).all(), "non-positive demand"
    for col in GRID_COLS:
        assert state[col].isin([0, 1]).all(), f"{col} must be 0/1"

    return state[
        ["timestamp", "hour", "day_of_week", "is_weekend",
         "irradiance_wh_m2", "solar_generation_kwh", "demand_kwh", *GRID_COLS]
    ]


if __name__ == "__main__":
    state = build_state_table()
    out = PROCESSED_DIR / "state_table.csv"
    state.to_csv(out, index=False)

    print("Saved:", out, "| rows:", len(state))
    print(state.head(8))
    print(f"\nAnnual solar:  {state['solar_generation_kwh'].sum():.0f} kWh")
    print(f"Annual demand: {state['demand_kwh'].sum():.0f} kWh")
    print("Grid availability %:", (100 * state[GRID_COLS].mean()).round(1).to_dict())
