import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

def generate_household_demand(
    year: int = 2025,
    noise_std: float = 0.06,
    base_load_min: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates an 8,760-hour residential electricity demand profile (kWh)

    based on a bottom-up appliance model for a modest urban Nigerian household.

    Parameters:
    -----------
    year : int
        The calendar year to generate timestamps for (non-leap: 8,760 hrs).
    noise_std : float
        Standard deviation for hourly Gaussian variation (default 6% of base).
    base_load_min : float
        Absolute floor in kWh to guarantee continuous background load (fridge/router).
    seed : int
        Random seed for reproducibility.

    Returns:
    --------
    pd.DataFrame
        DataFrame with 'timestamp', 'hour', 'day_of_week', 'is_weekend',
        and 'demand_kwh'.
    """
    rng = np.random.default_rng(seed)

    # 1. Base 24-hour weekday profile (sum = 7.35 kWh)
    weekday_base = np.array(
        [
            0.14,
            0.14,
            0.08,
            0.08,
            0.08,
            0.14,  # 00:00 - 05:00 (Night base)
            0.75,
            0.85,
            0.35,
            0.22,  # 06:00 - 09:00 (Morning peak)
            0.20,
            0.19,
            0.19,
            0.19,
            0.25,
            0.25,
            0.25,  # 10:00 - 16:00 (Midday plateau)
            0.30,
            0.45,
            0.60,
            0.50,
            0.45,
            0.30,
            0.14,  # 17:00 - 23:00 (Evening peak)
        ]
    )

    # 2. Base 24-hour weekend profile
    # People stay home: morning peak shifts later, afternoon TV/fans/chores increase
    weekend_base = np.array(
        [
            0.14,
            0.14,
            0.08,
            0.08,
            0.08,
            0.10,  # 00:00 - 05:00 (Slightly slower wake-up)
            0.25,
            0.60,
            0.80,
            0.50,  # 06:00 - 09:00 (Delayed morning peak)
            0.45,
            0.40,
            0.42,
            0.40,
            0.45,
            0.45,
            0.40,  # 10:00 - 16:00 (Occupants home: TV, fans, cooking)
            0.40,
            0.55,
            0.65,
            0.60,
            0.50,
            0.35,
            0.20,  # 17:00 - 23:00 (Slightly extended evening)
        ]
    )

    # 3. Create full hourly timestamp range
    timestamps = pd.date_range(
        start=f"{year}-01-01 00:00:00",
        end=f"{year}-12-31 23:00:00",
        freq="h",
    )

    df = pd.DataFrame({"timestamp": timestamps})
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # Monday=0, Sunday=6
    df["is_weekend"] = df["day_of_week"] >= 5

    # 4. Map the representative days
    base_demand = np.where(
        df["is_weekend"],
        weekend_base[df["hour"]],
        weekday_base[df["hour"]],
    )

    # 5. Add stochastic variation:
    # Multiplicative noise mimics real human variance without zeroing out spikes
    # e.g., N(1.0, 0.06) gives ~±12% swing 95% of the time
    noise = rng.normal(loc=1.0, scale=noise_std, size=len(df))
    stochastic_demand = base_demand * noise

    # Add occasional short high-power spikes (e.g. water pumping or laundry ironing)
    # 3% chance in any active hour (07:00-21:00) of a +0.25 to +0.50 kWh bump
    daytime_mask = (df["hour"] >= 7) & (df["hour"] <= 21)
    random_events = (
        (rng.random(len(df)) < 0.03) & daytime_mask
    ) * rng.uniform(0.20, 0.45, len(df))

    total_demand = stochastic_demand + random_events

    # 6. Apply floor to preserve continuous physical loads (refrigerator/standby)
    df["demand_kwh"] = np.maximum(total_demand, base_load_min).round(3)


    return df[
        ["timestamp", "hour", "day_of_week", "is_weekend", "demand_kwh"]
    ]


if __name__ == "__main__":
    solar_ts = pd.read_csv(PROCESSED_DIR / "solar_combined.csv", usecols=["timestamp"], parse_dates=["timestamp"])
    df_demand = generate_household_demand(year=int(solar_ts["timestamp"].dt.year.iloc[0]))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_demand.to_csv(PROCESSED_DIR / "household_demand.csv", index=False)
    print("Saved:", PROCESSED_DIR / "household_demand.csv")
    first_day = df_demand.head(24)

    print("Profile Shape:", df_demand.shape)
    print("\nFirst 24 hours:")
    print(df_demand.head(24)[["timestamp", "demand_kwh"]])

    print("\nSummary Statistics (kWh):")
    print(df_demand["demand_kwh"].describe().round(3))

    daily_totals = df_demand.resample("D", on="timestamp")["demand_kwh"].sum()
    print(f"\nAverage daily demand: {daily_totals.mean():.2f} kWh/day")
    print(
        f"Weekday mean: {daily_totals[daily_totals.index.dayofweek < 5].mean():.2f} kWh/day"
    )
    print(
        f"Weekend mean: {daily_totals[daily_totals.index.dayofweek >= 5].mean():.2f} kWh/day"
    )
    print("\nAnnual demand:", df_demand["demand_kwh"].sum().round(2), "kWh")

    print("\nMaximum hourly demand:",
      df_demand["demand_kwh"].max(), "kWh")
    plt.plot(first_day["hour"], first_day["demand_kwh"])
    plt.xlabel("Hour")
    plt.ylabel("Demand (kWh)")
    plt.title("Synthetic Household Demand — First Day")
    plt.xticks(range(24))
    plt.show()
