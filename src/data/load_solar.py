import sys
from pathlib import Path

# Add project root to sys.path so 'src' and root packages resolve cleanly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from src.simulation.solar import calculate_solar_generation
from src.simulation.config import PV_CAPACITY_KW, SYSTEM_EFFICIENCY

MONTH_ORDER = {
    "Jan-Feb": 1,
    "Mar-Apr": 3,
    "May-Jun": 5,
    "Jul-Aug": 7,
    "Sep-Oct": 9,

    "Nov-Dec": 11,
}

DATA_DIR = ROOT_DIR / "data" / "raw" / "solar"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def load_solar_data():
    files = sorted(DATA_DIR.glob("*.csv"), key=lambda file: MONTH_ORDER[file.stem])

    frames = []
    for file in files:
        df = pd.read_csv(file, skiprows=15)
        frames.append(df)

    solar = pd.concat(frames, ignore_index=True)

    solar["timestamp"] = pd.to_datetime(
        solar[["YEAR", "MO", "DY", "HR"]].rename(
            columns={
                "YEAR": "year",
                "MO": "month",
                "DY": "day",
                "HR": "hour",
            }
        )
    )

    # 1. Shift UTC to Lagos time (UTC+1)
    solar["timestamp"] = solar["timestamp"] + pd.Timedelta(hours=1)

    # 2. Fix boundary shift: trim next-year spillover and prepend Jan 1 00:00
    year = solar["timestamp"].dt.year.iloc[0]
    solar = solar[solar["timestamp"].dt.year == year]
    first = solar.iloc[[0]].copy()
    first[["timestamp", "ALLSKY_SFC_SW_DWN"]] = [pd.Timestamp(f"{year}-01-01 00:00"), 0.0]
    solar = pd.concat([first, solar], ignore_index=True)

    # 3. Sanity checks
    print("First timestamp:", solar["timestamp"].min())
    print("Last timestamp: ", solar["timestamp"].max())
    print("Rows:", len(solar))
    print("Duplicates:", solar["timestamp"].duplicated().sum())
    print("Missing timestamps:", solar["timestamp"].isna().sum())

    # 4. Compute PV generation
    solar["solar_generation_kwh"] = solar["ALLSKY_SFC_SW_DWN"].apply(
        lambda irradiance: calculate_solar_generation(
            irradiance_wh_m2=irradiance,
            pv_capacity_kw=PV_CAPACITY_KW,
            system_efficiency=SYSTEM_EFFICIENCY,
        )
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    solar.to_csv(PROCESSED_DIR / "solar_combined.csv", index=False)
    print("Saved:", PROCESSED_DIR / "solar_combined.csv")
    return solar


if __name__ == "__main__":
    load_solar_data()
