import pandas as pd
s = pd.read_csv("data/processed/solar_combined.csv", parse_dates=["timestamp"])
print(s.groupby(s["timestamp"].dt.hour)["ALLSKY_SFC_SW_DWN"].mean().idxmax())
