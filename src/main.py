import pandas as pd
s = pd.read_csv("data/processed/state_table.csv", parse_dates=["timestamp"])
print("rows:", len(s), "| years:", s.timestamp.dt.year.unique(), "| duplicate timestamps:", s.timestamp.duplicated().sum())
print("Feb rows:", (s.timestamp.dt.month == 2).sum(), "(696 = leap year, 672 = normal)")
