import pandas as pd
for p in ["evening", "uniform"]:
    r = pd.read_csv(f"data/results/sweep_{p}_runs.csv")
    r = r[(r.scenario == "severe") & (r.pv_kw == 3.0) & (r.battery_kwh == 2.5)].copy()
    r = r[r.scenario == "severe"].copy()
    base = r[r.strategy == "reactive"].set_index("seed")["total_cost_ngn"]
    r["saved_pct"] = 100 * (r.seed.map(base) - r.total_cost_ngn) / r.seed.map(base)
    r["split"] = r.seed.map(lambda s: "tune (0-9)" if s < 10 else "test (10-29)")
    print(p); print(r.groupby(["split", "strategy"]).saved_pct.agg(["mean", "std"]).round(1))
