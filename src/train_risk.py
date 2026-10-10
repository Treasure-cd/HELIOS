"""Train and evaluate the outage-risk model.  Usage: python src/train_risk.py"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from run_sweep import PATTERNS
from simulation.grid import SCENARIOS, generate_grid_availability
from simulation.risk import HORIZON_H, OutageTracker, RiskModel

TRAIN_SEEDS = range(100, 115)   # never used in any evaluation
TEST_SEEDS = range(0, 30)       # same seeds as the sweeps
THRESHOLDS = (0.10, 0.13)
RESULTS_DIR = Path("data/results")


def dataset(timestamps, hours, scenario, seed, weights):
    grid = generate_grid_availability(
        timestamps, seed=seed, start_hour_weights=weights, **SCENARIOS[scenario]
    )["grid"].to_numpy()
    tracker, X, y = OutageTracker(), [], []
    for t in range(len(grid) - HORIZON_H):
        tracker.step(bool(grid[t]))
        if grid[t] == 1:   # decisions only matter while the grid is up
            X.append(tracker.features(int(hours[t])))
            y.append(int((grid[t + 1 : t + 1 + HORIZON_H] == 0).any()))
    return np.array(X), np.array(y)


def stack(parts):
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


if __name__ == "__main__":
    state = pd.read_csv("data/processed/state_table.csv", parse_dates=["timestamp"])
    ts, hours = state["timestamp"], state["hour"].to_numpy()

    for pattern, weights in PATTERNS.items():
        X, y = stack([dataset(ts, hours, sc, s, weights)
                      for sc in SCENARIOS for s in TRAIN_SEEDS])
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
        RiskModel(clf.coef_[0], clf.intercept_[0]).save(RESULTS_DIR / f"risk_model_{pattern}.json")

        Xt, yt = stack([dataset(ts, hours, "severe", s, weights) for s in TEST_SEEDS])
        p = clf.predict_proba(Xt)[:, 1]
        hod = Xt[:, :24].argmax(1)
        hr_rate = np.array([y[X[:, h] == 1].mean() for h in range(24)])

        print(f"\n=== {pattern}: severe, seeds 0-29, trained on seeds 100-114 ===")
        print(f"P(outage starts within {HORIZON_H} h | grid up) = {yt.mean():.3f}")
        print(f"AUC: model {roc_auc_score(yt, p):.3f} | hour-of-day rate only "
              f"{roc_auc_score(yt, hr_rate[hod]):.3f} | 'current state continues' 0.500 (recall 0)")
        for thr in THRESHOLDS:
            flag = p >= thr
            prec = yt[flag].mean() if flag.any() else float("nan")
            print(f"threshold {thr:.2f}: flags {flag.mean():.1%} of grid-up hours | "
                  f"precision {prec:.3f} | recall {yt[flag].sum() / yt.sum():.3f}")
        print("mean predicted risk by hour 0-23:",
              np.round(pd.Series(p).groupby(hod).mean().to_numpy(), 2))
