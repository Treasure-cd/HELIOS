"""Outage-risk features and model. One feature path for training AND the strategy."""
import json
from collections import deque
from functools import lru_cache
from pathlib import Path

import numpy as np

HORIZON_H = 3          # predict: does an outage start within the next 3 hours?
CAP_SINCE = 72
CAP_LEN = 24
N_HOURS = 24
N_FEATURES = N_HOURS + 4


class OutageTracker:
    """Online memory of the grid's past. Sees only hours already observed."""

    def __init__(self):
        self.hist = deque(maxlen=72)       # 1 = outage hour, newest last
        self.n24 = 0
        self.n72 = 0
        self.since_end = CAP_SINCE         # hours since the last outage ended
        self.cur_len = 0
        self.last_len = 0

    def step(self, grid_up: bool) -> None:
        out = 0 if grid_up else 1
        if len(self.hist) == 72:
            self.n72 -= self.hist[0]
        if len(self.hist) >= 24:
            self.n24 -= self.hist[-24]
        self.hist.append(out)
        self.n24 += out
        self.n72 += out
        if grid_up:
            if self.cur_len > 0:
                self.last_len, self.cur_len, self.since_end = self.cur_len, 0, 0
            else:
                self.since_end = min(self.since_end + 1, CAP_SINCE)
        else:
            self.cur_len += 1

    def features(self, hour_of_day: int) -> np.ndarray:
        x = np.zeros(N_FEATURES)
        x[hour_of_day] = 1.0
        x[N_HOURS] = self.since_end / CAP_SINCE
        x[N_HOURS + 1] = min(self.last_len, CAP_LEN) / CAP_LEN
        x[N_HOURS + 2] = self.n24 / 24
        x[N_HOURS + 3] = self.n72 / 72
        return x


class RiskModel:
    def __init__(self, coef, intercept):
        self.coef = np.asarray(coef, dtype=float)
        self.intercept = float(intercept)

    def predict(self, x: np.ndarray) -> float:
        return float(1.0 / (1.0 + np.exp(-(x @ self.coef + self.intercept))))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(
            {"coef": self.coef.tolist(), "intercept": self.intercept}))

    @classmethod
    def load(cls, path) -> "RiskModel":
        d = json.loads(Path(path).read_text())
        return cls(d["coef"], d["intercept"])


@lru_cache(maxsize=None)
def load_model(pattern: str, results_dir: str = "data/results") -> RiskModel:
    return RiskModel.load(Path(results_dir) / f"risk_model_{pattern}.json")
