"""Oracle tests. Run from the project root:  python tests/test_oracle.py"""
import sys
from math import isclose
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

for p in (ROOT_DIR, SRC_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)


from simulation import config as cfg
from simulation.engine import Battery, run_simulation
from simulation.oracle import make_oracle_strategy
from simulation.strategies import reactive_strategy


def make_state(solar, demand, grid):
    ts = pd.date_range("2025-01-01 18:00", periods=len(demand), freq="h")
    return pd.DataFrame({
        "timestamp": ts, "hour": ts.hour, "is_weekend": False,
        "solar_generation_kwh": solar, "demand_kwh": demand, "grid_test": grid,
    })


def run(solar, demand, grid, soc=0.5, **kw):
    st = make_state(solar, demand, grid)
    strat = make_oracle_strategy(st, "grid_test", **kw)
    return run_simulation(st, strat, "grid_test", battery=Battery(initial_soc=soc))


def close(a, b): return isclose(a, b, abs_tol=1e-9)


def test_no_outage_ahead_behaves_like_reactive():
    n = 48
    rng = np.random.default_rng(1)
    solar = np.clip(np.sin((pd.date_range("2025-01-01 18:00", periods=n, freq="h").hour - 6) / 12 * np.pi), 0, None) * 2
    demand = rng.uniform(0.1, 1.0, n)
    grid = [1] * n
    st = make_state(solar, demand, grid)
    a = run_simulation(st, reactive_strategy, "grid_test")
    b = run_simulation(st, make_oracle_strategy(st, "grid_test"), "grid_test")
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-9)


def test_holds_energy_for_coming_outage():
    # hour 0 grid up, outage from hour 4 to 6 needing ~2.4 kWh; battery at 60%
    solar = [0.0] * 8
    demand = [0.8] * 8
    grid = [1, 1, 1, 1, 0, 0, 0, 1]
    res = run(solar, demand, grid, soc=0.6, precharge_h=0)
    # while waiting, battery must not be drained below what the outage needs
    needed_kwh = 3 * 0.8
    stored_at_outage = res.iloc[4].battery_soc_start * cfg.BATTERY_CAPACITY_KWH
    floor = cfg.BATTERY_MIN_SOC * cfg.BATTERY_CAPACITY_KWH
    assert stored_at_outage >= min(floor + needed_kwh / np.sqrt(cfg.BATTERY_ROUND_TRIP_EFFICIENCY),
                                   0.6 * cfg.BATTERY_CAPACITY_KWH) - 1e-9


def test_precharges_from_grid_before_outage():
    solar = [0.0] * 6
    demand = [0.5] * 6
    grid = [1, 1, 1, 0, 0, 1]
    res = run(solar, demand, grid, soc=cfg.BATTERY_MIN_SOC, precharge_h=3)
    assert res.grid_to_battery_kwh.sum() > 0
    assert res.generator_running.sum() == 0          # battery carried the outage


def test_no_precharge_when_outage_is_far_away():
    solar = [0.0] * 10
    demand = [0.3] * 10
    grid = [1] * 8 + [0, 1]
    res = run(solar, demand, grid, soc=cfg.BATTERY_MIN_SOC, precharge_h=2)
    assert res.iloc[:5].grid_to_battery_kwh.sum() == 0


def test_generator_only_after_battery_floor_in_outage():
    res = run([0.0] * 3, [0.8] * 3, [0, 0, 0], soc=0.4)
    r0 = res.iloc[0]
    assert r0.battery_to_load_kwh > 0 and r0.generator_running == 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print("ok  ", t.__name__)
    print(f"\nAll {len(tests)} oracle tests passed.")
