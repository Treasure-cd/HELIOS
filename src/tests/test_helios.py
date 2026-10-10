import sys
from math import isclose
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

for p in (ROOT_DIR, SRC_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from simulation import config as cfg
from simulation.engine import Battery, run_simulation
from simulation.metrics import avoided_vs, summarize_run
from simulation.strategies import make_helios_strategy, reactive_strategy


def make_state(solar, demand, grid):
    ts = pd.date_range("2025-01-01 20:00", periods=len(demand), freq="h")
    return pd.DataFrame({
        "timestamp": ts, "hour": ts.hour, "is_weekend": False,
        "solar_generation_kwh": solar, "demand_kwh": demand, "grid_test": grid,
    })


def one_hour(solar, demand, grid_up, strategy, soc=0.5):
    state = make_state([solar], [demand], [grid_up])
    return run_simulation(state, strategy, "grid_test", battery=Battery(initial_soc=soc)).iloc[0]


def close(a, b): return isclose(a, b, abs_tol=1e-9)


def test_keeps_reserve_when_grid_is_up():
    r = one_hour(0.0, 0.8, 1, make_helios_strategy(0.50), soc=0.50)
    assert r.battery_to_load_kwh == 0 and close(r.grid_to_load_kwh, 0.8)


def test_uses_battery_above_reserve_when_grid_is_up():
    r = one_hour(0.0, 0.8, 1, make_helios_strategy(0.50), soc=0.90)
    assert close(r.battery_to_load_kwh, 0.8) and r.grid_to_load_kwh == 0


def test_only_the_part_above_reserve_is_used_when_grid_is_up():
    # SOC just above reserve: battery gives its small surplus, grid covers the rest
    r = one_hour(0.0, 0.8, 1, make_helios_strategy(0.50), soc=0.52)
    assert 0 < r.battery_to_load_kwh < 0.8
    assert close(r.battery_to_load_kwh + r.grid_to_load_kwh, 0.8)
    assert r.battery_soc >= 0.50 - 1e-9


def test_dips_into_reserve_during_outage():
    r = one_hour(0.0, 0.8, 0, make_helios_strategy(0.50), soc=0.50)
    assert close(r.battery_to_load_kwh, 0.8) and r.generator_running == 0


def test_generator_only_after_battery_floor():
    r = one_hour(0.0, 0.8, 0, make_helios_strategy(0.50), soc=cfg.BATTERY_MIN_SOC)
    assert r.battery_to_load_kwh == 0 and close(r.generator_to_load_kwh, 0.8)


def test_reserve_at_floor_equals_reactive():
    rng = np.random.default_rng(3)
    n = 24 * 30
    hours = pd.date_range("2025-01-01", periods=n, freq="h").hour
    solar = np.clip(np.sin((hours - 6) / 12 * np.pi), 0, None) * rng.uniform(0, 3, n)
    state = make_state(solar, rng.uniform(0.1, 1.2, n), rng.integers(0, 2, n))
    a = run_simulation(state, reactive_strategy, "grid_test")
    b = run_simulation(state, make_helios_strategy(cfg.BATTERY_MIN_SOC), "grid_test")
    assert_frame_equal(a, b, check_exact=False, atol=1e-9)


def test_summarize_run_costs():
    # 1 hour outage served 100% by generator at 1 kWh, then 1 hour on grid at 1 kWh
    state = make_state([0.0, 0.0], [1.0, 1.0], [0, 1])
    def strat(h, b):
        from simulation.engine import Decision
        return Decision(generator_to_load=1.0) if not h.grid_available else Decision(grid_to_load=1.0)
    m = summarize_run(run_simulation(state, strat, "grid_test"))
    fuel_l = cfg.GENERATOR_FUEL_L_PER_H_NO_LOAD + (
        cfg.GENERATOR_FUEL_L_PER_H_FULL_LOAD - cfg.GENERATOR_FUEL_L_PER_H_NO_LOAD) / cfg.GENERATOR_RATED_KW
    assert m["gen_hours"] == 1 and close(m["fuel_l"], fuel_l)
    assert close(m["fuel_cost_ngn"], fuel_l * cfg.FUEL_PRICE_NGN_PER_L)
    assert close(m["grid_cost_ngn"], cfg.GRID_TARIFF_NGN_PER_KWH)
    assert close(m["total_cost_ngn"], m["fuel_cost_ngn"] + m["grid_cost_ngn"])


def test_avoided_vs_sign_convention():
    idx = pd.MultiIndex.from_tuples([("mild", "a"), ("mild", "b")], names=["scenario", "strategy"])
    s = pd.DataFrame({"gen_hours": [10.0, 4.0], "fuel_cost_ngn": [100.0, 40.0], "total_cost_ngn": [500.0, 520.0]}, index=idx)
    out = avoided_vs(s, "a")
    assert out.loc[("mild", "b"), "gen_hours_avoided"] == 6.0       # fewer hours = positive
    assert out.loc[("mild", "b"), "total_cost_ngn_avoided"] == -20  # costs more = negative


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print("ok  ", t.__name__)
    print(f"\nAll {len(tests)} HELIOS/metrics tests passed.")
