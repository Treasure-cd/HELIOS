import sys
from math import isclose, sqrt
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
from simulation.engine import Battery, Decision, run_simulation, generator_fuel_l

ETA = sqrt(cfg.BATTERY_ROUND_TRIP_EFFICIENCY)


def make_state(rows):
    # rows: list of (solar_kwh, demand_kwh, grid_up)
    n = len(rows)
    ts = pd.date_range("2025-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "timestamp": ts,
        "hour": ts.hour,
        "is_weekend": False,
        "solar_generation_kwh": [r[0] for r in rows],
        "demand_kwh": [r[1] for r in rows],
        "grid_test": [r[2] for r in rows],
    })


def run(rows, strategy, battery=None):
    return run_simulation(make_state(rows), strategy, grid_col="grid_test", battery=battery)


def close(a, b, tol=1e-9):
    return isclose(a, b, abs_tol=tol)


def test_do_nothing_leaves_load_unserved():
    res = run([(2.0, 0.5, 1), (0.0, 0.5, 0)], lambda h, b: Decision())
    assert close(res.unserved_energy_kwh.sum(), 1.0)
    assert close(res.solar_curtailed_kwh.sum(), 2.0)


def test_solar_serves_load_then_charges_battery():
    bat = Battery()
    soc0 = bat.soc
    res = run([(2.0, 0.5, 1)],
              lambda h, b: Decision(solar_to_load=0.5, solar_to_battery=1.5), bat)
    r = res.iloc[0]
    assert close(r.solar_to_load_kwh, 0.5) and close(r.solar_to_battery_kwh, 1.5)
    assert close(r.solar_curtailed_kwh, 0.0) and close(r.unserved_energy_kwh, 0.0)
    assert close(r.battery_soc, soc0 + 1.5 * ETA / cfg.BATTERY_CAPACITY_KWH)


def test_grid_request_ignored_during_outage():
    res = run([(0.0, 1.0, 0)], lambda h, b: Decision(grid_to_load=1.0, grid_to_battery=1.0))
    r = res.iloc[0]
    assert close(r.grid_to_load_kwh, 0.0) and close(r.grid_to_battery_kwh, 0.0)
    assert close(r.unserved_energy_kwh, 1.0)


def test_battery_cannot_go_below_floor():
    bat = Battery(initial_soc=cfg.BATTERY_MIN_SOC + 0.01)
    res = run([(0.0, 3.0, 0)], lambda h, b: Decision(battery_to_load=3.0), bat)
    r = res.iloc[0]
    expected = 0.01 * cfg.BATTERY_CAPACITY_KWH * ETA
    assert close(r.battery_to_load_kwh, expected)
    assert close(r.battery_soc, cfg.BATTERY_MIN_SOC)
    assert close(r.unserved_energy_kwh, 3.0 - expected)


def test_battery_power_limit():
    bat = Battery(initial_soc=1.0)
    res = run([(0.0, 4.0, 0)], lambda h, b: Decision(battery_to_load=4.0), bat)
    assert close(res.iloc[0].battery_to_load_kwh, cfg.BATTERY_MAX_DISCHARGE_KW)


def test_generator_rating_and_fuel():
    big = cfg.GENERATOR_RATED_KW + 2.0
    res = run([(0.0, big, 0)], lambda h, b: Decision(generator_to_load=big))
    r = res.iloc[0]
    assert close(r.generator_to_load_kwh, cfg.GENERATOR_RATED_KW)
    assert close(r.unserved_energy_kwh, 2.0)
    assert close(r.generator_fuel_l, cfg.GENERATOR_FUEL_L_PER_H_FULL_LOAD)
    half = cfg.GENERATOR_RATED_KW / 2
    expected = cfg.GENERATOR_FUEL_L_PER_H_NO_LOAD + 0.5 * (
        cfg.GENERATOR_FUEL_L_PER_H_FULL_LOAD - cfg.GENERATOR_FUEL_L_PER_H_NO_LOAD)
    assert close(generator_fuel_l(half), expected)
    assert generator_fuel_l(0.0) == 0.0


def test_no_simultaneous_charge_and_discharge():
    bat = Battery()
    res = run([(2.0, 1.0, 1)],
              lambda h, b: Decision(battery_to_load=1.0, solar_to_battery=1.0), bat)
    r = res.iloc[0]
    assert r.battery_to_load_kwh > 0 and r.battery_charge_kwh == 0


def test_round_trip_efficiency():
    bat = Battery(initial_soc=cfg.BATTERY_MIN_SOC)
    def strat(h, b):
        if h.i == 0:
            return Decision(solar_to_battery=2.0)
        return Decision(battery_to_load=2.5)
    res = run([(2.0, 0.0, 1), (0.0, 2.5, 0)], strat, bat)
    got = res.iloc[1].battery_to_load_kwh
    assert close(got, 2.0 * cfg.BATTERY_ROUND_TRIP_EFFICIENCY), got


def test_generator_can_charge_battery_within_rating():
    bat = Battery(initial_soc=0.3)
    res = run([(0.0, 1.0, 0)],
              lambda h, b: Decision(generator_to_load=1.0, generator_to_battery=10.0), bat)
    r = res.iloc[0]
    assert close(r.generator_to_load_kwh + r.generator_to_battery_kwh,
                 min(cfg.GENERATOR_RATED_KW, 1.0 + cfg.BATTERY_MAX_CHARGE_KW))


def test_fuzz_random_strategy_never_breaks_physics():
    """Random (even nonsense) requests must never produce impossible flows."""
    rng = np.random.default_rng(0)
    n = 24 * 60
    rows = list(zip(rng.uniform(0, 3, n), rng.uniform(0.05, 4, n), rng.integers(0, 2, n)))

    def chaotic(h, b):
        return Decision(*rng.uniform(-1, 5, 7))   # includes negatives and huge asks

    res = run(rows, chaotic)       # run_simulation checks every hour + full invariants
    assert len(res) == n


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"\nAll {len(tests)} engine tests passed.")
