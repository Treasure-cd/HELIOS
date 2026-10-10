import sys
from math import isclose
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

for p in (ROOT_DIR, SRC_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from simulation import config as cfg
from simulation.engine import Battery, run_simulation
from simulation.strategies import naive_strategy, reactive_strategy


def one_hour(solar, demand, grid_up, strategy, soc=0.5):
    ts = pd.date_range("2025-01-01 20:00", periods=1, freq="h")
    state = pd.DataFrame({
        "timestamp": ts, "hour": ts.hour, "is_weekend": False,
        "solar_generation_kwh": [solar], "demand_kwh": [demand], "grid_test": [grid_up],
    })
    return run_simulation(state, strategy, "grid_test", battery=Battery(initial_soc=soc)).iloc[0]


def close(a, b): return isclose(a, b, abs_tol=1e-9)


def test_naive_uses_grid_when_up():
    r = one_hour(0.0, 0.8, 1, naive_strategy)
    assert close(r.grid_to_load_kwh, 0.8) and r.generator_running == 0


def test_naive_starts_generator_immediately_on_outage():
    r = one_hour(0.0, 0.8, 0, naive_strategy)
    assert close(r.generator_to_load_kwh, 0.8) and r.battery_to_load_kwh == 0


def test_naive_never_touches_battery():
    r = one_hour(3.0, 0.5, 1, naive_strategy)
    assert r.battery_charge_kwh == 0 and close(r.solar_curtailed_kwh, 2.5)


def test_reactive_charges_from_surplus_solar():
    r = one_hour(3.0, 0.5, 1, reactive_strategy)
    assert r.battery_charge_kwh > 0 and r.grid_to_load_kwh == 0


def test_reactive_uses_battery_before_grid():
    r = one_hour(0.0, 0.8, 1, reactive_strategy)
    assert close(r.battery_to_load_kwh, 0.8) and r.grid_to_load_kwh == 0


def test_reactive_falls_back_to_generator_when_battery_empty_in_outage():
    r = one_hour(0.0, 0.8, 0, reactive_strategy, soc=cfg.BATTERY_MIN_SOC)
    assert r.battery_to_load_kwh == 0 and close(r.generator_to_load_kwh, 0.8)


def test_reactive_tops_up_generator_when_battery_is_short():
    r = one_hour(0.0, 0.8, 0, reactive_strategy, soc=cfg.BATTERY_MIN_SOC + 0.02)
    assert r.battery_to_load_kwh > 0 and r.generator_to_load_kwh > 0
    assert close(r.battery_to_load_kwh + r.generator_to_load_kwh, 0.8)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print("ok  ", t.__name__)
    print(f"\nAll {len(tests)} strategy tests passed.")
