"""
HELIOS simulation engine.

The engine knows PHYSICS, not STRATEGY.

    strategy(hour, battery) -> Decision      what the strategy WANTS (kWh requests)
    engine                  -> final flows   what the system can actually DO

For every hour the engine:
  1. asks the strategy for requested energy flows
  2. clips them to physical limits (supply, battery power/SOC, generator rating,
     grid availability)
  3. updates the battery state of charge
  4. checks the energy balance, and stops with an error if it is violated
  5. records the hour

Resolution order when serving the household load:
    solar -> battery -> grid -> generator
Requests are clipped to the load still unmet at each step, so a strategy should
request amounts that do not overlap. Any load left over is recorded as unserved
energy; the engine never starts the generator on its own.

Costs are NOT computed here (see metrics.py); the engine only records flows,
fuel use and battery state.
"""

from dataclasses import dataclass, fields
from math import sqrt
from typing import Callable, Optional

import pandas as pd

from simulation import config as cfg

DT_H = 1.0     # one timestep = one hour, so kW and kWh are numerically equal
EPS = 1e-9


# --------------------------------------------------------------------------
# Data passed to / from strategies
# --------------------------------------------------------------------------
@dataclass
class HourInput:
    """Everything a strategy may know about the current hour."""
    i: int
    timestamp: pd.Timestamp
    hour: int
    is_weekend: bool
    solar_kwh: float          # solar energy available this hour
    demand_kwh: float
    grid_available: bool


@dataclass
class Decision:
    """Requested energy flows for one hour, in kWh. All default to 0."""
    solar_to_load: float = 0.0
    solar_to_battery: float = 0.0
    grid_to_load: float = 0.0
    grid_to_battery: float = 0.0
    battery_to_load: float = 0.0
    generator_to_load: float = 0.0
    generator_to_battery: float = 0.0


Strategy = Callable[[HourInput, "Battery"], Decision]


# --------------------------------------------------------------------------
# Battery
# --------------------------------------------------------------------------
class Battery:
    """
    SOC is a fraction of nameplate capacity. Usable window: [min_soc, max_soc].
    Round-trip efficiency is split equally between charging and discharging.

    Flows are measured at the battery terminals:
      charge_in  : energy delivered INTO the battery   (stored += charge_in * eta)
      discharge_out : energy delivered OUT to the load (stored -= out / eta)
    """

    def __init__(
        self,
        capacity_kwh: float = cfg.BATTERY_CAPACITY_KWH,
        initial_soc: float = cfg.BATTERY_INITIAL_SOC,
        min_soc: float = cfg.BATTERY_MIN_SOC,
        max_soc: float = 1.0,
        max_charge_kw: float = cfg.BATTERY_MAX_CHARGE_KW,
        max_discharge_kw: float = cfg.BATTERY_MAX_DISCHARGE_KW,
        round_trip_efficiency: float = cfg.BATTERY_ROUND_TRIP_EFFICIENCY,
    ):
        self.capacity_kwh = capacity_kwh
        self.soc = initial_soc
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw
        self.eta = sqrt(round_trip_efficiency)   # one-way efficiency

    @property
    def stored_kwh(self) -> float:
        return self.soc * self.capacity_kwh

    def max_charge_input_kwh(self) -> float:
        """Most energy the battery can accept this hour."""
        room = (self.max_soc - self.soc) * self.capacity_kwh / self.eta
        return max(0.0, min(self.max_charge_kw * DT_H, room))

    def max_discharge_output_kwh(self) -> float:
        """Most energy the battery can deliver to the load this hour."""
        usable = (self.soc - self.min_soc) * self.capacity_kwh * self.eta
        return max(0.0, min(self.max_discharge_kw * DT_H, usable))

    def update(self, charge_in_kwh: float, discharge_out_kwh: float) -> None:
        stored = (
            self.stored_kwh
            + charge_in_kwh * self.eta
            - discharge_out_kwh / self.eta
        )
        self.soc = stored / self.capacity_kwh


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------
def generator_capacity_kwh() -> float:
    return cfg.GENERATOR_RATED_KW * DT_H


def generator_fuel_l(output_kwh: float) -> float:
    """Litres burned in an hour producing `output_kwh`. Zero if not running."""
    if output_kwh <= EPS:
        return 0.0
    load_fraction = min(output_kwh / generator_capacity_kwh(), 1.0)
    return cfg.GENERATOR_FUEL_L_PER_H_NO_LOAD + (
        cfg.GENERATOR_FUEL_L_PER_H_FULL_LOAD - cfg.GENERATOR_FUEL_L_PER_H_NO_LOAD
    ) * load_fraction


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
REQUIRED_COLUMNS = ["timestamp", "hour", "is_weekend", "demand_kwh"]


def run_simulation(
    state: pd.DataFrame,
    strategy: Strategy,
    grid_col: str,
    solar_col: str = "solar_generation_kwh",
    battery: Optional[Battery] = None,
    check_each_hour: bool = True,
) -> pd.DataFrame:
    """
    Run one strategy over one grid scenario.

    state    : the merged state table (one row per hour)
    grid_col : which outage scenario to use, e.g. "grid_moderate"
    solar_col: which solar column to use (lets us sweep PV size later)
    battery  : pass a configured Battery for sensitivity runs; default from config
    """
    missing = [c for c in REQUIRED_COLUMNS + [grid_col, solar_col] if c not in state]
    if missing:
        raise KeyError(f"state table is missing columns: {missing}")

    battery = battery or Battery()
    gen_cap = generator_capacity_kwh()
    request_fields = [f.name for f in fields(Decision)]
    records = []

    for i, r in enumerate(state.itertuples(index=False)):
        solar = max(0.0, float(getattr(r, solar_col)))
        demand = float(r.demand_kwh)
        grid_up = bool(getattr(r, grid_col))
        soc_start = battery.soc

        hour = HourInput(
            i=i,
            timestamp=r.timestamp,
            hour=int(r.hour),
            is_weekend=bool(r.is_weekend),
            solar_kwh=solar,
            demand_kwh=demand,
            grid_available=grid_up,
        )

        decision = strategy(hour, battery)
        req = {name: max(0.0, float(getattr(decision, name))) for name in request_fields}

        # ---- 1. serve the load: solar -> battery -> grid -> generator ----
        solar_to_load = min(req["solar_to_load"], solar, demand)
        remaining = demand - solar_to_load

        battery_to_load = min(
            req["battery_to_load"], remaining, battery.max_discharge_output_kwh()
        )
        remaining -= battery_to_load

        grid_to_load = min(req["grid_to_load"], remaining) if grid_up else 0.0
        remaining -= grid_to_load

        generator_to_load = min(req["generator_to_load"], remaining, gen_cap)
        remaining -= generator_to_load

        unserved = max(remaining, 0.0)

        # ---- 2. charge the battery (never in an hour it discharges) ----
        solar_to_battery = grid_to_battery = generator_to_battery = 0.0
        if battery_to_load <= EPS:
            room = battery.max_charge_input_kwh()
            solar_to_battery = min(req["solar_to_battery"], solar - solar_to_load, room)
            room -= solar_to_battery
            if grid_up:
                grid_to_battery = min(req["grid_to_battery"], room)
                room -= grid_to_battery
            generator_to_battery = min(
                req["generator_to_battery"], gen_cap - generator_to_load, room
            )

        charge_in = solar_to_battery + grid_to_battery + generator_to_battery
        curtailed = solar - solar_to_load - solar_to_battery
        grid_import = grid_to_load + grid_to_battery
        generator_output = generator_to_load + generator_to_battery

        # ---- 3. update battery ----
        battery.update(charge_in, battery_to_load)
        stored_change = (battery.soc - soc_start) * battery.capacity_kwh
        battery_losses = charge_in * (1 - battery.eta) + battery_to_load * (
            1 / battery.eta - 1
        )

        # ---- 4. energy balance (independent of how the flows were derived) ----
        # in : solar + grid + generator
        # out: load actually served + curtailed + net change in stored energy + losses
        served = demand - unserved
        balance_error = (solar + grid_import + generator_output) - (
            served + curtailed + stored_change + battery_losses
        )
        if check_each_hour and abs(balance_error) > 1e-7:
            raise AssertionError(
                f"Energy balance violated at {hour.timestamp}: error={balance_error:.3e}"
            )

        records.append(
            {
                "timestamp": hour.timestamp,
                "hour": hour.hour,
                "grid_available": int(grid_up),
                "demand_kwh": demand,
                "solar_kwh": solar,
                "solar_to_load_kwh": solar_to_load,
                "solar_to_battery_kwh": solar_to_battery,
                "solar_curtailed_kwh": curtailed,
                "grid_to_load_kwh": grid_to_load,
                "grid_to_battery_kwh": grid_to_battery,
                "battery_to_load_kwh": battery_to_load,
                "battery_charge_kwh": charge_in,
                "battery_losses_kwh": battery_losses,
                "generator_to_load_kwh": generator_to_load,
                "generator_to_battery_kwh": generator_to_battery,
                "generator_running": int(generator_output > EPS),
                "generator_fuel_l": generator_fuel_l(generator_output),
                "unserved_energy_kwh": unserved,
                "battery_soc_start": soc_start,
                "battery_soc": battery.soc,
                "energy_balance_error": balance_error,
            }
        )

    results = pd.DataFrame.from_records(records)
    results.attrs["grid_col"] = grid_col
    check_invariants(results, battery)
    return results


# --------------------------------------------------------------------------
# Whole-run physical checks
# --------------------------------------------------------------------------
def check_invariants(res: pd.DataFrame, battery: Battery) -> None:
    """Raise AssertionError if any physical rule was broken in the run."""
    flow_cols = [c for c in res.columns if c.endswith("_kwh") or c == "generator_fuel_l"]
    assert (res[flow_cols] >= -EPS).all().all(), "negative flow found"

    assert res["battery_soc"].between(battery.min_soc - 1e-9, battery.max_soc + 1e-9).all(), \
        "battery SOC left its allowed window"
    assert (res["battery_charge_kwh"] <= battery.max_charge_kw * DT_H + 1e-9).all(), \
        "battery charge power limit exceeded"
    assert (res["battery_to_load_kwh"] <= battery.max_discharge_kw * DT_H + 1e-9).all(), \
        "battery discharge power limit exceeded"
    assert not ((res["battery_to_load_kwh"] > EPS) & (res["battery_charge_kwh"] > EPS)).any(), \
        "battery charged and discharged in the same hour"

    gen_out = res["generator_to_load_kwh"] + res["generator_to_battery_kwh"]
    assert (gen_out <= generator_capacity_kwh() + 1e-9).all(), "generator rating exceeded"

    grid_flow = res["grid_to_load_kwh"] + res["grid_to_battery_kwh"]
    assert (grid_flow[res["grid_available"] == 0] <= EPS).all(), "grid used during an outage"

    assert (res["solar_to_load_kwh"] + res["solar_to_battery_kwh"]
            + res["solar_curtailed_kwh"] - res["solar_kwh"]).abs().max() < 1e-7, \
        "solar not fully accounted for"
    served = (res["solar_to_load_kwh"] + res["battery_to_load_kwh"]
              + res["grid_to_load_kwh"] + res["generator_to_load_kwh"])
    assert (served + res["unserved_energy_kwh"] - res["demand_kwh"]).abs().max() < 1e-7, \
        "demand not fully accounted for"
    assert res["energy_balance_error"].abs().max() < 1e-7, "energy balance violated"
