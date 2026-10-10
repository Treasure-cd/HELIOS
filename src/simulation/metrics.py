"""
Metrics and costs, computed from an engine results table (one run).

Costs use the assumptions in config.py. The engine records physical flows only;
all money is calculated here.
"""

import pandas as pd

from simulation import config as cfg

DEFAULT_COMPARE = ("gen_hours", "fuel_cost_ngn", "total_cost_ngn")


def summarize_run(res: pd.DataFrame) -> dict:
    """Headline numbers for one strategy on one grid scenario."""
    gen_hours = int(res["generator_running"].sum())
    fuel_l = res["generator_fuel_l"].sum()
    grid_kwh = (res["grid_to_load_kwh"] + res["grid_to_battery_kwh"]).sum()
    battery_out = res["battery_to_load_kwh"].sum()
    battery_in = res["battery_charge_kwh"].sum()

    demand = res["demand_kwh"].sum()
    unserved = res["unserved_energy_kwh"].sum()
    solar_generated = res["solar_kwh"].sum()
    solar_used = (res["solar_to_load_kwh"] + res["solar_to_battery_kwh"]).sum()

    fuel_cost = fuel_l * cfg.FUEL_PRICE_NGN_PER_L + gen_hours * cfg.GENERATOR_MAINTENANCE_NGN_PER_H
    grid_cost = grid_kwh * cfg.GRID_TARIFF_NGN_PER_KWH
    wear_cost = battery_out * cfg.BATTERY_WEAR_COST_PER_KWH

    # Renewable share of the load actually served. Battery output is credited as
    # solar in proportion to how much of what went INTO the battery was solar
    # (well-mixed approximation).
    solar_share_of_charge = res["solar_to_battery_kwh"].sum() / battery_in if battery_in > 0 else 0.0
    renewable_kwh = res["solar_to_load_kwh"].sum() + battery_out * solar_share_of_charge

    return {
        "gen_hours": gen_hours,
        "fuel_l": fuel_l,
        "fuel_cost_ngn": fuel_cost,
        "grid_kwh": grid_kwh,
        "grid_cost_ngn": grid_cost,
        "battery_out_kwh": battery_out,
        "battery_cycles": battery_out / (cfg.BATTERY_CAPACITY_KWH * cfg.BATTERY_MAX_DOD),
        "battery_wear_ngn": wear_cost,
        "total_cost_ngn": fuel_cost + grid_cost + wear_cost,
        "unserved_kwh": unserved,
        "solar_utilization_pct": 100 * solar_used / solar_generated if solar_generated else 0.0,
        "renewable_share_pct": 100 * renewable_kwh / (demand - unserved) if demand > unserved else 0.0,
    }


def avoided_vs(summary: pd.DataFrame, baseline: str, metrics=DEFAULT_COMPARE) -> pd.DataFrame:
    """
    How much LESS each strategy uses/costs than `baseline`, per scenario.
    Positive = better than baseline, negative = worse.

    `summary` must have a (scenario, strategy) MultiIndex.
    """
    metrics = list(metrics)
    base = summary.xs(baseline, level="strategy")[metrics]
    base_aligned = base.reindex(summary.index.get_level_values("scenario")).to_numpy()
    avoided = base_aligned - summary[metrics].to_numpy()
    return pd.DataFrame(avoided, index=summary.index, columns=[f"{m}_avoided" for m in metrics])
