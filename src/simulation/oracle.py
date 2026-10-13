import numpy as np
import pandas as pd

from simulation.engine import Battery, Decision, HourInput

MAX_OUTAGE_SCAN_H = 24


def make_oracle_strategy(
    state: pd.DataFrame,
    grid_col: str,
    solar_col: str = "solar_generation_kwh",
    lookahead_h: int = 24,
    precharge_h: int = 3,
):
    grid_up = state[grid_col].to_numpy().astype(bool)
    demand = state["demand_kwh"].to_numpy(dtype=float)
    solar = np.clip(state[solar_col].to_numpy(dtype=float), 0.0, None)
    net_load = np.maximum(demand - solar, 0.0)
    n = len(grid_up)

    def next_outage(i: int):
        """(start_index, length_h) of the next outage within the lookahead, else None."""
        window = grid_up[i + 1 : min(n, i + 1 + lookahead_h)]
        off = np.flatnonzero(~window)
        if off.size == 0:
            return None
        start = i + 1 +                                                              int(off[0])
        end = start
        while end < n and not grid_up[end] and end - start < MAX_OUTAGE_SCAN_H:
            end += 1
        return start, end - start

    def oracle_strategy(hour: HourInput, battery: Battery) -> Decision:
        solar_to_load = min(hour.solar_kwh, hour.demand_kwh)
        surplus = hour.solar_kwh - solar_to_load
        deficit = hour.demand_kwh - solar_to_load
        d = Decision(solar_to_load=solar_to_load, solar_to_battery=surplus)

        if not hour.grid_available:
            battery_to_load = min(deficit, battery.max_discharge_output_kwh())
            d.battery_to_load = battery_to_load
            d.generator_to_load = deficit - battery_to_load
            return d

        floor_kwh = battery.min_soc * battery.capacity_kwh
        target_kwh = floor_kwh
        upcoming = next_outage(hour.i)
        if upcoming is not None:
            start, length = upcoming
            needed_out = net_load[start : start + length].sum()
            target_kwh = min(floor_kwh + needed_out / battery.eta,
                             battery.max_soc * battery.capacity_kwh)

        usable_above_target = max(0.0, battery.stored_kwh - target_kwh) * battery.eta
        battery_to_load = min(deficit, usable_above_target, battery.max_discharge_output_kwh())
        d.battery_to_load = battery_to_load
        d.grid_to_load = deficit - battery_to_load

        if upcoming is not None and (upcoming[0] - hour.i) <= precharge_h:
            shortfall_kwh = target_kwh - battery.stored_kwh
            if shortfall_kwh > 0:
                d.grid_to_battery = max(0.0, shortfall_kwh / battery.eta - surplus)
        return d

    return oracle_strategy
