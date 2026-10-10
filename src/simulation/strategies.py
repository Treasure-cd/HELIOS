from simulation.engine import Battery, Decision, HourInput
from simulation.risk import OutageTracker


def naive_strategy(hour: HourInput, battery: Battery) -> Decision:
    solar_to_load = min(hour.solar_kwh, hour.demand_kwh)
    deficit = hour.demand_kwh - solar_to_load

    d = Decision(solar_to_load=solar_to_load)
    if hour.grid_available:
        d.grid_to_load = deficit
    else:
        d.generator_to_load = deficit
    return d


def reactive_strategy(hour: HourInput, battery: Battery) -> Decision:
    solar_to_load = min(hour.solar_kwh, hour.demand_kwh)
    surplus = hour.solar_kwh - solar_to_load
    deficit = hour.demand_kwh - solar_to_load

    d = Decision(solar_to_load=solar_to_load, solar_to_battery=surplus)

    battery_to_load = min(deficit, battery.max_discharge_output_kwh())
    d.battery_to_load = battery_to_load
    deficit -= battery_to_load

    if hour.grid_available:
        d.grid_to_load = deficit
    else:
        d.generator_to_load = deficit
    return d


def make_helios_strategy(reserve_soc: float):
    """
    HELIOS v1: reactive, but protects an outage reserve while the grid is up.
    Grid up  : battery may cover the shortfall only down to `reserve_soc`;
               the grid supplies the rest.
    Grid down: battery may be used down to its floor; generator covers the rest.
    No forecasting, no grid-to-battery charging.
    """

    def helios_strategy(hour: HourInput, battery: Battery) -> Decision:
        reserve = max(reserve_soc, battery.min_soc)
        solar_to_load = min(hour.solar_kwh, hour.demand_kwh)
        surplus = hour.solar_kwh - solar_to_load
        deficit = hour.demand_kwh - solar_to_load

        d = Decision(solar_to_load=solar_to_load, solar_to_battery=surplus)

        if hour.grid_available:
            above_reserve = max(0.0, (battery.soc - reserve) * battery.capacity_kwh * battery.eta)
            battery_to_load = min(deficit, above_reserve, battery.max_discharge_output_kwh())
            d.battery_to_load = battery_to_load
            d.grid_to_load = deficit - battery_to_load
        else:
            battery_to_load = min(deficit, battery.max_discharge_output_kwh())
            d.battery_to_load = battery_to_load
            d.generator_to_load = deficit - battery_to_load
        return d

    return helios_strategy

def make_helios_window_strategy(reserve_soc: float, start_h: int = 15, end_h: int = 22):
    """
    HELIOS v2: reactive most of the day, static reserve only in the
    pre-outage-risk window (default 15:00 to 22:59). No forecasting.
    """
    protected = make_helios_strategy(reserve_soc)

    def helios_window_strategy(hour: HourInput, battery: Battery) -> Decision:
        if start_h <= hour.i % 24 <= end_h:
            return protected(hour, battery)
        return reactive_strategy(hour, battery)

    return helios_window_strategy

def make_helios_predictive_strategy(model, threshold: float, reserve_soc: float = 0.5, precharge: bool = True):
    """
    HELIOS v3: reactive, unless the learned outage risk is >= threshold.
    When risk is high (grid up): protect a reserve AND top the battery up
    from the grid to that reserve. Build a fresh one per run (it has memory).
    """
    tracker = OutageTracker()
    protected = make_helios_strategy(reserve_soc)

    def helios_predictive_strategy(hour: HourInput, battery: Battery) -> Decision:
        tracker.step(hour.grid_available)
        if not hour.grid_available:
            return reactive_strategy(hour, battery)
        if model.predict(tracker.features(hour.hour)) < threshold:
            return reactive_strategy(hour, battery)

        d = protected(hour, battery)
        target = max(reserve_soc, battery.min_soc) * battery.capacity_kwh
        shortfall = target - battery.stored_kwh
        if precharge and shortfall > 0:
            d.grid_to_battery = max(0.0, shortfall / battery.eta - d.solar_to_battery)
        return d

    return helios_predictive_strategy


RESERVE_SWEEP = (0.30, 0.50, 0.70)   # fixed in advance, from the demand profile

STRATEGIES = {
    "naive": naive_strategy,
    "reactive": reactive_strategy,
    **{f"helios_r{round(r * 100)}": make_helios_strategy(r) for r in RESERVE_SWEEP},
    **{f"helios_w{round(r * 100)}": make_helios_window_strategy(r) for r in (0.30, 0.50)},
}
