# --- Solar (already used by load_solar.py) ---------------------------------
PV_CAPACITY_KW = 5.0             # [SCENARIO] sweep this (e.g. 2, 3, 5)
SYSTEM_EFFICIENCY = 0.80         # [ESTIMATED] wiring, inverter, dust, heat

# --- Battery (LiFePO4 home battery) ----------------------------------------
# [SCENARIO] hypothetical residential battery size
BATTERY_CAPACITY_KWH = 5.0
BATTERY_MAX_DOD = 0.80               # [ESTIMATED] usable fraction of capacity
BATTERY_ROUND_TRIP_EFFICIENCY = 0.90 # [ESTIMATED] LiFePO4 typically 90-95%
BATTERY_MAX_CHARGE_KW = 2.5          # [ESTIMATED] ~0.5C
BATTERY_MAX_DISCHARGE_KW = 2.5       # [ESTIMATED] ~0.5C
BATTERY_INITIAL_SOC = 0.50           # [SCENARIO] fraction of capacity at hour 0
BATTERY_RESERVE_SOC = 0.30           # [SCENARIO] HELIOS keeps this for outages

# Wear cost = purchase cost spread over lifetime energy throughput
BATTERY_COST_NGN_PER_KWH = 300_000   # [ESTIMATED] VERIFY with a local quote
BATTERY_CYCLE_LIFE = 4000            # [ESTIMATED] cycles at the DoD above
BATTERY_WEAR_COST_PER_KWH = BATTERY_COST_NGN_PER_KWH / (
    BATTERY_CYCLE_LIFE * BATTERY_MAX_DOD
)                                    # derived: NGN per kWh discharged (~94)

# derived limits
BATTERY_MIN_SOC = 1.0 - BATTERY_MAX_DOD

# --- Generator (small petrol set) ------------------------------------------
GENERATOR_RATED_KW = 3.0             # [ESTIMATED] small household set
# Fuel use is not proportional to load: a running set burns a lot even when
# lightly loaded. Linear model: litres/hour = no_load + slope * (load / rated)
GENERATOR_FUEL_L_PER_H_NO_LOAD = 0.30    # [ESTIMATED] VERIFY (spec sheet)
GENERATOR_FUEL_L_PER_H_FULL_LOAD = 1.20  # [ESTIMATED] VERIFY (spec sheet)
FUEL_PRICE_NGN_PER_L = 1000              # [SOURCED] NBS petrol average, July 2025
GENERATOR_MAINTENANCE_NGN_PER_H = 0.0    # [TODO] oil, servicing; omitted for now

# --- Grid ------------------------------------------------------------------
GRID_TARIFF_NGN_PER_KWH = 225     # [SOURCED] Band A, PHED (serves Akwa Ibom)
