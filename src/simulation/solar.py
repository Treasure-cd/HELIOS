def calculate_solar_generation(
    irradiance_wh_m2,
    pv_capacity_kw,
    system_efficiency,
):

    # Estimated electricity generation in kWh for the hour.
    irradiance_factor = irradiance_wh_m2 / 1000
    generation_kwh = pv_capacity_kw * irradiance_factor * system_efficiency
    return generation_kwh
