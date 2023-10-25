def calculate_installed_costs(wind_size, solar_size, hybrid_size):
    solar_cost_per_mw = 1100000
    wind_cost_per_mw = 1450000
    wind_installed_cost = wind_cost_per_mw * wind_size
    solar_installed_cost = solar_cost_per_mw * solar_size
    hybrid_installed_cost = (wind_cost_per_mw * hybrid_size/2) + (solar_cost_per_mw * hybrid_size/2)

    return wind_installed_cost, solar_installed_cost, hybrid_installed_cost

