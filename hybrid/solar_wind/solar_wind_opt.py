def calculate_PV_required_area(
        capacity: float,
        gcr: float,
        area_per_panel: float = 1.488 * .992,
        maximum_panel_power: float = .321,
        expansion_factor: float = 1.1,
        ) -> float:
    """
    Defaults are for PV Modules are 96-cell 1.488 x 0.992 m panel, Pmp=321W, Vmp=54V, Imp=5.9A (PVMismatch default)
    with a truck-width spacing between rows (expansion_factor = 1.1).
    
    Using gcr to calculate required area without considering stringing and row layout.
    Can get more detailed in future with row and stringing
    :param capacity: [DC kW]
    :param gcr: between 0 and 0.65
    :param area_per_panel [m^2]
    :param maximum_panel_power [kW]
    :param expansion_factor expands area by given multiplier to account for additional space requirements
    :return: required area [m^2]
    """
    panel_power_density = maximum_panel_power / area_per_panel
    array_power_density = panel_power_density * gcr
    # return (area_per_panel / (Pmp / dc_to_ac_ratio)) * (pv_size / gcr) * expansion_factor
    return expansion_factor * (capacity / array_power_density)


def calculate_PV_capacity(
        pv_area: float,
        gcr: float,
        area_per_panel: float = 1.488 * .992,
        maximum_panel_power: float = .321,
        expansion_factor: float = 1.1
        ) -> float:
    """
    Defaults are for PV Modules are 96-cell 1.488 x 0.992 m panel, Pmp=321W, Vmp=54V, Imp=5.9A (PVMismatch default)
    with a truck-width spacing between rows (expansion_factor = 1.1).

    Using gcr to calculate required area without considering stringing and row layout.
    Can get more detailed in future with row and stringing
    :param pv_area: [m^2]
    :param gcr: between 0 and 0.65
    :param area_per_panel [m^2]
    :param maximum_panel_power [kW]
    :param expansion_factor expands area by given multiplier to account for additional space requirements
    :return: capacity [DC kW]
    """
    panel_power_density = maximum_panel_power / area_per_panel
    array_power_density = panel_power_density * gcr
    return pv_area * array_power_density / expansion_factor


def calculate_turbine_radius(turb_size):
    """
    Using model fitted from SAM's turbine library, get an estimate of the turbine radius from the kW rating
    rating = 0.245*d^2.01, R-squared=0.976
    """
    return (turb_size / 0.245) ** (1 / 2.01) / 2.0


def optimize_solar_wind_AEP(scenario):
    """
    Assume single axis tracking with Premiun modules
    :param scenario
    :return: negative of solar and wind annual energy production
    """
    
    scenario.run_single(get_output=False)
    wind_aep = scenario.systems['Wind']['Windpower'].Outputs.annual_energy / 1000
    solar_aep = scenario.systems['Pvwatts']['Pvwattsv5'].Outputs.annual_energy / 1000
    
    return -wind_aep - solar_aep
