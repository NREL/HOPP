# inputs file

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

def set_inputs():

    #Step 1: User Inputs for scenario
    resource_year = 2013
    atb_years = [
                2022,
                # 2025,
                # 2030,
                # 2035
                ]
    policy = {
        'option 1': {'Wind ITC': 0, 'Wind PTC': 0, "H2 PTC": 0},
        # 'option 2': {'Wind ITC': 26, 'Wind PTC': 0, "H2 PTC": 0},
        # 'option 3': {'Wind ITC': 0, 'Wind PTC': 0.003, "H2 PTC": 0},
        # 'option 4': {'Wind ITC': 0, 'Wind PTC': 0.026, "H2 PTC": 0},
        # 'option 5': {'Wind ITC': 0, 'Wind PTC': 0.003, "H2 PTC": 0.6},
        'option 6': {'Wind ITC': 0, 'Wind PTC': 0.026, "H2 PTC": 3},
    }

    sample_site['year'] = resource_year
    useful_life = 30
    critical_load_factor = 1
    run_reopt_flag = False
    custom_powercurve = True
    storage_used = True
    battery_can_grid_charge = True
    grid_connected_hopp = False
    interconnection_size_mw = 1000
    electrolyzer_size = 1000

def establish_save_output_dict():
    """
    Establishes and returns a 'save_outputs_dict' dict
    for saving the relevant analysis variables for each site.
    """

    save_outputs_dict = dict()
    save_outputs_dict['Site Name'] = list()
    save_outputs_dict['Substructure Technology'] = list()
    save_outputs_dict['Site Lat'] = list()
    save_outputs_dict['Site Lon'] = list()
    save_outputs_dict['ATB Year'] = list()
    save_outputs_dict['Resource Year'] = list()
    save_outputs_dict['Turbine Model'] = list()
    save_outputs_dict['Critical Load Factor'] = list()
    save_outputs_dict['System Load (kW)'] = list()
    save_outputs_dict['Useful Life'] = list()
    save_outputs_dict['Wind PTC'] = list()
    save_outputs_dict['H2 PTC'] = list()
    save_outputs_dict['Wind ITC'] = list()
    save_outputs_dict['Discount Rate'] = list()
    save_outputs_dict['Debt Equity'] = list()
    save_outputs_dict['Hub Height (m)'] = list()
    save_outputs_dict['Storage Enabled'] = list()
    save_outputs_dict['Wind Cost kW'] = list()
    save_outputs_dict['Solar Cost kW'] = list()
    save_outputs_dict['Storage Cost kW'] = list()
    save_outputs_dict['Storage Cost kWh'] = list()
    save_outputs_dict['Storage Hours'] = list()
    save_outputs_dict['Wind MW built'] = list()
    save_outputs_dict['Solar MW built'] = list()
    save_outputs_dict['Storage MW built'] = list()
    save_outputs_dict['Storage MWh built'] = list()
    save_outputs_dict['Electrolyzer Stack Size'] = list()
    save_outputs_dict['Electrolyzer Total System Size'] = list()
    save_outputs_dict['Battery Can Grid Charge'] = list()
    save_outputs_dict['Grid Connected HOPP'] = list()
    save_outputs_dict['Built Interconnection Size'] = list()
    save_outputs_dict['Total Installed Cost $(HOPP)'] = list()
    save_outputs_dict['LCOE'] = list()
    save_outputs_dict['Total Annual H2 production (kg)'] = list()
    save_outputs_dict['Gut-Check Cost/kg H2 (non-levelized, includes elec if used)'] = list()
    save_outputs_dict['Levelized Cost/kg H2 HVDC (CF Method - using annual cashflows per technology)'] = list()
    save_outputs_dict['Levelized Cost/kg H2 HVDC inc. Operating Cost (CF Method - using annual cashflows per technology)'] = list()
    save_outputs_dict['Levelized Cost/kg H2 Pipeline (CF Method - using annual cashflows per technology)'] = list()
    save_outputs_dict['Levelized Cost/kg H2 Pipeline inc. Operating Cost (CF Method - using annual cashflows per technology)'] = list()
    save_outputs_dict['Grid Connected HOPP'] = list()
    save_outputs_dict['HOPP Total Electrical Generation'] = list()
    save_outputs_dict['Total Yearly Electrical Generation used by Electrolyzer'] = list()
    save_outputs_dict['Wind Capacity Factor'] = list()
    save_outputs_dict['HOPP Energy Shortfall'] = list()
    save_outputs_dict['HOPP Curtailment'] = list()
    save_outputs_dict['Battery Generation'] = list()
    save_outputs_dict['Electricity to Grid'] = list()
    save_outputs_dict['Electrolyzer Stack Size'] = list()
    save_outputs_dict['Electrolyzer Total System Size'] = list()
    save_outputs_dict['H2A scaled total install cost'] = list()
    save_outputs_dict['H2A scaled total install cost per kw'] = list()
    
    return save_outputs_dict

def save_the_things():
    save_outputs_dict = dict()
    save_outputs_dict['Site Name'] = (site_name)
    save_outputs_dict['Substructure Technology'] = (site_df['Substructure technology'])
    save_outputs_dict['Site Lat'] = (lat)
    save_outputs_dict['Site Lon'] = (lon)
    save_outputs_dict['ATB Year'] = (atb_year)
    save_outputs_dict['Resource Year'] = (resource_year)
    save_outputs_dict['Turbine Model'] = (turbine_model)
    save_outputs_dict['Critical Load Factor'] = (critical_load_factor)
    save_outputs_dict['System Load (kW)'] = (load)
    save_outputs_dict['Useful Life'] = (useful_life)
    save_outputs_dict['Wind PTC'] = (scenario['Wind PTC'])
    save_outputs_dict['H2 PTC'] = (scenario['H2 PTC'])
    save_outputs_dict['Wind ITC'] = (scenario['Wind ITC'])
    save_outputs_dict['Discount Rate'] = (discount_rate)
    save_outputs_dict['Debt Equity'] = (debt_equity_split)
    save_outputs_dict['Hub Height (m)'] = (tower_height)
    save_outputs_dict['Storage Enabled'] = (storage_used)
    save_outputs_dict['Wind Cost kW'] = (wind_cost_kw)
    save_outputs_dict['Solar Cost kW'] = (solar_cost_kw)
    save_outputs_dict['Storage Cost kW'] = (storage_cost_kw)
    save_outputs_dict['Storage Cost kWh'] = (storage_cost_kwh)
    save_outputs_dict['Storage Hours'] = (storage_hours)
    save_outputs_dict['Wind MW built'] = (wind_size_mw)
    save_outputs_dict['Solar MW built'] = (solar_size_mw)
    save_outputs_dict['Storage MW built'] = (storage_size_mw)
    save_outputs_dict['Storage MWh built'] = (storage_size_mwh)
    save_outputs_dict['Battery Can Grid Charge'] = (battery_can_grid_charge)
    save_outputs_dict['Built Interconnection Size'] = (hybrid_plant.interconnect_kw)
    save_outputs_dict['Total Installed Cost $(HOPP)'] = (total_hopp_installed_cost)
    save_outputs_dict['LCOE'] = (lcoe)
    save_outputs_dict['Total Annual H2 production (kg)'] = (H2_Results['hydrogen_annual_output'])
    save_outputs_dict['Gut-Check Cost/kg H2 (non-levelized, includes elec if used)'] = (gut_check_h2_cost_kg)
    save_outputs_dict['Levelized Cost/kg H2 HVDC (CF Method - using annual cashflows per technology)'] = (LCOH_cf_method)
    save_outputs_dict['Levelized Cost/kg H2 HVDC inc. Operating Cost (CF Method - using annual cashflows per technology)'] = (LCOH_cf_method_w_operating_costs)
    save_outputs_dict['Levelized Cost/kg H2 Pipeline (CF Method - using annual cashflows per technology)'] = (LCOH_cf_method_pipeline)
    save_outputs_dict['Levelized Cost/kg H2 Pipeline inc. Operating Cost (CF Method - using annual cashflows per technology)'] = (LCOH_cf_method_w_operating_costs_pipeline)
    save_outputs_dict['Grid Connected HOPP'] = (grid_connected_hopp)
    save_outputs_dict['HOPP Total Electrical Generation'] = (np.sum(hybrid_plant.grid.generation_profile[0:8760]))
    save_outputs_dict['Total Yearly Electrical Generation used by Electrolyzer'] = (total_elec_production)
    save_outputs_dict['Wind Capacity Factor'] = (hybrid_plant.wind._system_model.Outputs.capacity_factor)
    save_outputs_dict['HOPP Energy Shortfall'] = (np.sum(energy_shortfall_hopp))
    save_outputs_dict['HOPP Curtailment'] = (np.sum(combined_pv_wind_curtailment_hopp))
    save_outputs_dict['Battery Generation'] = (np.sum(battery_used))
    save_outputs_dict['Electricity to Grid'] = (np.sum(excess_energy))
    save_outputs_dict['Electrolyzer Stack Size'] = (H2A_Results['electrolyzer_size'])
    save_outputs_dict['Electrolyzer Total System Size'] = (H2A_Results['total_plant_size'])
    save_outputs_dict['H2A scaled total install cost'] = (H2A_Results['scaled_total_installed_cost'])
    save_outputs_dict['H2A scaled total install cost per kw'] = (H2A_Results['scaled_total_installed_cost_kw'])
    return save_outputs_dict