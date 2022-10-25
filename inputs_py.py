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
    save_outputs_dict['ATB Year'] = list()
    save_outputs_dict['Policy Option'] = list()
    save_outputs_dict['Resource Year'] = list()
    save_outputs_dict['Turbine Model'] = list()
    save_outputs_dict['Critical Load Factor'] = list()
    save_outputs_dict['System Load (kW)'] = list()
    save_outputs_dict['Useful Life'] = list()
    save_outputs_dict['Wind PTC ($/kW)'] = list()
    save_outputs_dict['H2 PTC ($/kg)'] = list()
    save_outputs_dict['ITC (%)'] = list()
    save_outputs_dict['Discount Rate (%)'] = list()
    save_outputs_dict['Debt Equity'] = list()
    save_outputs_dict['Hub Height (m)'] = list()
    save_outputs_dict['Storage Enabled'] = list()
    save_outputs_dict['Wind Cost ($/kW)'] = list()
    save_outputs_dict['Solar Cost ($/kW)'] = list()
    save_outputs_dict['Storage Cost ($/kW)'] = list()
    save_outputs_dict['Storage Cost ($/kWh)'] = list()
    save_outputs_dict['Electrolyzer Cost ($/kW)'] = list()
    save_outputs_dict['Storage Hours'] = list()
    save_outputs_dict['Wind built (MW)'] = list()
    save_outputs_dict['Solar built (MW)'] = list()
    save_outputs_dict['Storage built (MW)'] = list()
    save_outputs_dict['Storage built (MWh)'] = list()
    save_outputs_dict['Electrolyzer built (MW)'] = list()
    save_outputs_dict['Battery Can Grid Charge'] = list()
    save_outputs_dict['Grid Connected HOPP'] = list()
    save_outputs_dict['Built Interconnection Size (MW)'] = list()
    save_outputs_dict['Wind Total Installed Cost ($)'] = list()
    save_outputs_dict['Desal Total Installed Cost ($)'] = list()
    save_outputs_dict['HVDC Total Installed Cost ($)'] = list()
    save_outputs_dict['Pipeline Total Installed Cost ($)'] = list()
    save_outputs_dict['Electrolyzer Total Installed Capital Cost ($)'] = list()
    save_outputs_dict['Total Plant Capital Cost HVDC ($)'] = list()
    save_outputs_dict['Total Plant Capital Cost Pipeline ($)'] = list()
    save_outputs_dict['Wind Total Operating Cost ($)'] = list()
    save_outputs_dict['Wind Fixed Operating Cost ($)'] = list()
    save_outputs_dict['Electrolyzer Total Operating Cost ($)'] = list()
    save_outputs_dict['Desal Total Operating Cost ($)'] = list()
    save_outputs_dict['HVDC Total Operating Cost ($)'] = list()
    save_outputs_dict['Pipeline Total Operating Cost ($)'] = list()
    save_outputs_dict['Total Plant Operating Cost HVDC ($)'] = list()
    save_outputs_dict['Total Plant Operating Cost Pipeline ($)'] = list()
    save_outputs_dict['Real LCOE ($/MWh)'] = list()
    save_outputs_dict['Nominal LCOE ($/MWh)'] = list()
    save_outputs_dict['Total Annual H2 production (kg)'] = list()
    save_outputs_dict['H2 total tax credit ($)'] = list()
    save_outputs_dict['Total ITC HVDC scenario ($)'] = list()
    save_outputs_dict['Total ITC Pipeline scenario ($)'] = list()
    save_outputs_dict['TLCC Wind ($)'] = list()
    save_outputs_dict['TLCC H2 ($)'] = list()
    save_outputs_dict['TLCC Desal ($)'] = list()
    save_outputs_dict['TLCC HVDC ($)'] = list()
    save_outputs_dict['TLCC Pipeline ($)'] = list()
    save_outputs_dict['LCOH Wind contribution ($/kg)'] = list()
    save_outputs_dict['LCOH H2 contribution ($/kg)'] = list()
    save_outputs_dict['LCOH Desal contribution ($/kg)'] = list()
    save_outputs_dict['LCOH HVDC contribution ($/kg)'] = list()
    save_outputs_dict['LCOH Pipeline contribution ($/kg)'] = list()    
    save_outputs_dict['H_LCOE HVDC scenario no operating costs (uses LCOE calculator) ($/kg)'] = list()
    save_outputs_dict['H_LCOE HVDC scenario w/ operating costs (uses LCOE calculator) ($/kg)'] = list()
    save_outputs_dict['H_LCOE Pipeline scenario no operating costs (uses LCOE calculator) ($/kg)'] = list()
    save_outputs_dict['H_LCOE Pipeline scenario w/ operating costs (uses LCOE calculator) ($/kg)'] = list()
    save_outputs_dict['Levelized Cost/kg H2 HVDC scenario (CF Method - using annual cashflows per technology) ($/kg)'] = list()
    save_outputs_dict['Levelized Cost/kg H2 Pipeline scenario (CF Method - using annual cashflows per technology) ($/kg)'] = list()
    save_outputs_dict['Gut-Check Cost/kg H2 Pipeline scenario (non-levelized, includes elec if used) ($/kg)'] = list()
    save_outputs_dict['Gut-Check Cost/kg H2 HVDC scenario (non-levelized, includes elec if used) ($/kg)'] = list()
    save_outputs_dict['Grid Connected HOPP'] = list()
    save_outputs_dict['HOPP Total Electrical Generation (kWh)'] = list()
    save_outputs_dict['Total Yearly Electrical Generation used by Electrolyzer (kWh)'] = list()
    save_outputs_dict['Wind Capacity Factor (%)'] = list()
    save_outputs_dict['Electrolyzer Capacity Factor (%)'] = list()
    save_outputs_dict['HOPP Annual Energy Shortfall (kWh)'] = list()
    save_outputs_dict['HOPP Annual Energy Curtailment (kWh)'] = list()
    save_outputs_dict['Battery Generation (kWh)'] = list()
    save_outputs_dict['Electricity to Grid (kWh)'] = list()
    
    return save_outputs_dict

def save_the_things():
    save_outputs_dict = dict()
    save_outputs_dict['Site Name'] = (site_name)
    save_outputs_dict['Substructure Technology'] = (site_df['Substructure technology'])
    save_outputs_dict['ATB Year'] = (atb_year)
    save_outputs_dict['Policy Option'] = (policy_option)
    save_outputs_dict['Resource Year'] = (resource_year)
    save_outputs_dict['Turbine Model'] = (turbine_model)
    save_outputs_dict['Critical Load Factor'] = (critical_load_factor)
    save_outputs_dict['System Load (kW)'] = (kw_continuous)
    save_outputs_dict['Useful Life'] = (useful_life)
    save_outputs_dict['Wind PTC ($/kW)'] = (scenario['Wind PTC'])
    save_outputs_dict['H2 PTC ($/kg)'] = (scenario['H2 PTC'])
    save_outputs_dict['ITC (%)'] = (scenario['Wind ITC'])
    save_outputs_dict['Discount Rate (%)'] = (discount_rate*100)
    save_outputs_dict['Debt Equity'] = (debt_equity_split)
    save_outputs_dict['Hub Height (m)'] = (tower_height)
    save_outputs_dict['Storage Enabled'] = (storage_used)
    save_outputs_dict['Wind Cost ($/kW)'] = (wind_cost_kw)
    save_outputs_dict['Solar Cost ($/kW)'] = (solar_cost_kw)
    save_outputs_dict['Storage Cost ($/kW)'] = (storage_cost_kw)
    save_outputs_dict['Storage Cost ($/kWh)'] = (storage_cost_kwh)
    save_outputs_dict['Electrolyzer Cost ($/kW)'] = (electrolyzer_capex_kw)
    save_outputs_dict['Storage Hours'] = (storage_hours)
    save_outputs_dict['Wind built (MW)'] = (wind_size_mw)
    save_outputs_dict['Solar built (MW)'] = (solar_size_mw)
    save_outputs_dict['Storage built (MW)'] = (storage_size_mw)
    save_outputs_dict['Storage built (MWh)'] = (storage_size_mwh)
    save_outputs_dict['Electrolyzer built (MW)'] = (electrolyzer_size_mw)
    save_outputs_dict['Battery Can Grid Charge'] = (battery_can_grid_charge)
    save_outputs_dict['Grid Connected HOPP'] = (grid_connected_hopp)
    save_outputs_dict['Built Interconnection Size (MW)'] = (hybrid_plant.interconnect_kw)
    save_outputs_dict['Wind Total Installed Cost ($)'] = (total_hopp_installed_cost)
    save_outputs_dict['Desal Total Installed Cost ($)'] = (total_desal_cost)
    save_outputs_dict['HVDC Total Installed Cost ($)'] = (total_export_system_cost)
    save_outputs_dict['Pipeline Total Installed Cost ($)'] = (total_h2export_system_cost)
    save_outputs_dict['Electrolyzer Total Installed Capital Cost ($)'] = (electrolyzer_total_capital_cost)
    save_outputs_dict['Total Plant Capital Cost HVDC ($)'] = (total_system_installed_cost_hvdc)
    save_outputs_dict['Total Plant Capital Cost Pipeline ($)'] = (total_system_installed_cost_pipeline)
    save_outputs_dict['Wind Total Operating Cost ($)'] = (annual_operating_cost_wind)
    save_outputs_dict['Wind Fixed Operating Cost ($)'] = (fixed_om_cost_wind)
    save_outputs_dict['Electrolyzer Total Operating Cost ($)'] = (annual_operating_cost_h2)
    save_outputs_dict['Desal Total Operating Cost ($)'] = (annual_operating_cost_desal)
    save_outputs_dict['HVDC Total Operating Cost ($)'] = (total_export_om_cost)
    save_outputs_dict['Pipeline Total Operating Cost ($)'] = (opex_pipeline)
    save_outputs_dict['Total Plant Operating Cost HVDC ($)'] = (total_annual_operating_costs_hvdc)
    save_outputs_dict['Total Plant Operating Cost Pipeline ($)'] = (total_annual_operating_costs_pipeline)
    save_outputs_dict['Real LCOE ($/MWh)'] = (lcoe*10)
    save_outputs_dict['Nominal LCOE ($/MWh)'] = (lcoe_nom*10)
    save_outputs_dict['Total Annual H2 production (kg)'] = (H2_Results['hydrogen_annual_output'])
    save_outputs_dict['H2 total tax credit ($)'] = (np.sum(h2_tax_credit))
    save_outputs_dict['Total ITC HVDC scenario ($)'] = (total_itc_hvdc)
    save_outputs_dict['Total ITC Pipeline scenario ($)'] = (total_itc_pipeline)
    save_outputs_dict['TLCC Wind ($)'] = (tlcc_wind_costs)
    save_outputs_dict['TLCC H2 ($)'] = (tlcc_h2_costs)
    save_outputs_dict['TLCC Desal ($)'] = (tlcc_desal_costs)
    save_outputs_dict['TLCC HVDC ($)'] = (tlcc_hvdc_costs)
    save_outputs_dict['TLCC Pipeline ($)'] = (tlcc_pipeline_costs)
    save_outputs_dict['LCOH Wind contribution ($/kg)'] = (LCOH_cf_method_wind)
    save_outputs_dict['LCOH H2 contribution ($/kg)'] = (LCOH_cf_method_h2_costs)
    save_outputs_dict['LCOH Desal contribution ($/kg)'] = (LCOH_cf_method_desal_costs)
    save_outputs_dict['LCOH HVDC contribution ($/kg)'] = (LCOH_cf_method_pipeline)
    save_outputs_dict['LCOH Pipeline contribution ($/kg)'] =  (LCOH_cf_method_hvdc) 
    save_outputs_dict['H_LCOE HVDC scenario no operating costs (uses LCOE calculator) ($/kg)'] = (h_lcoe_no_op_cost_hvdc)
    save_outputs_dict['H_LCOE HVDC scenario w/ operating costs (uses LCOE calculator) ($/kg)'] = (h_lcoe_hvdc)
    save_outputs_dict['H_LCOE Pipeline scenario no operating costs (uses LCOE calculator) ($/kg)'] = (h_lcoe_no_op_cost_pipeline)
    save_outputs_dict['H_LCOE Pipeline scenario w/ operating costs (uses LCOE calculator) ($/kg)'] = (h_lcoe_pipeline)
    save_outputs_dict['Levelized Cost/kg H2 HVDC scenario (CF Method - using annual cashflows per technology) ($/kg)'] = (LCOH_cf_method_total_hvdc)
    save_outputs_dict['Levelized Cost/kg H2 Pipeline scenario (CF Method - using annual cashflows per technology) ($/kg)'] = (LCOH_cf_method_total_pipeline)
    save_outputs_dict['Gut-Check Cost/kg H2 Pipeline scenario (non-levelized, includes elec if used) ($/kg)'] = (gut_check_h2_cost_kg_pipeline)
    save_outputs_dict['Gut-Check Cost/kg H2 HVDC scenario (non-levelized, includes elec if used) ($/kg)'] = (gut_check_h2_cost_kg_hvdc)
    save_outputs_dict['Grid Connected HOPP'] = (grid_connected_hopp)
    save_outputs_dict['HOPP Total Electrical Generation (kWh)'] = (np.sum(hybrid_plant.grid.generation_profile[0:8760]))
    save_outputs_dict['Total Yearly Electrical Generation used by Electrolyzer (kWh)'] = (total_elec_production)
    save_outputs_dict['Wind Capacity Factor (%)'] = (hybrid_plant.wind.capacity_factor)
    save_outputs_dict['Electrolyzer Capacity Factor (%)'] = (H2_Results['cap_factor']*100)
    save_outputs_dict['HOPP Annual Energy Shortfall (kWh)'] = (np.sum(energy_shortfall_hopp))
    save_outputs_dict['HOPP Annual Energy Curtailment (kWh)'] = (np.sum(combined_pv_wind_curtailment_hopp))
    save_outputs_dict['Battery Generation (kWh)'] = (np.sum(battery_used))
    save_outputs_dict['Electricity to Grid (kWh)'] = (np.sum(excess_energy))

    return save_outputs_dict