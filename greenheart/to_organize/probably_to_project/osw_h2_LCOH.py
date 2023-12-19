import os
import sys
sys.path.append('')
from dotenv import load_dotenv
import pandas as pd
import json
from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.technologies.sites import flatirons_site as sample_site
from hopp.utilities.keys import set_developer_nrel_gov_key
# from plot_reopt_results import plot_reopt_results
# from run_reopt import run_reopt
from greenheart.to_organize.H2_Analysis.hopp_for_h2 import hopp_for_h2
from greenheart.to_organize.H2_Analysis.hopp_for_h2 import run_h2a as run_h2a #no h2a function
from greenheart.to_organize.H2_Analysis.simple_dispatch import SimpleDispatch
from greenheart.to_organize.H2_Analysis.simple_cash_annuals import simple_cash_annuals
import hopp.simulation.technologies.hydrogen.electrolysis.run_h2_PEM as run_h2_PEM
import numpy as np
import numpy_financial as npf
from lcoe.lcoe import lcoe as lcoe_calc
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
from hopp.tools.resource import *
from hopp.tools.resource.resource_loader import site_details_creator

from greenheart.to_organize import hopp_tools_steel
import copy
from greenheart.to_organize import plot_results
from greenheart.to_organize import run_profast_for_hydrogen
from greenheart.to_organize.hopp_tools_steel import hoppDict
import yaml

"""
OSW-H2 LCOH analysis
"""

#Set API key
load_dotenv()
NREL_API_KEY = os.getenv("NREL_API_KEY")
set_developer_nrel_gov_key('NREL_API_KEY')  # Set this key manually here if you are not setting it using the .env

#Step 1: User Inputs for scenario
save_hybrid_plant_yaml = True # hybrid_plant requires special processing of the SAM objects
save_model_input_yaml = True # saves the inputs for each model/major function
save_model_output_yaml = True # saves the outputs for each model/major function

resource_year = 2013
atb_years = [
            2025,
            # 2030,
            # 2035
            ]
'''
Tech neutral ITC and PTC for years after 2022
Wind PTC [$/kWh] is in 1992 dollars adjusted for inflation in run_pyfast_for_hydrogen.py
H2 PTC [$/kg] is in 2022 dollars adjusted for inflation in run_pyfast_for_hydrogen.py
Wind ITC [%] applied to wind capital expenditure and HVDC cabling in run_pyfast_for_hydrogen.py
Base: Base credit only 100% full valuation. No prevailing wages. No bonus credits.
Max: 100% full valuation credit. Prevailing wages met.
Bonus: 100% full valuation credit. Prevailing wages met. Bonus content credit met.
'''
policy = {
    'No Policy': {'Wind ITC': 0, 'Wind PTC': 0, "H2 PTC": 0},
    # 'Base PTC': {'Wind ITC': 0, 'Wind PTC': 0.003, "H2 PTC": 0.60},
    # 'Max PTC': {'Wind ITC': 0, 'Wind PTC': 0.015, "H2 PTC": 3.00},
    # 'Base ITC': {'Wind ITC': 0.06, 'Wind PTC': 0.00, "H2 PTC": 0.60},
    # 'Max ITC': {'Wind ITC': 0.30, 'Wind PTC': 0.00, "H2 PTC": 3.00},
    # 'Bonus PTC': {'Wind ITC': 0, 'Wind PTC': 0.0165, "H2 PTC": 3.00},
    # 'Bonus ITC': {'Wind ITC': 0.40, 'Wind PTC': 0.00, "H2 PTC": 3.00}
}

sample_site['year'] = resource_year
useful_life = 30
critical_load_factor = 1
run_reopt_flag = False
custom_powercurve = False   #A flag that is applicable when using PySam WindPower (not FLORIS)
storage_used = False
battery_can_grid_charge = False
grid_connected_hopp = False
floris = True      #TODO: Set to True and get the floris files working

# Technology sizing
interconnection_size_mw = 1000
electrolyzer_size_mw = 1000
wind_size_mw = 1000
solar_size_mw = 0
storage_size_mw = 0
storage_size_mwh = 0

#TODO: Should all turbines be used for all years?
turbine_name = [
                '12MW',
                # '15MW',
                # '18MW'
                ]


scenario_choice = 'Offshore Wind-H2 Analysis'

site_selection = [
                'Site 1',
                # 'Site 2',
                # 'Site 3',
                # 'Site 4'
                ]

scenario = dict()
kw_continuous = electrolyzer_size_mw * 1000
load = [kw_continuous for x in
        range(0, 8760)]  # * (sin(x) + pi) Set desired/required load profile for plant

#Site lat and lon will be set by data loaded from Orbit runs

# Financial inputs
discount_rate = 0.10 # nominal return based on 2022 ATB basline workbook
debt_equity_split = 68.5    # 2022 ATB uses 68.5% debt

# Wind costs input from ORBIT analysis
h2_model ='Simple'  #Basic cost model based on H2a and HFTO program record for PEM electrolysis
# h2_model = 'H2A'

# These inputs are not used in this analysis (no solar or storage)
solar_cost_kw = 9999
storage_cost_kw = 250
storage_cost_kwh = 240

# Flags (TODO: remove them and update documentation)
forced_sizes = True # no REopt
force_electrolyzer_cost = False


# Enable Ability to purchase/sell electricity to/from grid. Price Defined in $/kWh
# sell_price = 0.01
# buy_price = 0.01
sell_price = False
buy_price = False

# Set paths for results, floris and orbit
parent_path = os.path.abspath('')
results_dir = parent_path + '/examples/H2_Analysis/results/'
results = pd.DataFrame()
floris_dir = parent_path + '/floris_input_files/'

# Site specific turbine information
path = ('examples/H2_Analysis/OSW_H2_sites_turbines_and_costs.xlsx')
xl = pd.ExcelFile(path)

# which plots to show
plot_power_production = True
plot_battery = True
plot_grid = True
plot_h2 = True
plot_desal = True
plot_wind = True
plot_hvdcpipe = True
plot_hvdcpipe_lcoh = True


for option in policy:
    for atb_year in atb_years:
        for site_location in site_selection:
            site_number = site_location.split(' ')[1]
            for turbine_model in turbine_name:

                hopp_dict = hoppDict(save_model_input_yaml, save_model_output_yaml)
                sub_dict = {
                    'policy': policy[option],
                    'atb_year': atb_year,
                    'site_location': site_location,
                    'parent_path': parent_path,
                    'load': load,
                    'kw_continuous': kw_continuous,
                    'sample_site': sample_site,
                    'discount_rate': discount_rate,
                    'forced_sizes': forced_sizes,
                    'force_electrolyzer_cost': force_electrolyzer_cost,
                    'wind_size': wind_size_mw,
                    'solar_size': solar_size_mw,
                    'storage_size_mw': storage_size_mw,
                    'storage_size_mwh': storage_size_mwh,
                    'solar_cost_kw': solar_cost_kw,
                    # 'storage_cost_kw': storage_cost_kw,
                    # 'storage_cost_kwh': storage_cost_kwh,
                    'debt_equity_split': debt_equity_split,
                    'useful_life': useful_life,
                    'critical_load_factor': critical_load_factor,
                    'run_reopt_flag': run_reopt_flag,
                    'custom_powercurve': custom_powercurve,
                    'storage_used': storage_used,
                    'battery_can_grid_charge': battery_can_grid_charge,
                    'grid_connected_hopp': grid_connected_hopp,
                    'interconnection_size_mw': interconnection_size_mw,
                    'electrolyzer_size_mw': electrolyzer_size_mw,
                    'scenario':
                        {
                            'Useful Life': useful_life,
                            'Debt Equity': debt_equity_split,
                            'discount_rate': discount_rate,
                        },
                    'sell_price': False,
                    'buy_price': False,
                    'h2_model': h2_model,
                    'results_dir': results_dir,
                    'scenario_choice': scenario_choice,
                }

                hopp_dict.add('Configuration', sub_dict)

                plot_dict = {
                    'plot':
                        {
                            'plot_power_production': True,
                            'plot_battery': True,
                            'plot_grid': True,
                            'plot_h2': True,
                            'plot_desal': True,
                            'plot_wind': True,
                            'plot_hvdcpipe': True,
                            'plot_hvdcpipe_lcoh': True,
                        }
                }

                hopp_dict.add('Configuration', plot_dict)


                # set policy values
                hopp_dict, scenario, policy_option = hopp_tools_steel.set_policy_values(hopp_dict, scenario, policy, option)
                print(scenario['Wind PTC'])

                turbinesheet = turbine_model[-4:]
                scenario_df = xl.parse(turbinesheet)
                scenario_df.set_index(["Parameter"], inplace = True)

                site_df = scenario_df[site_location]

                # turbine_model = str(site_df['Turbine rating'])+'MW'

                turbine_rating = site_df['Turbine rating']

                # set turbine values
                hopp_dict, scenario, nTurbs, floris_config = hopp_tools_steel.set_turbine_model(hopp_dict, turbine_model, scenario, parent_path,floris_dir, floris)

                scenario['Useful Life'] = useful_life

                # financials
                hopp_dict, scenario = hopp_tools_steel.set_financial_info(hopp_dict, scenario, debt_equity_split, discount_rate)

                # set electrolyzer information
                #TODO: Verify that we want to use conservative replacement scenario and costs for different years
                electrolysis_scale = 'Centralized'
                electrolyzer_replacement_scenario = 'Conservative'
                hopp_dict, electrolyzer_capex_kw,capex_ratio_dist, electrolyzer_energy_kWh_per_kg, time_between_replacement =  hopp_tools_steel.set_electrolyzer_info(hopp_dict,atb_year,electrolysis_scale,electrolyzer_replacement_scenario,turbine_rating,direct_coupling = False)

                # Extract Scenario Information from ORBIT Runs
                # Load Excel file of scenarios
                # OSW sites and cost file including turbines 8/16/2022

                # site info
                hopp_dict, site_df, sample_site = hopp_tools_steel.set_site_info(hopp_dict, site_df, sample_site)

                site_name = site_df['Representative region']
                fixed_or_floating_wind = site_df['Substructure technology']
                site = SiteInfo(sample_site, hub_height=scenario['Tower Height'])

                hopp_dict.add('Configuration', {'site': site})

                #Assign Orbit results to scenario cost details
                total_capex = site_df['Total CapEx']
                wind_cost_kw = copy.deepcopy(total_capex)
                wind_om_cost_kw = site_df['OpEx, $/kW-yr']
                wind_net_cf = site_df['Assumed NCF']

                # set export financials
                wind_cost_kw, wind_om_cost_kw, total_export_system_cost, total_export_om_cost = hopp_tools_steel.set_export_financials(wind_size_mw,
                                                                                                                                wind_cost_kw,
                                                                                                                                wind_om_cost_kw,
                                                                                                                                useful_life,
                                                                                                                                site_df)

                # set wind financials
                new_wind_cost_kw, new_wind_om_cost_kw, new_wind_net_cf = hopp_tools_steel.set_turbine_financials(turbine_model,
                                                                                                            fixed_or_floating_wind,
                                                                                                            atb_year,
                                                                                                            wind_cost_kw,
                                                                                                            wind_om_cost_kw,
                                                                                                            wind_net_cf,
                                                                                                            parent_path)

                wind_cost_kw = copy.deepcopy(new_wind_cost_kw)
                wind_om_cost_kw = copy.deepcopy(new_wind_om_cost_kw)


                #Plot Wind Data to ensure offshore data is sound
                wind_data = site.wind_resource._data['data']
                wind_speed = [W[2] for W in wind_data]
                plot_results.plot_wind_results(wind_data, site_name, site_df['Representative coordinates'], results_dir, plot_wind)
                print("Wind speed check: ", np.max(wind_speed))

                # Run HOPP
                print('Site name: ', site_name, 'Turbine rating: ',turbine_rating, 'Total Capex: ', total_capex, 'ATB Capex', new_wind_cost_kw)
                hopp_dict, combined_pv_wind_power_production_hopp, energy_shortfall_hopp, combined_pv_wind_curtailment_hopp, hybrid_plant, wind_size_mw, solar_size_mw, lcoe = \
                hopp_tools_steel.run_HOPP(
                    hopp_dict,
                    scenario,
                    site,
                    sample_site,
                    forced_sizes,
                    solar_size_mw,
                    wind_size_mw,
                    storage_size_mw,
                    storage_size_mwh,
                    wind_cost_kw,
                    solar_cost_kw,
                    storage_cost_kw,
                    storage_cost_kwh,
                    kw_continuous,
                    load,
                    electrolyzer_size_mw,
                    wind_om_cost_kw,
                    nTurbs,
                    floris_config,
                    floris,
                )

                wind_plant_size = hybrid_plant.wind.system_capacity_kw
                print('Wind plant size: ',hybrid_plant.wind.system_capacity_kw,\
                    )

                #Step 4: Plot HOPP Results
                plot_results.plot_HOPP(combined_pv_wind_power_production_hopp,
                                        energy_shortfall_hopp,
                                        combined_pv_wind_curtailment_hopp,
                                        load,
                                        results_dir,
                                        site_name,
                                        atb_year,
                                        turbine_model,
                                        hybrid_plant,
                                        plot_power_production)

                #Step 5: Run Simple Dispatch Model (no battery is used in system)
                hopp_dict, combined_pv_wind_storage_power_production_hopp, battery_SOC, battery_used, excess_energy = \
                    hopp_tools_steel.run_battery(
                    hopp_dict,
                    energy_shortfall_hopp,
                    combined_pv_wind_curtailment_hopp,
                    combined_pv_wind_power_production_hopp
                    )

                plot_results.plot_battery_results(
                    combined_pv_wind_curtailment_hopp,
                    energy_shortfall_hopp,
                    combined_pv_wind_storage_power_production_hopp,
                    combined_pv_wind_power_production_hopp,
                    battery_SOC,
                    battery_used,
                    results_dir,
                    site_name,atb_year,turbine_model,
                    load,
                    plot_battery,
                )

                # grid information (no grid is used in system)
                hopp_dict, cost_to_buy_from_grid, profit_from_selling_to_grid, energy_to_electrolyzer = hopp_tools_steel.grid(
                    hopp_dict,
                    combined_pv_wind_storage_power_production_hopp,
                    sell_price,
                    excess_energy,
                    buy_price,
                    kw_continuous,
                    plot_grid,
                )

                #Step 6: Run the H2_PEM model
                h2_model = 'Simple'
                hopp_dict, H2_Results, H2A_Results, electrical_generation_timeseries = hopp_tools_steel.run_H2_PEM_sim(
                    hopp_dict,
                    hybrid_plant,
                    energy_to_electrolyzer,
                    scenario,
                    wind_size_mw,
                    solar_size_mw,
                    electrolyzer_size_mw,
                    kw_continuous,
                    electrolyzer_capex_kw,
                    lcoe,
                )

                plot_results.plot_h2_results(H2_Results,
                                electrical_generation_timeseries,
                                results_dir,
                                site_name,atb_year,turbine_model,
                                load,
                                plot_h2)

                #Step 6b: Run desal model
                hopp_dict, desal_capex, desal_opex = hopp_tools_steel.desal_model(
                    hopp_dict,
                    H2_Results,
                    electrolyzer_size_mw,
                    electrical_generation_timeseries,
                    useful_life,
                )

                # compressor model
                hopp_dict,compressor, compressor_results = hopp_tools_steel.compressor_model(hopp_dict)

                #Pressure Vessel Model Example
                hopp_dict,storage_input, storage_output = hopp_tools_steel.pressure_vessel(hopp_dict)

                # pipeline model
                total_h2export_system_cost, opex_pipeline, dist_to_port_value = hopp_tools_steel.pipeline(site_df,
                                                                                H2_Results,
                                                                                useful_life,
                                                                                storage_input)


                # plot HVDC vs pipe
                plot_results.plot_hvdcpipe(total_export_system_cost,
                                            total_h2export_system_cost,
                                            site_name,
                                            atb_year,
                                            dist_to_port_value,
                                            results_dir)

                revised_renewable_cost = hybrid_plant.grid.total_installed_cost
                hydrogen_annual_production = H2_Results['hydrogen_annual_output']
                hydrogen_storage_capacity_kg = 0
                hydrogen_storage_cost_USDprkg = 0
                water_cost = 0.006868 #($/gal) average of green steel sites' water cost


                h2_ptc = scenario['H2 PTC']
                wind_ptc = scenario['Wind PTC']
                wind_itc = scenario['Wind ITC']

                print(revised_renewable_cost)
                #Run HVDC export scenario
                export_hvdc = True  #HVDC scenario
                h2a_solution,h2a_summary,lcoh_breakdown,electrolyzer_installed_cost_kw = run_profast_for_hydrogen.run_profast_for_hydrogen(site_name,electrolyzer_size_mw,H2_Results,\
                                electrolyzer_capex_kw,time_between_replacement,hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg,\
                                desal_capex,desal_opex,useful_life,atb_year,water_cost,wind_size_mw,solar_size_mw, \
                                hybrid_plant,revised_renewable_cost,wind_om_cost_kw,
                                total_export_system_cost,
                                total_export_om_cost,
                                export_hvdc, combined_pv_wind_storage_power_production_hopp,
                                grid_connected_hopp,h2_ptc,wind_ptc,wind_itc)
                lcoh = h2a_solution['price']
                print('LCOH: ', lcoh)
                # # Max hydrogen production rate [kg/hr]
                max_hydrogen_production_rate_kg_hr = np.max(H2_Results['hydrogen_hourly_production'])
                max_hydrogen_delivery_rate_kg_hr  = np.mean(H2_Results['hydrogen_hourly_production'])

                electrolyzer_capacity_factor = H2_Results['cap_factor']

                test = dict()

                test['Site Name'] = site_name
                test['ATB Year'] = atb_year
                test['HVDC Export'] = export_hvdc
                test['Pipeline Export'] = False
                test['Plant life'] = useful_life
                test['Policy'] = option
                test['LCOH: total ($/kg)'] = lcoh_breakdown['LCOH: total ($/kg)']
                test['Turbine size (MW)'] = turbine_rating
                test['Wind Plant size (MW)'] = wind_size_mw
                test['Wind Plant Size Adjusted for Turbine Rating(MW)'] = wind_plant_size /1000
                test['Electrolyzer size (MW)'] = electrolyzer_size_mw
                test['Load Profile (kW)'] = kw_continuous
                test['Energy to Electrolyzer (kW)'] = np.sum(energy_to_electrolyzer)
                test['Wind capacity factor (%)'] = hybrid_plant.wind.capacity_factor
                test['Electrolyzer capacity factor (%)'] = electrolyzer_capacity_factor
                test['H2 production (kg)'] = hydrogen_annual_production
                test['LCOH ($/kg)'] = lcoh
                test['LCOH: Compression & storage ($/kg)'] = lcoh_breakdown['LCOH: Compression & storage ($/kg)']
                test['LCOH: Electrolyzer CAPEX ($/kg)']= lcoh_breakdown['LCOH: Electrolyzer CAPEX ($/kg)']
                test['LCOH: Desalination CAPEX ($/kg)'] = lcoh_breakdown['LCOH: Desalination CAPEX ($/kg)']
                test['LCOH: HVDC Export CAPEX ($/kg)'] = lcoh_breakdown['LCOH: HVDC Export CAPEX ($/kg)']
                test['LCOH: Pipeline Export CAPEX ($/kg)'] = lcoh_breakdown['LCOH: Pipeline Export CAPEX ($/kg)']
                test['LCOH: HVDC Export FOM ($/kg)'] = lcoh_breakdown['LCOH: HVDC Export FOM ($/kg)']
                test['LCOH: Pipeline Export FOM ($/kg)'] = lcoh_breakdown['LCOH: Pipeline Export FOM ($/kg)']
                test['LCOH: Electrolyzer FOM ($/kg)'] = lcoh_breakdown['LCOH: Electrolyzer FOM ($/kg)']
                test['LCOH: Electrolyzer VOM ($/kg)']=lcoh_breakdown['LCOH: Electrolyzer VOM ($/kg)']
                test['LCOH: Desalination FOM ($/kg)'] = lcoh_breakdown['LCOH: Desalination FOM ($/kg)']
                test['LCOH: Renewable plant ($/kg)'] = lcoh_breakdown['LCOH: Renewable plant ($/kg)']
                test['LCOH: Renewable FOM ($/kg)']= lcoh_breakdown['LCOH: Renewable FOM ($/kg)']
                test['LCOH: Taxes ($/kg)']=lcoh_breakdown['LCOH: Taxes ($/kg)']
                test['LCOH: Water consumption ($/kg)'] = lcoh_breakdown['LCOH: Water consumption ($/kg)']
                test['LCOH: Finances ($/kg)'] = lcoh_breakdown['LCOH: Finances ($/kg)']



                test = pd.DataFrame(test,index=[0])
                print(test)
                results = pd.concat([results,test])

                #Run pipeline export system financials
                export_hvdc = False
                h2a_solution,h2a_summary,lcoh_breakdown,electrolyzer_installed_cost_kw = run_profast_for_hydrogen.run_profast_for_hydrogen(site_name,electrolyzer_size_mw,H2_Results,\
                    electrolyzer_capex_kw,time_between_replacement,hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg,\
                    desal_capex,desal_opex,useful_life,atb_year,water_cost,wind_size_mw,solar_size_mw, \
                    hybrid_plant,revised_renewable_cost,wind_om_cost_kw,
                    total_h2export_system_cost,
                    opex_pipeline,
                    export_hvdc, combined_pv_wind_storage_power_production_hopp,
                    grid_connected_hopp,h2_ptc,wind_ptc,wind_itc)
                lcoh = h2a_solution['price']
                print('LCOH: ', lcoh)
                # # Max hydrogen production rate [kg/hr]
                max_hydrogen_production_rate_kg_hr = np.max(H2_Results['hydrogen_hourly_production'])
                max_hydrogen_delivery_rate_kg_hr  = np.mean(H2_Results['hydrogen_hourly_production'])

                electrolyzer_capacity_factor = H2_Results['cap_factor']

                test = dict()

                test['Site Name'] = site_name
                test['ATB Year'] = atb_year
                test['HVDC Export'] = export_hvdc
                test['Pipeline Export'] = True
                test['Plant life'] = useful_life
                test['Policy'] = option
                test['Turbine size (MW)'] = turbine_rating
                test['LCOH: total ($/kg)'] = lcoh_breakdown['LCOH: total ($/kg)']
                test['Wind Plant size (MW)'] = wind_size_mw
                test['Wind Plant Size Adjusted for Turbine Rating(MW)'] = wind_plant_size /1000
                test['Electrolyzer size (MW)'] = electrolyzer_size_mw
                test['Load Profile (kW)'] = kw_continuous
                test['Energy to Electrolyzer (kW)'] = np.sum(energy_to_electrolyzer)
                test['Wind capacity factor (%)'] = hybrid_plant.wind.capacity_factor
                test['Electrolyzer capacity factor (%)'] = electrolyzer_capacity_factor
                test['H2 production (kg)'] = hydrogen_annual_production
                test['LCOH ($/kg)'] = lcoh
                test['LCOH: Compression & storage ($/kg)'] = lcoh_breakdown['LCOH: Compression & storage ($/kg)']
                test['LCOH: Electrolyzer CAPEX ($/kg)']= lcoh_breakdown['LCOH: Electrolyzer CAPEX ($/kg)']
                test['LCOH: Desalination CAPEX ($/kg)'] = lcoh_breakdown['LCOH: Desalination CAPEX ($/kg)']
                test['LCOH: HVDC Export CAPEX ($/kg)'] = lcoh_breakdown['LCOH: HVDC Export CAPEX ($/kg)']
                test['LCOH: Pipeline Export CAPEX ($/kg)'] = lcoh_breakdown['LCOH: Pipeline Export CAPEX ($/kg)']
                test['LCOH: HVDC Export FOM ($/kg)'] = lcoh_breakdown['LCOH: HVDC Export FOM ($/kg)']
                test['LCOH: Pipeline Export FOM ($/kg)'] = lcoh_breakdown['LCOH: Pipeline Export FOM ($/kg)']
                test['LCOH: Electrolyzer FOM ($/kg)'] = lcoh_breakdown['LCOH: Electrolyzer FOM ($/kg)']
                test['LCOH: Electrolyzer VOM ($/kg)']=lcoh_breakdown['LCOH: Electrolyzer VOM ($/kg)']
                test['LCOH: Desalination FOM ($/kg)'] = lcoh_breakdown['LCOH: Desalination FOM ($/kg)']
                test['LCOH: Renewable plant ($/kg)'] = lcoh_breakdown['LCOH: Renewable plant ($/kg)']
                test['LCOH: Renewable FOM ($/kg)']= lcoh_breakdown['LCOH: Renewable FOM ($/kg)']
                test['LCOH: Taxes ($/kg)']=lcoh_breakdown['LCOH: Taxes ($/kg)']
                test['LCOH: Water consumption ($/kg)'] = lcoh_breakdown['LCOH: Water consumption ($/kg)']
                test['LCOH: Finances ($/kg)'] = lcoh_breakdown['LCOH: Finances ($/kg)']




                test = pd.DataFrame(test,index=[0])
                print(test)
                results = pd.concat([results,test])

results.to_csv(os.path.join(results_dir, "H2_Analysis_osw_h2.csv"))
print('Done')