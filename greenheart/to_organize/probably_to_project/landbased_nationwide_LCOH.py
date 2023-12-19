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
#from greenheart.to_organize import run_profast_for_hydrogen Unsure if it is meant to be run_profast
from greenheart.to_organize.hopp_tools_steel import hoppDict
import yaml

"""
Landbased LCOH analysis
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
            2030,
            2035
            ]
policy = {
    'No Policy': {'Wind ITC': 0, 'Wind PTC': 0, "H2 PTC": 0},
    'Base': {'Wind ITC': 0, 'Wind PTC': 0.0051, "H2 PTC": 0.6},
    'Max': {'Wind ITC': 0, 'Wind PTC': 0.0256, "H2 PTC": 3},
}

sample_site['year'] = resource_year
useful_life = 25
critical_load_factor = 1
run_reopt_flag = False
custom_powercurve = True    #A flag that is applicable when using PySam WindPower (not FLORIS)
storage_used = False
battery_can_grid_charge = False
grid_connected_hopp = False
floris = False

# Technology sizing
interconnection_size_mw = 100
electrolyzer_size_mw = 100
wind_size_mw = 100
solar_size_mw = 0
storage_size_mw = 0
storage_size_mwh = 0

scenario_choice = 'Landbased Analysis'


# Site details
N_lat = 5 #50 # number of data points
N_lon = 5 #95
desired_lats = np.linspace(23.833504, 49.3556, N_lat)
desired_lons = np.linspace(-129.22923, -65.7146, N_lon)
load_resource_from_file = False
resource_dir = Path(__file__).parent.parent.parent / "resource_files"
sitelist_name = 'filtered_site_details_{}_lats_{}_lons_{}_resourceyear'.format(N_lat, N_lon, resource_year)

if load_resource_from_file:
    # Loads resource files in 'resource_files', finds nearest files to 'desired_lats' and 'desired_lons'
    site_details = resource_loader_file(resource_dir, desired_lats, desired_lons, resource_year)  # Return contains
    site_details.to_csv(os.path.join(resource_dir, 'site_details.csv'))
    site_details = filter_sites(site_details, location='usa only')
else:
    # Creates the site_details file containing grid of lats, lons, years, and wind and solar filenames (blank
    # - to force API resource download)
    if os.path.exists(sitelist_name):
        site_details = pd.read_csv(sitelist_name)
    else:
        site_details = site_details_creator.site_details_creator(desired_lats, desired_lons, resource_dir)
        # Filter to locations in USA
        site_details = filter_sites(site_details, location='usa only')
        site_details.to_csv(sitelist_name)
print("Site Details Created")

scenario = dict()
kw_continuous = electrolyzer_size_mw * 1000
load = [kw_continuous for x in
        range(0, 8760)]  # * (sin(x) + pi) Set desired/required load profile for plant

#Site lat and lon will be set by data loaded from Orbit runs

# Financial inputs
discount_rate = 0.0824
debt_equity_split = 0

# Wind costs input from ORBIT analysis
h2_model ='Simple'  #Basic cost model based on H2a and HFTO program record for PEM electrolysis
# h2_model = 'H2A'

# These inputs are not used in this analysis (no solar or storage)
solar_cost_kw = 9999
# storage_cost_kw = 250
# storage_cost_kwh = 240

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

print('Parent path = ', parent_path)

# Site specific turbine information
path = ('examples/H2_Analysis/landbased_costs_ATB.xlsx')
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
        for i, site_deet in enumerate(site_details.iterrows()):
            if i == 0: continue
            else:
                site_deet = site_deet[1]
                print(site_deet)
            lat = site_deet['Lat']
            lon = site_deet['Lon']
            site_name = (str(lat)+","+str(lon))
            location_number = site_deet['site_nums']

            hopp_dict = hoppDict(save_model_input_yaml, save_model_output_yaml)
            sub_dict = {
                'policy': policy[option],
                'atb_year': atb_year,
                'site_location': site_name,
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

            scenario_df = xl.parse()
            scenario_df.set_index(["Parameter"], inplace = True)

            site_df = scenario_df["Site"]
            site_df["Representative coordinates"] = site_name


            turbine_model = str(int(site_df['Turbine Rating']))+'MW'
            turbine_rating = site_df['Turbine Rating']
            print(turbine_model, "Turbine model")

            # set turbine values
            hopp_dict, scenario, nTurbs, floris_config = hopp_tools_steel.set_turbine_model(hopp_dict, turbine_model, scenario, parent_path,floris_dir, floris)

            scenario['Useful Life'] = useful_life

            # financials
            hopp_dict, scenario = hopp_tools_steel.set_financial_info(hopp_dict, scenario, debt_equity_split, discount_rate)

            # set electrolyzer information
            electrolysis_scale = 'Centralized'
            electrolyzer_replacement_scenario = 'Conservative'
            hopp_dict, electrolyzer_capex_kw,capex_ratio_dist, electrolyzer_energy_kWh_per_kg, time_between_replacement =  hopp_tools_steel.set_electrolyzer_info(hopp_dict,atb_year,electrolysis_scale,electrolyzer_replacement_scenario,turbine_rating,direct_coupling = False)
            # Extract Scenario Information from ORBIT Runs
            # Load Excel file of scenarios
            # OSW sites and cost file including turbines 8/16/2022

            # site info
            hopp_dict, site_df, sample_site = hopp_tools_steel.set_site_info(hopp_dict, site_df, sample_site)

            #fixed_or_floating_wind = site_df['Substructure technology']
            site = SiteInfo(sample_site, hub_height=scenario['Tower Height'])

            hopp_dict.add('Configuration', {'site': site})

            #Assign scenario cost details
            if atb_year == 2025:
                total_capex = site_df['2025 Wind CapEx ($/kW)']
                wind_om_cost_kw = site_df['2025 Wind OpEx ($/kW-yr)']
                storage_cost_kw = site_df['2025 Storage ($/kW)']
                storage_cost_kwh = site_df['2025 Storage ($/kWh)']
            if atb_year == 2030:
                total_capex = site_df['2030 Wind CapEx ($/kW)']
                wind_om_cost_kw = site_df['2030 Wind OpEx ($/kW-yr)']
                storage_cost_kw = site_df['2030 Storage ($/kW)']
                storage_cost_kwh = site_df['2030 Storage ($/kWh)']
            if atb_year == 2035:
                total_capex = site_df['2035 Wind CapEx ($/kW)']
                wind_om_cost_kw = site_df['2035 Wind OpEx ($/kW-yr)']
                storage_cost_kw = site_df['2035 Storage ($/kW)']
                storage_cost_kwh = site_df['2035 Storage ($/kWh)']


            capex_multiplier = site_df['CapEx Multiplier']
            wind_cost_kw = copy.deepcopy(total_capex) * capex_multiplier


            #Plot Wind Data to ensure offshore data is sound
            wind_data = site.wind_resource._data['data']
            wind_speed = [W[2] for W in wind_data]
            plot_results.plot_wind_results(wind_data, location_number, site_df['Representative coordinates'], results_dir, plot_wind)


            # Run HOPP

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
            print('Wind plant size: ',hybrid_plant.wind.system_capacity_kw)

            #Step 4: Plot HOPP Results
            plot_results.plot_HOPP(combined_pv_wind_power_production_hopp,
                                    energy_shortfall_hopp,
                                    combined_pv_wind_curtailment_hopp,
                                    load,
                                    results_dir,
                                    location_number,
                                    atb_year,
                                    turbine_model,
                                    hybrid_plant,
                                    plot_power_production)

            #Step 5: Run Simple Dispatch Model
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
                location_number,atb_year,turbine_model,
                load,
                plot_battery,
            )

            # grid information
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
                                        location_number,atb_year,turbine_model,
                                        load,
                                        plot_h2)

            #Step 6b: Run desal model
            hopp_dict, desal_capex, desal_opex, desal_annuals = hopp_tools_steel.desal_model(
                hopp_dict,
                H2_Results,
                electrolyzer_size_mw,
                electrical_generation_timeseries,
                useful_life,
            )

            revised_renewable_cost = hybrid_plant.grid.total_installed_cost
            hydrogen_annual_production = H2_Results['hydrogen_annual_output']
            hydrogen_storage_capacity_kg = 0
            hydrogen_storage_cost_USDprkg = 0
            water_cost = 0.006868 #($/gal) average of green steel sites' water cost

            h2_ptc = scenario['H2 PTC']
            wind_ptc = scenario['Wind PTC']
            h2a_solution,h2a_summary,lcoh_breakdown,electrolyzer_installed_cost_kw = run_profast_for_hydrogen.run_profast_for_hydrogen(site_name,electrolyzer_size_mw,H2_Results,\
                            electrolyzer_capex_kw,time_between_replacement,hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg,\
                            desal_capex,desal_opex,useful_life,water_cost,wind_size_mw,solar_size_mw, \
                            hybrid_plant,revised_renewable_cost,wind_om_cost_kw,grid_connected_hopp,h2_ptc,wind_ptc)
            lcoh = h2a_solution['price']
            print('LCOH: ', lcoh)
            # # Max hydrogen production rate [kg/hr]
            max_hydrogen_production_rate_kg_hr = np.max(H2_Results['hydrogen_hourly_production'])
            max_hydrogen_delivery_rate_kg_hr  = np.mean(H2_Results['hydrogen_hourly_production'])

            electrolyzer_capacity_factor = H2_Results['cap_factor']

            print_results = False
            print_h2_results = False
            test = dict()

            test['Site Name'] = site_name
            test['Lat'] = lat
            test['Lon'] = lon
            test['ATB Year'] = atb_year
            test['Plant life'] = useful_life
            test['Policy'] = option
            test['Turbine size (MW)'] = turbine_rating
            test['Wind Plant size (MW)'] = wind_size_mw
            test['Wind Plant Size Adjusted for Turbine Rating(MW)'] = wind_plant_size /1000
            test['Electrolyzer size (MW)'] = electrolyzer_size_mw
            test['Load Profile (kW)'] = kw_continuous
            test['Energy to Electrolyzer (kW)'] = np.sum(energy_to_electrolyzer)
            test['Wind capacity factor (%)'] = hybrid_plant.wind.capacity_factor
            test['Electrolyzer capacity factor (%)'] = electrolyzer_capacity_factor
            test['LCOH ($/kg)'] = lcoh
            test['LCOH: Compression & storage ($/kg)'] = lcoh_breakdown['LCOH: Compression & storage ($/kg)']
            test['LCOH: Electrolyzer CAPEX ($/kg)']= lcoh_breakdown['LCOH: Electrolyzer CAPEX ($/kg)']
            test['LCOH: Desalination CAPEX ($/kg)'] = lcoh_breakdown['LCOH: Desalination CAPEX ($/kg)']
            test['LCOH: Electrolyzer FOM ($/kg)'] = lcoh_breakdown['LCOH: Electrolyzer FOM ($/kg)']
            test['LCOH: Electrolyzer VOM ($/kg)']=lcoh_breakdown['LCOH: Electrolyzer VOM ($/kg)']
            test['LCOH: Desalination FOM ($/kg)'] = lcoh_breakdown['LCOH: Desalination FOM ($/kg)']
            test['LCOH: Renewable plant ($/kg)'] = lcoh_breakdown['LCOH: Renewable plant ($/kg)']
            test['LCOH: Renewable FOM ($/kg)']= lcoh_breakdown['LCOH: Renewable FOM ($/kg)']
            test['LCOH: Taxes ($/kg)']=lcoh_breakdown['LCOH: Taxes ($/kg)']
            test['LCOH: Water consumption ($/kg)'] = lcoh_breakdown['LCOH: Water consumption ($/kg)']
            test['LCOH: Finances ($/kg)'] = lcoh_breakdown['LCOH: Finances ($/kg)']
            test['LCOH: total ($/kg)'] = lcoh_breakdown['LCOH: total ($/kg)']



            test = pd.DataFrame(test,index=[0])
            print(test)
            results = pd.concat([results,test])


results.to_csv(os.path.join(results_dir, "H2_Analysis_Landbased.csv"))
print('Done')