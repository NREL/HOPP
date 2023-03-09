import os
import sys
sys.path.append('')
from dotenv import load_dotenv
import pandas as pd
# import json
from hybrid.sites import SiteInfo
from hybrid.sites import flatirons_site as sample_site
from hybrid.keys import set_developer_nrel_gov_key
# from plot_reopt_results import plot_reopt_results
# from run_reopt import run_reopt
# from examples.H2_Analysis.hopp_for_h2 import hopp_for_h2
# from examples.H2_Analysis.run_h2a import run_h2a as run_h2a
# from examples.H2_Analysis.simple_dispatch import SimpleDispatch
# from examples.H2_Analysis.simple_cash_annuals import simple_cash_annuals
# import examples.H2_Analysis.run_h2_PEM as run_h2_PEM
import numpy as np
# import numpy_financial as npf
# from lcoe.lcoe import lcoe as lcoe_calc
# import matplotlib.pyplot as plt
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
from tools.resource import *
from tools.resource.resource_loader import site_details_creator

import hopp_tools_steel
import copy 
import plot_results
import run_pyfast_for_hydrogen
from hopp_tools_steel import hoppDict
# import yaml

import linecache
import tracemalloc
import gc

#for memory issues, to diagnose

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))
    
tracemalloc.start()


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
# atb_years = [2030, 2035]
atb_years = [2025, 2030, 2035]

policy = {
    'No Policy': {'Wind ITC': 0, 'Wind PTC': 0, "H2 PTC": 0},
    'Base': {'Wind ITC': 0, 'Wind PTC': 0.0051, "H2 PTC": 0.6}, #per kg,  0.015, 0-.45kg
    'Max': {'Wind ITC': 0, 'Wind PTC': 0.0256, "H2 PTC": 3}, #5x for prevailing wage and apprenticeship, but that cost is not reflected
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
# N_lat = 50 #50 #5 number of data points
# N_lon = 95 #95 #5
# desired_lats = np.linspace(23.833504, 49.3556, N_lat)
# desired_lons = np.linspace(-129.22923, -65.7146, N_lon)
desired_lats = [33.162, 35.23328, 39.353056, 36.85, 44.15, 42.06, 45.71]
desired_lons = [-102.87, -83.8, -112.573611, -119.1, -73.82, -74.50, -121.80]
#GA: 33.162, -83.8 Site 8 (4)
#TX: 35.23328, -102.87 Site 2 (2)
#UT: 39.353056, -112.573611 Site 17 (9)
#CA: 36.85, -119.1 Site 24
#NY North: 44.15, -73.82 Site 28
#NY South: 42.06, -74.50 Site 31
#OR: 45.71, -121.80 Site 36

load_resource_from_file = False #True
resource_dir = Path(__file__).parent.parent.parent / "resource_files"
# print('Resource directory: ', resource_dir)
# sitelist_name = 'filtered_site_details_{}_lats_{}_lons_{}_resourceyear_full.csv'.format(N_lat, N_lon, resource_year)
sitelist_name = 'filtered_site_details_TX_GA_UT_V.csv'


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
        print('Found site details file', sitelist_name)
    else:
        site_details = site_details_creator.site_details_creator(desired_lats, desired_lons, resource_dir)
        # Filter to locations in USA
        site_details = filter_sites(site_details, location='usa only')
        print('Site details: ', site_details)
        site_details.to_csv(sitelist_name)
# print("Site Details Created")

scenario = dict()
kw_continuous = electrolyzer_size_mw * 1000
load = [kw_continuous for x in
        range(0, 8760)]  # * (sin(x) + pi) Set desired/required load profile for plant

#Site lat and lon will be set by data loaded from Orbit runs

# Financial inputs
discount_rate = 0.0824
debt_equity_split = 60 #cec changed from 0 to 60 on 2/1/23

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
sell_price = False #0.01
buy_price = False #0.01

# Set paths for results, floris and orbit
parent_path = os.path.abspath('')
results_dir = parent_path + '/examples/H2_Analysis/results_LBW_Feb2023/sites_V'
results = pd.DataFrame()
floris_dir = parent_path + '/floris_input_files/'

# Site specific turbine information
path = ('examples/H2_Analysis/landbased_costs_ATB.xlsx')
xl = pd.ExcelFile(path)

# Plots
plot_power_production = True
plot_battery = False
plot_grid = False
plot_h2 = True
plot_desal = False
plot_wind = False
plot_hvdcpipe = False
plot_hvdcpipe_lcoh = False

# atb_year = 2025 #memory issues, trying one year at a time


for atb_year in atb_years:
    for option in policy:
        print('Policy: ', option, 'ATB Year: ', atb_year)
        
        #read in results file to make a list of what has already been run. If sims get interrupted, you can just hit start again

        results_file = os.path.join(results_dir, "H2_Analysis_LBW_{}_{}.csv".format(atb_year, option))
        if os.path.exists(results_file):
            print(results_file)
            check_sims = pd.read_csv(results_file, names=['Site Name', 'Lat', 'Lon', 'ATB Year', 'Plant life', 'Policy',\
                         'Turbine size (MW)', 'Wind Plant size (MW)', 'Wind Plant Size Adjusted for Turbine Rating(MW)',\
                         'Electrolyzer size (MW)', 'Load Profile (kW)', 'Energy to Electrolyzer (kW)', 'Wind capacity factor (%)',\
                         'Electrolyzer capacity factor (%)', 'LCOH ($/kg)', 'LCOH: Compression & storage ($/kg)',\
                         'LCOH: Electrolyzer CAPEX ($/kg)', 'LCOH: Desalination CAPEX ($/kg)', 'LCOH: Electrolyzer FOM ($/kg)',\
                         'LCOH: Electrolyzer VOM ($/kg)', 'LCOH: Desalination FOM ($/kg)', 'LCOH: Renewable plant ($/kg)',\
                         'LCOH: Renewable FOM ($/kg)', 'LCOH: Taxes ($/kg)', 'LCOH: Water consumption ($/kg)',\
                         'LCOH: Finances ($/kg)', 'LCOH: total ($/kg)'],dtype= {'Lat': float, 'Lon': float})#, usecols = ['Site Name'])
            print('Checking results in: ', results_file)
            
            check_sims_lat = check_sims['Lat'].values.tolist()
            check_sims_lon = check_sims['Lon'].values.tolist()
            check_site_names = list(zip(check_sims_lat,check_sims_lon))
            # print('check_site_names: ', check_site_names)

        
        for i, site_deet in enumerate(site_details.iterrows()):
            
            if i == 0: continue
            else:
                #print("site_deet[1] is: ", site_deet[1])
                site_deet = site_deet[1]
                
            
            lat = site_deet['Lat'] #'%.7f'%site_deet['Lat']
            lon = site_deet['Lon'] #'%.7f'%site_deet['Lon']
            site_name = (lat,lon)
            print('Site Name: ', site_name)
            # print('Check site name: ', check_site_names)
            
            if os.path.exists(results_file) and site_name in check_site_names:
                # if site_name in check_site_names:
                print('Site name matched check site name', i)
                # print(check_site_names)
                continue


            # print(check_site_names)
            print(i, ' of ', len(site_details))
            
            location_number = site_deet['site_nums']
            
            site_name = lat, lon
    
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
            } #What is this?
    
            hopp_dict.add('Configuration', plot_dict)
    
            # set policy values
            hopp_dict, scenario, policy_option = hopp_tools_steel.set_policy_values(hopp_dict, scenario, policy, option)
            # print(scenario['Wind PTC'])
    
            scenario_df = xl.parse()
            scenario_df.set_index(["Parameter"], inplace = True)
    
            site_df = scenario_df["Site"]
            site_df["Representative coordinates"] = site_name
    
            turbine_model = str(int(site_df['Turbine Rating']))+'MW'
            turbine_rating = site_df['Turbine Rating']
    
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
            #wind resource file is incorrect, so the resource files are being saved, but when they are being checked, they are not in the location that this SiteInfo and WindResource thinks they are in, so it is redownloading every time
            # path_resource='/Users/cclark2/Desktop/HOPP/HOPP/resource_files/wind/
            wind_resource_file='/Users/cclark2/Desktop/HOPP/HOPP/resource_files/wind/{}_{}_windtoolkit_2013_60min_160m_200m.srw'.format(lat, lon)
            # print('Wind  Resource File: ', wind_resource_file)
            site = SiteInfo(sample_site, wind_resource_file=wind_resource_file, hub_height=scenario['Tower Height'])
    
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
            wind_cost_kw = copy.deepcopy(total_capex) * capex_multiplier #TODO:ASK KAITLIN
    
    
            # # Plot Wind Data to ensure offshore data is sound
            # wind_data = site.wind_resource._data['data']
            # wind_speed = [W[2] for W in wind_data]
            # plot_results.plot_wind_results(wind_data, location_number, site_df['Representative coordinates'], results_dir, plot_wind)
    
    
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
            #print('Wind plant size: ',hybrid_plant.wind.system_capacity_kw)
    
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
            h2a_solution,h2a_summary,lcoh_breakdown,electrolyzer_installed_cost_kw = run_pyfast_for_hydrogen.run_pyfast_for_hydrogen(site_name,\
                            electrolyzer_size_mw,H2_Results,electrolyzer_capex_kw,time_between_replacement,hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg,\
                            desal_capex,desal_opex,useful_life,water_cost,wind_size_mw,solar_size_mw, \
                            hybrid_plant,revised_renewable_cost,wind_om_cost_kw,grid_connected_hopp,h2_ptc,wind_ptc,atb_year)
            lcoh = h2a_solution['price']
            #print('LCOH: ', lcoh)
            # # Max hydrogen production rate [kg/hr]
            max_hydrogen_production_rate_kg_hr = np.max(H2_Results['hydrogen_hourly_production'])
            max_hydrogen_delivery_rate_kg_hr  = np.mean(H2_Results['hydrogen_hourly_production'])
            
            electrolyzer_capacity_factor = H2_Results['cap_factor']
    
            print_results = False
            print_h2_results = False
            test = dict() #make empty dict and populate it with lat/long/policy option/year results combo
    
            site_name = str(lat)+','+str(lon)
    
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

            test = pd.DataFrame(test,index=[0]) #convert dict to df
            test.to_csv(results_file, mode='a', index=False, header=False)

            print('Results written', site_name, atb_year, option)
            
            # #Calls memory allocation snapshot definition from beginning of file, displays top 10 mem eaters
            # snapshot = tracemalloc.take_snapshot()
            # display_top(snapshot)             
            # gc.collect() #collects opened plot files to help close them


# results.to_csv(os.path.join(results_dir, "H2_Analysis_Landbased.csv"))
print('Done')
