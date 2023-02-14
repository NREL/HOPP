import os
import sys
sys.path.append('')
from dotenv import load_dotenv
import pandas as pd
import json
from hybrid.sites import SiteInfo
from hybrid.keys import set_developer_nrel_gov_key
# from plot_reopt_results import plot_reopt_results
# from run_reopt import run_reopt
from examples.H2_Analysis.hopp_for_h2 import hopp_for_h2
from examples.H2_Analysis.run_h2a import run_h2a as run_h2a
from examples.H2_Analysis.simple_dispatch import SimpleDispatch
from examples.H2_Analysis.simple_cash_annuals import simple_cash_annuals
import examples.H2_Analysis.run_h2_PEM as run_h2_PEM
import numpy as np
import numpy_financial as npf
from lcoe.lcoe import lcoe as lcoe_calc
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import time
warnings.filterwarnings("ignore")

import hopp_tools
# import hopp_tools_steel
import hopp_tools_steel_ESG as hopp_tools_steel
import inputs_py
import copy 
import plot_results
from hopp_tools_steel import hoppDict
import yaml
import run_RODeO
# import run_pyfast_for_hydrogen
import run_pyfast_for_hydrogen_esg as run_pyfast_for_hydrogen
import run_pyfast_for_steel
import distributed_pipe_cost_analysis

def batch_generator_kernel(arg_list):

    # Read in arguments
    [policy, i, atb_year, site_location, electrolysis_scale,run_RODeO_selector,floris,\
     grid_connection_scenario,grid_price_scenario,electrolyzer_replacement_scenario,\
         parent_path,results_dir,fin_sum_dir,rodeo_output_dir,floris_dir,path,\
     save_hybrid_plant_yaml,save_model_input_yaml,save_model_output_yaml] = arg_list
    
    
    from hybrid.sites import flatirons_site as sample_site # For some reason we have to pull this inside the definition
    
    # # Uncomment and adjust these values if you want to run this script on its own (not as a function)
    # i = 'option 1'
    # policy = {'option 1': {'Wind ITC': 0, 'Wind PTC': 0, "H2 PTC": 0}}
    # atb_year = 2020
    # site_location = 'Site 2'
    # electrolysis_scale = 'Centralized'
    # run_RODeO_selector = True
    # floris = False
    # grid_connection_scenario = 'off-grid'
    # grid_price_scenario = 'retail_peak'
    # electrolyzer_replacement_scenario = 'Standard'
    # # Set paths for results, floris and orbit
    # parent_path = os.path.abspath('')
    # results_dir = parent_path + '/examples/H2_Analysis/results/'
    # floris_dir = parent_path + '/floris_input_files/'
    # path = ('examples/H2_Analysis/green_steel_site_renewable_costs_ATB.xlsx')
    # rodeo_output_dir = 'examples\\H2_Analysis\\RODeO_files\\Output_test\\'
    # fin_sum_dir = parent_path + '/examples/H2_Analysis/financial_summary_results/'
    # save_hybrid_plant_yaml = True # hybrid_plant requires special processing of the SAM objects
    # save_model_input_yaml = True # saves the inputs for each model/major function
    # save_model_output_yaml = True # saves the outputs for each model/major function

    """
    Perform a LCOH analysis for an offshore wind + Hydrogen PEM system
    
    Missing Functionality:
    1. Figure out H2A Costs or ammortize cost/kw electrolyzer figure and add opex
    
    ~1. Offshore wind site locations and cost details (4 sites, $1300/kw capex + BOS cost which will come from Orbit Runs)~
    
    2. Cost Scaling Based on Year (Have Weiser et. al report with cost scaling for fixed and floating tech, will implement)
    3. Cost Scaling Based on Plant Size (Shields et. Al report)
    4. Integration Required:
    * Pressure Vessel Model~
    * HVDC Model 
    * Pipeline Model
    
    5. Model Development Required:
    - Floating Electrolyzer Platform
    """
    
    #Set API key
    load_dotenv()
    NREL_API_KEY = os.getenv("NREL_API_KEY")
    set_developer_nrel_gov_key('NREL_API_KEY')  # Set this key manually here if you are not setting it using the .env
    saveme_elenya=False
    save_elenya_cost = False
    #Step 1: User Inputs for scenario
    resource_year = 2013
    
    sample_site['year'] = resource_year
    useful_life = 30
    critical_load_factor = 1
    run_reopt_flag = False
    custom_powercurve = True    #A flag that is applicable when using PySam WindPower (not FLORIS)
    storage_used = False
    battery_can_grid_charge = False
    grid_connected_hopp = False
    if grid_connection_scenario=='off-grid':
        grid_connected_bool=False
    else:
        grid_connected_bool=True
    # grid_connected_rodeo = False
    # run_RODeO_selector = False
    
    # Technology sizing
    interconnection_size_mw = 1000
    electrolyzer_size_mw = 1000
    wind_size_mw = 1000
    # solar_size_mw = 400 #ESG
    # storage_size_mw = 100 #ESG
    # storage_size_mwh = 400 #ESG
    
    solar_sizes=[0,200]#[100,250,500,750] #[0,0,100,250,500,750]
    storage_sizes_mw=[0,50]#,100,100,100,200]#[0,100,100]#[0,100,100,100,200]
    storage_sizes_mwh=[0,300]#,100,100,400,400]#[0,100,400] #[0,100,100,400,400]
    scenario_choice = 'Green Steel Ammonia Analysis'
    
    scenario = dict()
    kw_continuous = electrolyzer_size_mw * 1000
    load = [kw_continuous for x in
            range(0, 8760)]  # * (sin(x) + pi) Set desired/required load profile for plant
    
    #Site lat and lon will be set by data loaded from Orbit runs
    
    # Financial inputs
    discount_rate = 0.07
    debt_equity_split = 60
    
    # Wind costs input from ORBIT analysis
    h2_model ='Simple'  #Basic cost model based on H2a and HFTO program record for PEM electrolysis
    # h2_model = 'H2A'
    
    # These inputs are not used in this analysis (no solar or storage)
    st_xl=pd.read_csv('/Users/egrant/Desktop/HOPP-GIT/ESG_CostRelatedDocs/storage_costs_ATB.csv',index_col=0)
    storage_costs=st_xl[str(atb_year)]
    
    #storage_capex_per_kW=storage_costs['Battery Energy Capital Cost ($/kWh)']*(storage_size_mwh/storage_size_mw) + storage_costs['Battery Power Capital Cost ($/kW)'] #$/kW
    #battery_fom=storage_costs['Fixed O&M Perc']*storage_capex_per_kW #$/kW-year
    ref_size_battery=1000*storage_costs['Reference Charge Rate (MW)']
    ref_cap_battery=1000*storage_costs['Reference Capacity (MWh)'] #4 hour
    battery_lr=0.21
    battery_b=np.log2(1-battery_lr)
    
    #I hope the solar costs are over-written :)
    solar_cost_kw = 640 #ESG
    solar_om_cost_kw=15
    # storage_cost_kw = 250 #Jenn's code is 1500
    # storage_cost_kwh = 240 #Jenn's code is 380

    storage_cost_kwh=storage_costs['Battery Energy Capital Cost ($/kWh)'] #ESG
    storage_cost_kw=storage_costs['Battery Power Capital Cost ($/kW)'] #ESG
    # Flags (TODO: remove them and update documentation)
    forced_sizes = True
    force_electrolyzer_cost = False
    
    
    # Enable Ability to purchase/sell electricity to/from grid. Price Defined in $/kWh
    # sell_price = 0.01
    # buy_price = 0.01
    sell_price = False
    buy_price = False
        
    print('Parent path = ', parent_path)
    
    # Site specific turbine information
    xl = pd.ExcelFile(path)
    for solar_size_mw in solar_sizes:
        for stosto in range(len(storage_sizes_mw)):
            storage_size_mw=storage_sizes_mw[stosto]
            storage_size_mwh=storage_sizes_mwh[stosto]
            save_outputs_dict = inputs_py.establish_save_output_dict()
            save_all_runs = list()
            
            # which plots to show
            plot_power_production = False
            plot_battery = False
            plot_grid = False
            plot_h2 = False
            plot_desal = True
            plot_wind = False
            plot_hvdcpipe = True
            plot_hvdcpipe_lcoh = True
            

            
            # Read in gams exe and license location
            # Create a .txt file in notepad with the locations of the gams .exe file, the .gms RODeO
            # version that you want to use, and the location of the gams license file. The text
            # should look something like this: 
            # "C:\\GAMS\\win64\\24.8\\gams.exe" ..\\RODeO\\Storage_dispatch_SCS license=C:\\GAMS\\win64\\24.8\\gamslice.txt
            # Do not push this file to the remote repository because it will be different for every user
            # and for every machine, depending on what version of gams they are using and where it is installed
            if run_RODeO_selector == True:
                with open('gams_exe_license_locations.txt') as f:
                    gams_locations_rodeo_version = f.readlines()
                f.close()
            
            hopp_dict = hoppDict(save_model_input_yaml, save_model_output_yaml)
            
            sub_dict = {
                'policy': policy[i],
                'atb_year': atb_year,
                'site_location': site_location,
                'parent_path': parent_path,
                # 'load': load,
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
                'storage_cost_kw': storage_cost_kw,
                'storage_cost_kwh': storage_cost_kwh,
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
            hopp_dict, scenario, policy_option = hopp_tools_steel.set_policy_values(hopp_dict, scenario, policy, i)
            print(scenario['Wind PTC'])

            scenario_df = xl.parse()
            scenario_df.set_index(["Parameter"], inplace = True)
            
            site_df = scenario_df[site_location]

            turbine_model = str(site_df['Turbine Rating'])+'MW'

            # set turbine values
            hopp_dict, scenario, nTurbs, floris_config = hopp_tools_steel.set_turbine_model(hopp_dict, turbine_model, scenario, parent_path,floris_dir, floris)
            
            scenario['Useful Life'] = useful_life

            # financials
            hopp_dict, scenario = hopp_tools_steel.set_financial_info(hopp_dict, scenario, debt_equity_split, discount_rate)

            # set electrolyzer information
            hopp_dict, electrolyzer_capex_kw, electrolyzer_energy_kWh_per_kg, time_between_replacement =  hopp_tools_steel.set_electrolyzer_info(hopp_dict, atb_year,electrolysis_scale,electrolyzer_replacement_scenario)

            # Extract Scenario Information from ORBIT Runs
            # Load Excel file of scenarios
            # OSW sites and cost file including turbines 8/16/2022 

            # site info
            hopp_dict, site_df, sample_site = hopp_tools_steel.set_site_info(hopp_dict, site_df, sample_site)
            site_name = site_df['State']
            #fixed_or_floating_wind = site_df['Substructure technology']
            site = SiteInfo(sample_site, hub_height=scenario['Tower Height'])
            
            hopp_dict.add('Configuration', {'site': site})

            #Assign scenario cost details
            if atb_year == 2020:
                total_capex = site_df['2020 CapEx']
                wind_om_cost_kw = site_df['2020 OpEx ($/kw-yr)']
            if atb_year == 2025:
                total_capex = site_df['2025 CapEx']
                wind_om_cost_kw = site_df['2025 OpEx ($/kw-yr)']
            if atb_year == 2030:
                total_capex = site_df['2030 CapEx']
                wind_om_cost_kw = site_df['2030 OpEx ($/kw-yr)']
            if atb_year == 2035:
                total_capex = site_df['2035 CapEx']
                wind_om_cost_kw = site_df['2035 OpEx ($/kw-yr)']
            solar_om_cost_kw=site_df[str(atb_year) + ' PV OpEx'] #ESG
            solar_capex_multiplier=site_df['PV Capex Multiplier'] #ESG
            solar_capex=site_df[str(atb_year) + ' PV base installed cost'] #ESG
            solar_cost_kw=solar_capex * solar_capex_multiplier #ESG

            capex_multiplier = site_df['CapEx Multiplier']
            wind_cost_kw = copy.deepcopy(total_capex) * capex_multiplier

            # # set export financials
            # wind_cost_kw, wind_om_cost_kw, total_export_system_cost, total_export_om_cost = hopp_tools.set_export_financials(wind_size_mw, 
            #                                                                                                                 wind_cost_kw,
            #                                                                                                                 wind_om_cost_kw,
            #                                                                                                                 useful_life,
            #                                                                                                                 site_df)
            # # set wind financials
            # new_wind_cost_kw, new_wind_om_cost_kw, new_wind_net_cf = hopp_tools.set_turbine_financials(turbine_model, 
            #                                                                                             fixed_or_floating_wind,
            #                                                                                             atb_year,
            #                                                                                             wind_cost_kw,
            #                                                                                             wind_om_cost_kw,
            #                                                                                             wind_net_cf,
            #                                                                                             parent_path)
            #Plot Wind Data to ensure offshore data is sound
            wind_data = site.wind_resource._data['data']
            wind_speed = [W[2] for W in wind_data]
            plot_results.plot_wind_results(wind_data, site_name, site_df['Representative coordinates'], results_dir, plot_wind)

            #Plot Wind Cost Contributions
            # Plot a nested pie chart of results
            # TODO: Remove export system from pieplot
            # plot_results.plot_pie(site_df, site_name, turbine_model, results_dir)
            
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
                            floris, solar_om_cost_kw,
                        )
                
            generation_summary_df = pd.DataFrame({'Generation profile (kW)': hybrid_plant.grid.generation_profile[0:8760] })
            #generation_summary_df.to_csv(os.path.join(results_dir, 'Generation Summary_{}_{}_{}_{}.csv'.format(site_name,atb_year,turbine_model,scenario['Powercurve File'])))


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
                site_name,atb_year,turbine_model,
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
            
            # Step #: Calculate hydrogen pipe costs for distributed case
            if electrolysis_scale == 'Distributed':
                # High level estimate of max hydrogen flow rate. Doesn't have to be perfect, but should be slightly conservative (higher efficiency)
                hydrogen_max_hourly_production_kg = max(energy_to_electrolyzer)/electrolyzer_energy_kWh_per_kg 
                
                # Run pipe cost analysis module
                pipe_network_cost_total_USD,pipe_network_costs_USD,pipe_material_cost_bymass_USD =\
                    distributed_pipe_cost_analysis.hydrogen_steel_pipeline_cost_analysis(parent_path,turbine_model,hydrogen_max_hourly_production_kg,site_name)
                
            
            # Step 6: Run RODeO or Pyfast for hydrogen
            
            if run_RODeO_selector == True:
                rodeo_scenario,lcoh,electrolyzer_capacity_factor,hydrogen_storage_duration_hr,hydrogen_storage_capacity_kg,\
                    hydrogen_annual_production,water_consumption_hourly,RODeO_summary_results_dict,hydrogen_hourly_results_RODeO,\
                        electrical_generation_timeseries,electrolyzer_installed_cost_kw,hydrogen_storage_cost_USDprkg\
                    = run_RODeO.run_RODeO(atb_year,site_name,turbine_model,wind_size_mw,solar_size_mw,electrolyzer_size_mw,\
                            energy_to_electrolyzer,electrolyzer_energy_kWh_per_kg,hybrid_plant,electrolyzer_capex_kw,wind_om_cost_kw,useful_life,time_between_replacement,\
                            grid_connection_scenario,grid_price_scenario,gams_locations_rodeo_version,rodeo_output_dir)
                    
            else:
            # If not running RODeO, run H2A via PyFAST
                # Currently only works for offgrid
                #grid_string = 'offgrid'    
                #scenario_name = 'steel_'+str(atb_year)+'_'+ site_location.replace(' ','-') +'_'+turbine_model+'_'+grid_string
                esgsaveme_str=site_df['State'] + '_{}_Wind{}_Solar{}_Storage{}MWH_Storage{}MW'.format(atb_year,wind_size_mw,solar_size_mw,storage_size_mwh,storage_size_mw)
                #Run the H2_PEM model to get hourly hydrogen output, capacity factor, water consumption, etc.
                h2_model = 'Simple'
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
                    lcoe, esgsaveme_str #ESG
                )
                
                #Step 6b: Run desal model
                hopp_dict, desal_capex, desal_opex, desal_annuals = hopp_tools_steel.desal_model(
                    hopp_dict,
                    H2_Results, 
                    electrolyzer_size_mw, 
                    electrical_generation_timeseries, 
                    useful_life,
                )
                
                hydrogen_annual_production = H2_Results['hydrogen_annual_output']
                # time_between_replacement=H2_Results['time_until_replacement'] #ESG
                # hydrogen_max_hourly_production_kg = max(H2_Results['hydrogen_hourly_production'])

                # Calculate required storage capacity to meet a flat demand profile. In the future, we could customize this to
                # work with any demand profile
                
                # Storage costs as a function of location
                if site_location == 'Site 1':
                    storage_type = 'Buried pipes'
                elif site_location == 'Site 2':
                    storage_type = 'Salt cavern'
                elif site_location == 'Site 3':
                    storage_type = 'Buried pipes'
                elif site_location == 'Site 4':
                    storage_type = 'Salt cavern'
                elif site_location == 'Site 5':
                    storage_type = 'Buried pipes' #Unsure
                
                hydrogen_production_storage_system_output_kgprhr,hydrogen_storage_capacity_kg,hydrogen_storage_capacity_MWh_HHV,hydrogen_storage_duration_hr,hydrogen_storage_cost_USDprkg,storage_status_message\
                    = hopp_tools.hydrogen_storage_capacity_cost_calcs(H2_Results,electrolyzer_size_mw,storage_type)   
                print(storage_status_message)
                
                # Run PyFAST to get LCOH
                
                # Municipal water rates and wastewater treatment rates combined ($/gal)
                if site_location == 'Site 1': # Site 1 - Indiana
                    water_cost = 0.00612
                elif site_location == 'Site 2': # Site 2 - Texas
                    water_cost = 0.00811
                elif site_location == 'Site 3': # Site 3 - Iowa
                    water_cost = 0.00634
                elif site_location == 'Site 4': # Site 4 - Mississippi
                    water_cost = 0.00844
                elif site_location =='Site 5': # Site 5 - Wyoming  
                    water_cost=0.00533 #Commercial water cost for Cheyenne https://www.cheyennebopu.org/Residential/Billing-Rates/Water-Sewer-Rates
            
            
                h2a_solution,h2a_summary,lcoh_breakdown,electrolyzer_installed_cost_kw = run_pyfast_for_hydrogen. run_pyfast_for_hydrogen(site_location,electrolyzer_size_mw,H2_Results,\
                                                electrolyzer_capex_kw,time_between_replacement,hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg,\
                                                desal_capex,desal_opex,useful_life,water_cost,wind_size_mw,solar_size_mw,hybrid_plant,wind_om_cost_kw,grid_connected_hopp)
                
                lcoh = h2a_solution['price']

                if save_elenya_cost:
                    cost_dict = {'site_location':site_location,'electrolyzer_size_mw':electrolyzer_size_mw,'electrolyzer_capex_kw':electrolyzer_capex_kw,'time_between_replacement':time_between_replacement,\
                        'hydrogen_storage_capacity_kg':hydrogen_storage_capacity_kg,'hydrogen_storage_cost_USDprkg':hydrogen_storage_cost_USDprkg,'desal_capex':desal_capex,'desal_opex':desal_opex,\
                            'useful_life':useful_life,'water_cost':water_cost,'wind_size_mw':wind_size_mw,'solar_size_mw':solar_size_mw,'hybrid_plant':hybrid_plant.grid.total_installed_cost,'wind_om_cost_kw':wind_om_cost_kw}
                    cost_series = pd.Series(cost_dict)
                    cost_series.to_csv('/Users/egrant/Desktop/HOPP-GIT/CapFac_Tests/TestBasicPEMControl/CostInfo/' + site_df['State'] + '_Wind{}_Solar{}_Storage{}MWH'.format(wind_size_mw,solar_size_mw,storage_size_mwh))
                    #cost_dict = {'site_location':site_location,'electrolyzer_size_mw':electrolyzer_size_mw,'electrolyzer_capex_kw':electrolyzer_capex_kw,''time_between_replacement}


                

                
                
            # Step 7: Calculate break-even cost of steel and ammonia production
            lime_unit_cost = site_df['Lime ($/metric tonne)'] + site_df['Lime Transport ($/metric tonne)']
            carbon_unit_cost = site_df['Carbon ($/metric tonne)'] + site_df['Carbon Transport ($/metric tonne)']
            iron_ore_pellets_unit_cost = site_df['Iron Ore Pellets ($/metric tonne)'] + site_df['Iron Ore Pellets Transport ($/metric tonne)']
            hopp_dict,steel_economics_from_pyfast, steel_economics_summary, steel_breakeven_price, steel_annual_production_mtpy,steel_price_breakdown = hopp_tools_steel.steel_LCOS(hopp_dict,lcoh,hydrogen_annual_production,
                                                                                                                    lime_unit_cost,
                                                                                                                    carbon_unit_cost,
                                                                                                                    iron_ore_pellets_unit_cost)
            
            cooling_water_cost = 0.000113349938601175 # $/Gal
            iron_based_catalyst_cost = 23.19977341 # $/kg
            oxygen_cost = 0.0285210891617726       # $/kg 
            hopp_dict,ammonia_economics_from_pyfast, ammonia_economics_summary, ammonia_breakeven_price, ammonia_annual_production_kgpy,ammonia_price_breakdown = hopp_tools_steel.levelized_cost_of_ammonia(hopp_dict,lcoh,hydrogen_annual_production,
                                                                                                                    cooling_water_cost,
                                                                                                                    iron_based_catalyst_cost,
                                                                                                                    oxygen_cost)
                    
            # Step 7: Write outputs to file
            
            total_h2export_system_cost=0
            opex_pipeline=0
            total_export_system_cost=0
            total_export_om_cost=0
            
            if run_RODeO_selector == True:             
                policy_option,turbine_model,scenario['Useful Life'], wind_cost_kw, solar_cost_kw,\
                scenario['Debt Equity'], atb_year, scenario['H2 PTC'],scenario['Wind ITC'],\
                discount_rate, tlcc_wind_costs, tlcc_solar_costs, tlcc_hvdc_costs, tlcc_total_costs,run_RODeO_selector,lcoh,\
                wind_itc_total, total_itc_hvdc = hopp_tools.write_outputs_RODeO(electrical_generation_timeseries,\
                                    hybrid_plant,
                                    total_export_system_cost,
                                    total_export_om_cost,
                                    cost_to_buy_from_grid,
                                    electrolyzer_capex_kw, 
                                    electrolyzer_installed_cost_kw,
                                    hydrogen_storage_cost_USDprkg,
                                    time_between_replacement,
                                    profit_from_selling_to_grid,
                                    useful_life,
                                    atb_year,
                                    policy_option,
                                    scenario,
                                    wind_cost_kw,
                                    solar_cost_kw,
                                    discount_rate,
                                    solar_size_mw,
                                    results_dir,
                                    fin_sum_dir,
                                    site_name,
                                    turbine_model,
                                    electrolysis_scale,
                                    scenario_choice,
                                    lcoe,
                                    run_RODeO_selector,
                                    grid_connection_scenario,
                                    grid_price_scenario,
                                    lcoh,
                                    electrolyzer_capacity_factor,
                                    hydrogen_storage_duration_hr,
                                    hydrogen_storage_capacity_kg,
                                    hydrogen_annual_production,
                                    water_consumption_hourly,
                                    RODeO_summary_results_dict,
                                    steel_annual_production_mtpy,
                                    steel_breakeven_price,
                                    steel_price_breakdown,
                                    ammonia_annual_production_kgpy,
                                    ammonia_breakeven_price,
                                    ammonia_price_breakdown) 
            else:
                policy_option,turbine_model,scenario['Useful Life'], wind_cost_kw, solar_cost_kw,\
                scenario['Debt Equity'], atb_year, scenario['H2 PTC'],scenario['Wind ITC'],\
                discount_rate, tlcc_wind_costs, tlcc_solar_costs, tlcc_hvdc_costs, tlcc_total_costs,run_RODeO_selector,lcoh,\
                wind_itc_total, total_itc_hvdc = hopp_tools.write_outputs_PyFAST(electrical_generation_timeseries,\
                                    hybrid_plant,
                                    total_export_system_cost,
                                    total_export_om_cost,
                                    cost_to_buy_from_grid,
                                    electrolyzer_capex_kw,
                                    electrolyzer_installed_cost_kw,
                                    hydrogen_storage_cost_USDprkg,
                                    time_between_replacement,
                                    profit_from_selling_to_grid,
                                    useful_life,
                                    atb_year,
                                    policy_option,
                                    scenario,
                                    wind_cost_kw,
                                    solar_cost_kw,
                                    discount_rate,
                                    solar_size_mw,
                                    results_dir,
                                    fin_sum_dir,
                                    site_name,
                                    turbine_model,
                                    electrolysis_scale,
                                    scenario_choice,
                                    lcoe,
                                    run_RODeO_selector,
                                    grid_connection_scenario,
                                    grid_price_scenario,
                                    lcoh,
                                    H2_Results,
                                    hydrogen_storage_duration_hr,
                                    hydrogen_storage_capacity_kg,
                                    lcoh_breakdown,
                                    steel_annual_production_mtpy,
                                    steel_breakeven_price,
                                    steel_price_breakdown,
                                    ammonia_annual_production_kgpy,
                                    ammonia_breakeven_price,
                                    ammonia_price_breakdown) 
            #savemename=results_dir + 'Solar_Sensitivity222/' + site_df['State'] + '_{}_Wind{}_Solar{}_Storage{}MWH_Storage{}MW'.format(atb_year,wind_size_mw,solar_size_mw,storage_size_mwh,storage_size_mw)
            []
            if saveme_elenya:
                capfac_dir='/Users/egrant/Desktop/HOPP-GIT/CapFac_Tests/'
                #cf_subdir='FAKKEKEUpdatedCosts_' +electrolysis_scale +'_' + grid_connection_scenario +'_'+ grid_price_scenario +'_Policy'+ list(policy.keys())[0].replace('option ','-') + '/'
                # cf_subdir = 'NEWPEM_BasicControl/'
                cf_subdir = 'NEWPEM_BasicControl-UpdatedReplacementTime/'
                if not os.path.isdir(capfac_dir + cf_subdir):
                    os.mkdir(capfac_dir + cf_subdir)
                    savemename=capfac_dir + cf_subdir + site_df['State'] + '_{}_Wind{}_Solar{}_Storage{}MWH_Storage{}MW'.format(atb_year,wind_size_mw,solar_size_mw,storage_size_mwh,storage_size_mw)
                else:
                    savemename=capfac_dir + cf_subdir + site_df['State'] + '_{}_Wind{}_Solar{}_Storage{}MWH_Storage{}MW'.format(atb_year,wind_size_mw,solar_size_mw,storage_size_mwh,storage_size_mw)

                # df_timeseries= pd.DataFrame([combined_pv_wind_curtailment_hopp, 
                #                                                 energy_shortfall_hopp, 
                #                                                 combined_pv_wind_storage_power_production_hopp, 
                #                                                 combined_pv_wind_power_production_hopp, load,
                #                                                 battery_SOC, battery_used,H2_Results['hydrogen_hourly_production'],energy_to_electrolyzer,H2_Results['electrolyzer_total_efficiency']],
                #                                                 ['Curtailment', 'Shortfall', 'Wind + PV + Storage', 
                #                                                 'Wind + PV Generation', 'Load', 'State of Charge', 'Battery Used','H2 Production','Energy to Electrolyzer','Electrolyzer Efficiency'])
                # df_timeseries.T.to_csv(savemename + '_TimeSeries.csv')
                df_annual=pd.DataFrame([lcoe,lcoh,hydrogen_annual_production,hydrogen_production_storage_system_output_kgprhr,steel_annual_production_mtpy,grid_connection_scenario,electrolysis_scale],\
                    ['LCOE','LCOH','Annual H2 Prod','H2 Storage System Output','Steel Annual Production MtPY','Grid Scenario','Electrolysis Scale'])
                df_annual.T.to_csv(savemename + '_AnnualData.csv')
            print('WOOHOO')
            #[lcoe, lcoh, hybrid_plant.lcoe_nom['hybrid'],hybrid_plant.power_sources['wind'].capacity_factor]
                # plot_results.donut(steel_price_breakdown,results_dir, 
                #                     site_name, atb_year, policy_option)

                        
        
                            

                        


                        

                #         #Step 6: Run the H2_PEM model
                #         h2_model = 'Simple'
                #         H2_Results, H2A_Results, electrical_generation_timeseries = hopp_tools.run_H2_PEM_sim(hybrid_plant,
                #                                                                                                 energy_to_electrolyzer,
                #                                                                                                 scenario,
                #                                                                                                 wind_size_mw,
                #                                                                                                 solar_size_mw,
                #                                                                                                 electrolyzer_size_mw,
                #                                                                                                 kw_continuous,
                #                                                                                                 electrolyzer_capex_kw,
                #                                                                                                 lcoe)

                #         plot_results.plot_h2_results(H2_Results, 
                #                                     electrical_generation_timeseries,
                #                                     results_dir,
                #                                     site_name,atb_year,turbine_model,
                #                                     load,
                #                                     plot_h2)

                #         #Step 6b: Run desal model
                #         desal_capex, desal_opex, desal_annuals = hopp_tools.desal_model(H2_Results, 
                #                                                         electrolyzer_size_mw, 
                #                                                         electrical_generation_timeseries, 
                #                                                         useful_life)

                #         # compressor model
                #         compressor, compressor_results = hopp_tools.compressor_model()

                #         #Pressure Vessel Model Example
                #         storage_input, storage_output = hopp_tools.pressure_vessel()

                #         # pipeline model
                #         total_h2export_system_cost, opex_pipeline, dist_to_port_value = hopp_tools.pipeline(site_df, 
                #                                                                         H2_Results, 
                #                                                                         useful_life, 
                #                                                                         storage_input)
                        
                        
                #         # plot HVDC vs pipe 
                #         plot_results.plot_hvdcpipe(total_export_system_cost,
                #                                     total_h2export_system_cost,
                #                                     site_name,
                #                                     atb_year,
                #                                     dist_to_port_value,
                #                                     results_dir)

                        


