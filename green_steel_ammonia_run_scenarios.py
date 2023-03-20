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
import hopp_tools_steel
import inputs_py
import copy 
import plot_results
from hopp_tools_steel import hoppDict
import yaml
import run_RODeO
import run_profast_for_hydrogen
import run_profast_for_steel
import distributed_pipe_cost_analysis
import hopp_tools_run_wind_solar
import LCA_single_scenario
import LCA_single_scenario_ProFAST
from hybrid.PEM_Model_2Push import run_PEM_master

def batch_generator_kernel(arg_list):

    # Read in arguments
    [policy, i, atb_year, site_location, electrolysis_scale,run_RODeO_selector,floris,\
     grid_connection_scenario,grid_price_scenario,\
     direct_coupling,steel_annual_production_rate_target_tpy,parent_path,results_dir,fin_sum_dir,rodeo_output_dir,floris_dir,path,\
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
    #steel_annual_production_rate_target_tpy = 1278981.78


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
    
    # Calculate target hydrogen and electricity demand
    hydrogen_consumption_for_steel = 0.06596 # metric tonnes of hydrogen/metric tonne of steel production
    
    # Annual hydrogen production target to meet steel production target
    hydrogen_production_target_kgpy = steel_annual_production_rate_target_tpy*1000*hydrogen_consumption_for_steel
    
    electrolyzer_energy_kWh_per_kg = 55.5 # Eventually need to re-arrange things to get this from set_electrolyzer_info
    
    # Annual electricity target to meet hydrogen production target - use this to calculate renewable plant sizing
    electricity_production_target_MWhpy = hydrogen_production_target_kgpy*electrolyzer_energy_kWh_per_kg/1000


    #Set API key
    load_dotenv()
    NREL_API_KEY = os.getenv("NREL_API_KEY")
    set_developer_nrel_gov_key('NREL_API_KEY')  # Set this key manually here if you are not setting it using the .env
    
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
    # grid_connected_rodeo = False
    #run_RODeO_selector = False
    
    # Technology sizing
    interconnection_size_mw = 1000
    electrolyzer_size_mw = 1000
    wind_size_mw = 1000
    solar_sizes_mw=[0,100,200]
    storage_sizes_mw=[0]#,100,100,200]
    storage_sizes_mwh = [0]#,100,400,400]
    solar_size_mw = 0
    storage_size_mw = 0
    storage_size_mwh = 0
    battery_for_minimum_electrolyzer_op=True#If true, then dispatch battery (if on) to supply minimum power for operation to PEM, otherwise use it for rated PEM power
    user_defined_stack_replacement_time = False #if true then not dependent on pem performance and set to constant
    if electrolysis_scale=='Centralized':
        default_n_pem_clusters=8
    else:
        default_n_pem_clusters = 1 #to be set to nTurbs


    scenario_choice = 'Green Steel Ammonia Analysis'
    
    scenario = dict()
    kw_continuous = electrolyzer_size_mw * 1000
    load = [kw_continuous for x in
            range(0, 8760)]  # * (sin(x) + pi) Set desired/required load profile for plant
    if battery_for_minimum_electrolyzer_op:
        battery_dispatch_load = list(0.1*np.array(load))
    else:
        battery_dispatch_load = list(np.array(load))
    #Site lat and lon will be set by data loaded from Orbit runs
    
    # Financial inputs
    discount_rate = 0.07
    debt_equity_split = 60
    
    # Wind costs input from ORBIT analysis
    h2_model ='Simple'  #Basic cost model based on H2a and HFTO program record for PEM electrolysis
    # h2_model = 'H2A'
    
    # These inputs are not used in this analysis (no solar or storage)
    solar_cost_kw = 9999 #THESE ARE OVERWRITTEN LATER
    storage_cost_kw = 250 #THESE ARE OVERWRITTEN LATER
    storage_cost_kwh = 240 #THESE ARE OVERWRITTEN LATER
    st_xl=pd.read_csv(parent_path + '/examples/H2_Analysis/storage_costs_ATB.csv',index_col=0)
    storage_costs=st_xl[str(atb_year)]
    storage_cost_main_kwh=storage_costs['Battery Energy Capital Cost ($/kWh)']
    storage_cost_main_kw=storage_costs['Battery Power Capital Cost ($/kW)'] 
    storage_om_percent = 0.025 #percent of capex
    renewable_plant_cost = {}

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
    
    save_outputs_dict = inputs_py.establish_save_output_dict()
    save_all_runs = list()
    
    # which plots to show
    plot_power_production = False
    plot_battery = True
    plot_grid = True
    plot_h2 = True
    plot_desal = True
    plot_wind = True
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
                'plot_power_production': False,
                'plot_battery': False,
                'plot_grid': False,
                'plot_h2': False,
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
    
    turbine_rating = site_df['Turbine Rating']

    # set turbine values
    hopp_dict, scenario, nTurbs, floris_config = hopp_tools_steel.set_turbine_model(hopp_dict, turbine_model, scenario, parent_path,floris_dir, floris)
     
    scenario['Useful Life'] = useful_life

    # financials
    hopp_dict, scenario = hopp_tools_steel.set_financial_info(hopp_dict, scenario, debt_equity_split, discount_rate)

    # set electrolyzer information
    hopp_dict, electrolyzer_capex_kw, capex_ratio_dist, electrolyzer_energy_kWh_per_kg, time_between_replacement =  hopp_tools_steel.set_electrolyzer_info(hopp_dict, atb_year,electrolysis_scale,grid_connection_scenario,turbine_rating,direct_coupling)

    # Extract Scenario Information from ORBIT Runs
    # Load Excel file of scenarios
    # OSW sites and cost file including turbines 8/16/2022 

    # site info
    # solar_size_mw=0
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

    capex_multiplier = site_df['CapEx Multiplier']
    wind_cost_kw = copy.deepcopy(total_capex) * capex_multiplier
    hopp_dict.main_dict['Configuration']['wind_om_cost_kw']=wind_om_cost_kw
    hopp_dict.main_dict['Configuration']['wind_cost_kw']=wind_cost_kw
    solar_main_om_cost_kw=site_df[str(atb_year) + ' PV OpEx']
    solar_capex_multiplier=site_df['PV Capex Multiplier']
    solar_capex=site_df[str(atb_year) + ' PV base installed cost']
    solar_main_cost_kw=solar_capex * solar_capex_multiplier 

    # run_wind_plant

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
    #start for-loop!

    for si,solar_size_mw in enumerate(solar_sizes_mw):
        renewable_plant_cost['wind']={'o&m_per_kw':wind_om_cost_kw,'capex_per_kw':wind_cost_kw,'size_mw':wind_size_mw}
        # hopp_tools_run_wind_solar.run_HOPP_wind_only()
        # hopp_tools_run_wind_solar.run_HOPP_solar_only()
        if si==0:
            run_wind_plant=True
        else:
            run_wind_plant=False
        if solar_size_mw>0:
            solar_cost_kw = copy.copy(solar_main_cost_kw)
            solar_om_cost_kw = copy.copy(solar_main_om_cost_kw)
        else:
            solar_om_cost_kw=0
            solar_cost_kw=0
        hopp_dict.main_dict['Configuration']['solar_size']=solar_size_mw
        hopp_dict.main_dict['Configuration']['solar_cost_kw']=solar_cost_kw
        hopp_dict.main_dict['Configuration']['solar_om_cost_kw']=solar_om_cost_kw
        renewable_plant_cost['pv']={'o&m_per_kw':solar_om_cost_kw,
        'capex_per_kw':solar_cost_kw,
        'size_mw':solar_size_mw}



        for bi,storage_size_mw in enumerate(storage_sizes_mw):
            if storage_size_mw>0:
                storage_size_mwh = storage_sizes_mwh[bi]
                storage_hours = storage_size_mwh/storage_size_mw
                storage_cost_kw=copy.copy(storage_cost_main_kw)
                storage_cost_kwh=copy.copy(storage_cost_main_kwh)
            else:
                storage_cost_kw=0
                storage_cost_kwh=0
                storage_hours= 0
            hopp_dict.main_dict['Configuration']['storage_size_mw']=storage_size_mw
            hopp_dict.main_dict['Configuration']['storage_size_mwh']=storage_size_mwh
            hopp_dict.main_dict['Configuration']['battery_cost_kw']=storage_cost_kw
            hopp_dict.main_dict['Configuration']['battery_cost_kwh']=storage_cost_kwh
            # hopp_dict['Configuration']['battery_om_percent'] = storage_om_percent
            renewable_plant_cost['battery']={'capex_per_kw':solar_cost_kw,
            'capex_per_kwh':solar_cost_kw,
            'o&m_percent':storage_om_percent,
            'size_mw':storage_size_mw,
            'size_mwh':storage_size_mwh,
            'storage_hours':storage_hours}
            




            # Run HOPP
            hopp_dict, plant_power_production, plant_shortfall_hopp, plant_curtailment_hopp, hybrid_plant, wind_size_mw, solar_size_mw, lcoe = \
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
                            solar_om_cost_kw,
                            nTurbs,
                            floris_config,
                            floris,
                            run_wind_plant
                        )
            
            if run_wind_plant:
                cf_wind_annuals = hybrid_plant.wind._financial_model.Outputs.cf_annual_costs
                wind_itc_total = hybrid_plant.wind._financial_model.Outputs.itc_total
                wind_plant_power = hybrid_plant.wind.generation_profile
                if solar_size_mw>0:
                    solar_plant_power = hybrid_plant.pv.generation_profile[0:len(wind_plant_power)]
                hopp_dict.main_dict['Configuration']['wind_plant_object']=hybrid_plant.wind
                if floris:
                    #ACTUAL WIND SIZE
                    hopp_dict.main_dict['Configuration']['n_Turbs']=hybrid_plant.wind._system_model.nTurbs
                    hopp_dict.main_dict['Configuration']['turb_rating_kw']=hybrid_plant.wind._system_model.turb_rating
                    hopp_dict.main_dict['Configuration']['wind_size_mw']=hybrid_plant.wind._system_model.nTurbs*hybrid_plant.wind._system_model.turb_rating*(1/1000)
                    wind_size_mw=hybrid_plant.wind._system_model.nTurbs*hybrid_plant.wind._system_model.turb_rating*(1/1000)
                combined_pv_wind_power_production_hopp = plant_power_production
                # wind_plant_power = np.copy(plant_power_production)
                # wind_only_lcoe = copy.copy(lcoe)
                #TODO change wind_size_mw to account for oversizing
            else:
                # solar_storage_only_lcoe=copy.copy(lcoe)
                if solar_size_mw>0:
                    pv_plant_power = hybrid_plant.pv.generation_profile[0:len(wind_plant_power)]
                    combined_pv_wind_power_production_hopp = np.array(pv_plant_power) + np.array(wind_plant_power)
                else:
                    combined_pv_wind_power_production_hopp=plant_power_production
            
            energy_shortfall_hopp = [x - y for x, y in
                             zip(battery_dispatch_load,combined_pv_wind_power_production_hopp)]
            energy_shortfall_hopp = [x if x > 0 else 0 for x in energy_shortfall_hopp]
            combined_pv_wind_curtailment_hopp = [x - y for x, y in
                             zip(combined_pv_wind_power_production_hopp,load)]
            combined_pv_wind_curtailment_hopp = [x if x > 0 else 0 for x in combined_pv_wind_curtailment_hopp]
            combined_pv_wind_curtailment_hopp[0]=0

            generation_summary_df = pd.DataFrame({'Generation profile (kW)': hybrid_plant.grid.generation_profile[0:8760] })
            #generation_summary_df.to_csv(os.path.join(results_dir, 'Generation Summary_{}_{}_{}_{}.csv'.format(site_name,atb_year,turbine_model,scenario['Powercurve File'])))


            #Step 4: Plot HOPP Results
            # plot_results.plot_HOPP(combined_pv_wind_power_production_hopp,
            #                         energy_shortfall_hopp,
            #                         combined_pv_wind_curtailment_hopp,
            #                         load,
            #                         results_dir,
            #                         site_name,
            #                         atb_year,
            #                         turbine_model,
            #                         hybrid_plant,
            #                         plot_power_production)

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
                    
                pipeline_material_cost = pipe_network_costs_USD['Total material cost ($)'].sum()
                    
                # Eventually replace with calculations   
                if site_name == 'TX':
                    cabling_material_cost = 44553030

                if site_name == 'IA':
                    cabling_material_cost = 44514220
                if site_name == 'IN':
                    cabling_material_cost = 44553030
                if site_name == 'WY': 
                    cabling_material_cost = 44514220
                if site_name == 'MS':
                    cabling_material_cost = 62751510
                transmission_cost = 0
                
                cabling_vs_pipeline_cost_difference = cabling_material_cost - pipeline_material_cost  
                
                turbine_power_electronics_savings = 13
                
            elif electrolysis_scale == 'Centralized':
                cabling_vs_pipeline_cost_difference = 0
                if grid_connection_scenario == 'hybrid-grid' or grid_connection_scenario == 'grid-only':
                    if site_name == 'TX':
                        transmission_cost = 83409258
                    if site_name == 'IA':
                        transmission_cost = 68034484
                    if site_name == 'IN':
                        transmission_cost = 81060771
                    if site_name == 'WY': 
                        transmission_cost = 68034484
                    if site_name == 'MS':
                        transmission_cost = 77274704
                else:
                    transmission_cost = 0
                    
                turbine_power_electronics_savings = 0
                    
            revised_wind_renewable_cost = hybrid_plant.grid.total_installed_cost - cabling_vs_pipeline_cost_difference - turbine_power_electronics_savings*wind_size_mw*1000 + transmission_cost
            renewable_plant_cost['wind_savings_dollars']={'turbine_power_electronics_savings_dollars':-1*turbine_power_electronics_savings*wind_size_mw*1000,
            'tranmission_cost_dollars':transmission_cost,'cabling_vs_pipeline_cost_difference_dollars':-1*cabling_vs_pipeline_cost_difference}
            # Step 6: Run RODeO or Profast for hydrogen
            
            if run_RODeO_selector == True:
                rodeo_scenario,lcoh,electrolyzer_capacity_factor,hydrogen_storage_duration_hr,hydrogen_storage_capacity_kg,\
                    hydrogen_annual_production,water_consumption_hourly,RODeO_summary_results_dict,hydrogen_hourly_results_RODeO,\
                        electrical_generation_timeseries,electrolyzer_installed_cost_kw,hydrogen_storage_cost_USDprkg\
                    = run_RODeO.run_RODeO(atb_year,site_name,turbine_model,electrolysis_scale,policy_option,policy,i,wind_size_mw,solar_size_mw,electrolyzer_size_mw,\
                            energy_to_electrolyzer,electrolyzer_energy_kWh_per_kg,hybrid_plant,renewable_plant_cost,electrolyzer_capex_kw,capex_ratio_dist,wind_om_cost_kw,useful_life,time_between_replacement,\
                            grid_connection_scenario,grid_price_scenario,gams_locations_rodeo_version,rodeo_output_dir)
                
                        
                hydrogen_lifecycle_emissions = LCA_single_scenario.hydrogen_LCA_singlescenario(grid_connection_scenario,atb_year,site_name,turbine_model,electrolysis_scale,\
                                                                                            policy_option,grid_price_scenario,electrolyzer_energy_kWh_per_kg,hydrogen_hourly_results_RODeO)
                
                # Max hydrogen production rate [kg/hr]
                max_hydrogen_production_rate_kg_hr = hydrogen_hourly_results_RODeO['Electrolyzer hydrogen production [kg/hr]'].max()
                max_hydrogen_delivery_rate_kg_hr = hydrogen_hourly_results_RODeO['Product Sold (units of product)'].max()  
                
                electrolyzer_capacity_factor = RODeO_summary_results_dict['input capacity factor']
                
            else:
            # If not running RODeO, run H2A via ProFAST
                # Currently only works for offgrid
                #grid_string = 'offgrid'    
                #scenario_name = 'steel_'+str(atb_year)+'_'+ site_location.replace(' ','-') +'_'+turbine_model+'_'+grid_string
                
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
                    lcoe,
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
                    storage_type = 'Salt cavern' #Unsure
                
                hydrogen_production_storage_system_output_kgprhr,hydrogen_storage_capacity_kg,hydrogen_storage_capacity_MWh_HHV,hydrogen_storage_duration_hr,hydrogen_storage_cost_USDprkg,storage_status_message\
                    = hopp_tools_steel.hydrogen_storage_capacity_cost_calcs(H2_Results,electrolyzer_size_mw,storage_type)   
                print(storage_status_message)
                
                # Run ProFAST to get LCOH
                
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
            
            
                electrolyzer_efficiency_while_running = []
                water_consumption_while_running = []
                hydrogen_production_while_running = []
                for j in range(len(H2_Results['electrolyzer_total_efficiency'])):
                    if H2_Results['hydrogen_hourly_production'][j] > 0:
                        electrolyzer_efficiency_while_running.append(H2_Results['electrolyzer_total_efficiency'][j])
                        water_consumption_while_running.append(H2_Results['water_hourly_usage'][j])
                        hydrogen_production_while_running.append(H2_Results['hydrogen_hourly_production'][j])

                # Specify grid cost year for ATB year
                if atb_year == 2020:
                    grid_year = 2025
                elif atb_year == 2025:
                    grid_year = 2030
                elif atb_year == 2030:
                    grid_year = 2035
                elif atb_year == 2035:
                    grid_year = 2040
                        
                # Read in csv for grid prices
                grid_prices = pd.read_csv('examples/H2_Analysis/annual_average_retail_prices.csv',index_col = None,header = 0)
                elec_price = grid_prices.loc[grid_prices['Year']==grid_year,site_name].tolist()[0]
                
                electrolysis_total_EI_policy_grid,electrolysis_total_EI_policy_offgrid\
                    = LCA_single_scenario_ProFAST.hydrogen_LCA_singlescenario_ProFAST(grid_connection_scenario,atb_year,site_name,policy_option,hydrogen_production_while_running,\
                                                                    electrolyzer_energy_kWh_per_kg)
                        
                h2a_solution,h2a_summary,lcoh_breakdown,electrolyzer_installed_cost_kw = run_profast_for_hydrogen. run_profast_for_hydrogen(site_location,electrolyzer_size_mw,H2_Results,\
                                                electrolyzer_capex_kw,time_between_replacement,electrolyzer_energy_kWh_per_kg,hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg,\
                                                desal_capex,desal_opex,useful_life,water_cost,wind_size_mw,solar_size_mw,hybrid_plant,renewable_plant_cost,wind_om_cost_kw,grid_connected_hopp,grid_connection_scenario, atb_year, site_name, policy_option, energy_to_electrolyzer, elec_price, grid_price_scenario)
                
                lcoh = h2a_solution['price']
                # # Max hydrogen production rate [kg/hr]
                max_hydrogen_production_rate_kg_hr = np.max(H2_Results['hydrogen_hourly_production'])
                max_hydrogen_delivery_rate_kg_hr  = np.mean(H2_Results['hydrogen_hourly_production'])
                
                electrolyzer_capacity_factor = H2_Results['cap_factor']

            # Calculate hydrogen transmission cost and add to LCOH
            hopp_dict,h2_transmission_economics_from_profast,h2_transmission_economics_summary,h2_transmission_price,h2_transmission_price_breakdown = hopp_tools_steel.levelized_cost_of_h2_transmission(hopp_dict,max_hydrogen_production_rate_kg_hr,
            max_hydrogen_delivery_rate_kg_hr,electrolyzer_capacity_factor,atb_year,site_name)
            
            lcoh = lcoh + h2_transmission_price
            print('LCOH without policy:', lcoh)
            # Policy impacts on LCOH
            
            if run_RODeO_selector == True: 
                lcoh,lcoh_reduction_Ren_PTC,lcoh_reduction_H2_PTC, = hopp_tools_steel.policy_implementation_for_RODeO(grid_connection_scenario, atb_year, site_name, turbine_model, electrolysis_scale, policy_option, grid_price_scenario, electrolyzer_energy_kWh_per_kg, hydrogen_hourly_results_RODeO, RODeO_summary_results_dict, hydrogen_annual_production, useful_life, lcoh)
                
                print('LCOH with policy:', lcoh)
            
            # Step 7: Calculate break-even cost of steel production without oxygen and heat integration
            lime_unit_cost = site_df['Lime ($/metric tonne)'] + site_df['Lime Transport ($/metric tonne)']
            carbon_unit_cost = site_df['Carbon ($/metric tonne)'] + site_df['Carbon Transport ($/metric tonne)']
            iron_ore_pellets_unit_cost = site_df['Iron Ore Pellets ($/metric tonne)'] + site_df['Iron Ore Pellets Transport ($/metric tonne)']
            o2_heat_integration = 0
            hopp_dict,steel_economics_from_profast, steel_economics_summary, steel_breakeven_price, steel_annual_production_mtpy,steel_price_breakdown = hopp_tools_steel.steel_LCOS(hopp_dict,lcoh,hydrogen_annual_production,
                                                                                                                    lime_unit_cost,
                                                                                                                    carbon_unit_cost,
                                                                                                                    iron_ore_pellets_unit_cost,
                                                                                                                    o2_heat_integration,atb_year,site_name)
            
            
            # Calcualte break-even price of steel WITH oxygen and heat integration
            o2_heat_integration = 1
            hopp_dict,steel_economics_from_profast_integration, steel_economics_summary_integration, steel_breakeven_price_integration, steel_annual_production_mtpy_integration,steel_price_breakdown_integration = hopp_tools_steel.steel_LCOS(hopp_dict,lcoh,hydrogen_annual_production,
                                                                                                                    lime_unit_cost,
                                                                                                                    carbon_unit_cost,
                                                                                                                    iron_ore_pellets_unit_cost,
                                                                                                                    o2_heat_integration,atb_year,site_name)
            
            
            # Calculate break-even price of ammonia
            cooling_water_cost = 0.000113349938601175 # $/Gal
            iron_based_catalyst_cost = 23.19977341 # $/kg
            oxygen_cost = 0.0285210891617726       # $/kg 
            hopp_dict,ammonia_economics_from_profast, ammonia_economics_summary, ammonia_breakeven_price, ammonia_annual_production_kgpy,ammonia_price_breakdown = hopp_tools_steel.levelized_cost_of_ammonia(hopp_dict,lcoh,hydrogen_annual_production,
                                                                                                                    cooling_water_cost,
                                                                                                                    iron_based_catalyst_cost,
                                                                                                                    oxygen_cost, 
                                                                                                                    atb_year,site_name)
                    
            # Step 7: Write outputs to file
            
            total_h2export_system_cost=0
            opex_pipeline=0
            total_export_system_cost=0
            total_export_om_cost=0
            
            if run_RODeO_selector == True:             
                policy_option,turbine_model,scenario['Useful Life'], wind_cost_kw, solar_cost_kw,\
                scenario['Debt Equity'], atb_year, scenario['H2 PTC'],scenario['Wind ITC'],\
                discount_rate, tlcc_wind_costs, tlcc_solar_costs, tlcc_hvdc_costs, tlcc_total_costs,run_RODeO_selector,lcoh,\
                wind_itc_total, total_itc_hvdc = hopp_tools_steel.write_outputs_RODeO(electrical_generation_timeseries,\
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
                                    h2_transmission_price,
                                    lcoh_reduction_Ren_PTC,
                                    lcoh_reduction_H2_PTC,
                                    electrolyzer_capacity_factor,
                                    hydrogen_storage_duration_hr,
                                    hydrogen_storage_capacity_kg,
                                    hydrogen_annual_production,
                                    water_consumption_hourly,
                                    RODeO_summary_results_dict,
                                    steel_annual_production_mtpy,
                                    steel_breakeven_price,
                                    steel_price_breakdown,
                                    steel_breakeven_price_integration,
                                    ammonia_annual_production_kgpy,
                                    ammonia_breakeven_price,
                                    ammonia_price_breakdown) 
            else:
                policy_option,turbine_model,scenario['Useful Life'], wind_cost_kw, solar_cost_kw,\
                scenario['Debt Equity'], atb_year, scenario['H2 PTC'],scenario['Wind ITC'],\
                discount_rate, tlcc_wind_costs, tlcc_solar_costs, tlcc_hvdc_costs, tlcc_total_costs,run_RODeO_selector,lcoh,\
                wind_itc_total, total_itc_hvdc = hopp_tools_steel.write_outputs_ProFAST(electrical_generation_timeseries,\
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
                                    wind_size_mw,
                                    solar_size_mw,
                                    electrolyzer_size_mw,
                                    discount_rate,
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
                                    h2_transmission_price,
                                    H2_Results,
                                    hydrogen_storage_duration_hr,
                                    hydrogen_storage_capacity_kg,
                                    lcoh_breakdown,
                                    steel_annual_production_mtpy,
                                    steel_breakeven_price,
                                    steel_price_breakdown,
                                    steel_breakeven_price_integration,
                                    ammonia_annual_production_kgpy,
                                    ammonia_breakeven_price,
                                    ammonia_price_breakdown,cf_wind_annuals,wind_itc_total) 

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

                        


