import os
import sys
sys.path.append('')
from dotenv import load_dotenv
import pandas as pd
#import jsonrun_profast_for_hydrogen
from hybrid.sites import SiteInfo
from hybrid.sites import flatirons_site as sample_site
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
from multiprocessing import Pool
warnings.filterwarnings("ignore")

import hopp_tools
import inputs_py
import copy 
import plot_results
import run_RODeO
import run_profast_for_hydrogen
import run_profast_for_steel

from green_steel_ammonia_run_scenarios import batch_generator_kernel

# Establish directories
parent_path = os.path.abspath('')
#results_dir = parent_path + '\\examples\\H2_Analysis\\results\\'
results_dir = parent_path + '/examples/H2_Analysis/results/'
fin_sum_dir = parent_path + '/examples/H2_Analysis/Phase1B/Fin_summary/'
energy_profile_dir = parent_path + '/examples/H2_Analysis/Phase1B/Energy_profiles/'
price_breakdown_dir = parent_path + '/examples/H2_Analysis/Phase1B/ProFAST_price/'
floris_dir = parent_path + '/floris_input_files/'
orbit_path = ('examples/H2_Analysis/OSW_H2_sites_turbines_and_costs.xlsx')
renewable_cost_path = ('examples/H2_Analysis/green_steel_site_renewable_costs_ATB.xlsx')
floris = False

# Turn to False to run ProFAST for hydrogen LCOH 
run_RODeO_selector = False

# Grid price scenario ['wholesale','retail-peaks','retail-flat']
grid_price_scenario = 'retail-flat'

if run_RODeO_selector == True:
    # RODeO requires output directory in this format, but apparently this format
    # is problematic for people who use Mac
    rodeo_output_dir = 'examples\\H2_Analysis\\RODeO_files\\Output_test\\'
else:
    # People who use Mac probably won't be running RODeO, so we can just give
    # the model a dummy string for this variable
    rodeo_output_dir = 'examples/H2_Analysis/RODeO_files/Output_test/'

# Distributed scale power electronics direct coupling information
direct_coupling = True

# Electrolzyer cost case ('Mid' or 'Low')
electrolyzer_cost_case = 'Low'

# Degradation penalties for capital costs to estimate cost of plant oversizing
electrolyzer_degradation_power_increase = 0.13
wind_plant_degradation_power_decrease = 0.08

# Determine if run with electrolyzer degradation or not
electrolyzer_degradation_penalty = True

# Determine if PEM stack operation is optimized or not
pem_control_type = 'basic' #use 'optimize' for Sanjana's controller; 'basic' to not optimize
    
save_hybrid_plant_yaml = True # hybrid_plant requires special processing of the SAM objects
save_model_input_yaml = True # saves the inputs for each model/major function
save_model_output_yaml = True # saves the outputs for each model/major function

# Target steel production rate. Note that this is the production after taking into account
# steel plant capacity factor. E.g., if CF is 0.9, divide the number below by 0.9 to get
# the total steel plant capacity used for economic calculations
steel_annual_production_rate_target_tpy = 1000000

if __name__ == '__main__':
#-------------------- Define scenarios to run----------------------------------
    
    atb_years = [
                #2020,
                #2025,
                2030,
                #2035
                ]

    policy = {
        'no-policy': {'Wind ITC': 0, 'Wind PTC': 0, "H2 PTC": 0, 'Storage ITC': 0},
        #'base': {'Wind ITC': 0, 'Wind PTC': 0.0051, "H2 PTC": 0.6, 'Storage ITC': 0.06},
        #'max': {'Wind ITC': 0, 'Wind PTC': 0.03072, "H2 PTC": 3.0, 'Storage ITC': 0.5},   
        # 'max on grid hybrid': {'Wind ITC': 0, 'Wind PTC': 0.0051, "H2 PTC": 0.60, 'Storage ITC': 0.06},
        # 'max on grid hybrid': {'Wind ITC': 0, 'Wind PTC': 0.026, "H2 PTC": 0.60, 'Storage ITC': 0.5},
        # 'option 3': {'Wind ITC': 0.06, 'Wind PTC': 0, "H2 PTC": 0.6}, 
        # 'option 4': {'Wind ITC': 0.3, 'Wind PTC': 0, "H2 PTC": 3},
        # 'option 5': {'Wind ITC': 0.5, 'Wind PTC': 0, "H2 PTC": 3}, 
    }
    
    
    site_selection = [
                    #'Site 1',
                    'Site 2',
                    #'Site 3',
                    #'Site 4',
                    #'Site 5'
                    ] 
    
    electrolysis_cases = [
                          'Centralized',
                          #'Distributed'
                          ]
    
    grid_connection_cases = [
                            'off-grid',
                            #'grid-only',
                            #'hybrid-grid'
                            ]

    storage_capacity_cases = [
                            1.0,
                            #1.25,
                            #1.5
                            ] 

    num_pem_stacks= 6
    run_solar_param_sweep=False
#---- Create list of arguments to pass to batch generator kernel --------------    
    arg_list = []
    for i in policy:
        for atb_year in atb_years:
            for site_location in site_selection:
                for electrolysis_scale in electrolysis_cases:
                    for grid_connection_scenario in grid_connection_cases:
                        for storage_capacity_multiplier in storage_capacity_cases:
                            arg_list.append([policy, i, atb_year, site_location, electrolysis_scale,run_RODeO_selector,floris,\
                                            grid_connection_scenario,grid_price_scenario,\
                                            direct_coupling,electrolyzer_cost_case,electrolyzer_degradation_power_increase,wind_plant_degradation_power_decrease,\
                                                steel_annual_production_rate_target_tpy,parent_path,results_dir,fin_sum_dir,energy_profile_dir,price_breakdown_dir,rodeo_output_dir,floris_dir,renewable_cost_path,\
                                            save_hybrid_plant_yaml,save_model_input_yaml,save_model_output_yaml,num_pem_stacks,run_solar_param_sweep,electrolyzer_degradation_penalty,\
                                                pem_control_type,storage_capacity_multiplier])
    for runs in range(len(arg_list)):
        batch_generator_kernel(arg_list[runs])
    []
# ------------------ Run HOPP-RODeO/PyFAST Framework to get LCOH ---------------            
    # with Pool(processes=8) as pool:
    #         pool.map(batch_generator_kernel, arg_list)
            