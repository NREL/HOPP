import os
import sys
sys.path.append('')
from dotenv import load_dotenv
import pandas as pd
import json
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
import run_pyfast_for_hydrogen
import run_pyfast_for_steel

from green_steel_ammonia_run_scenarios import batch_generator_kernel

if __name__ == '__main__':
    
#-------------------- Define scenarios to run----------------------------------
    
    atb_years = [
                2022,
                #2025,
                # 2030,
                # 2035
                ]

    policy = {
        'option 1': {'Wind ITC': 0, 'Wind PTC': 0, "H2 PTC": 0},
        # 'option 2': {'Wind ITC': 26, 'Wind PTC': 0, "H2 PTC": 0},
        # 'option 3': {'Wind ITC': 6, 'Wind PTC': 0, "H2 PTC": 0.6},
        # 'option 4': {'Wind ITC': 30, 'Wind PTC': 0, "H2 PTC": 3},
        # 'option 5': {'Wind ITC': 50, 'Wind PTC': 0, "H2 PTC": 3},
    }
    
    turbine_name = [
                    #'12MW',
                    #'15MW',
                    '18MW'
                    ]
    
    site_selection = [
                    'Site 1',
                    # 'Site 2',
                    # 'Site 3',
                    # 'Site 4'
                    ] 
#---- Create list of arguments to pass to batch generator kernel --------------    
    arg_list = []
    for i in policy:
        for atb_year in atb_years:
            for site_location in site_selection:
                for turbine_model in turbine_name:
                    arg_list.append([policy, i, atb_year, site_location, turbine_name, turbine_model])
       
#------------------ Run HOPP-RODeO/PyFAST Framework to get LCOH ---------------            
    with Pool(processes=2) as pool:
            pool.map(batch_generator_kernel, arg_list)
            
