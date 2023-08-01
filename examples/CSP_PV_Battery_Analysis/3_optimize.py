from pathlib import Path
import csv
import json
import pprint
import pandas as pd
import numpy as np
import humpday
import functools
import pyDOE2 as pyDOE

from skopt import gp_minimize

from alt_dev.optimization_problem_alt import HybridSizingProblem
from alt_dev.optimization_driver_alt import OptimizationDriver

from examples.CSP_PV_Battery_Analysis.print_output import print_BCR_table, print_hybrid_output
from examples.CSP_PV_Battery_Analysis.simulation_init import DesignProblem, get_example_path_root

def maxBCR(result):
    "String comes from hybrid_simulation_outputs keys"
    return -result['Hybrid Benefit cost Ratio (-)']

def minimize_real_lcoe(result):
    "String comes from hybrid_simulation_outputs keys"
    return result['Hybrid Real Levelized Cost of Energy ($/MWh)']

def minimize_nom_loce(result):
    return result['Hybrid Nominal Levelized Cost of Energy ($/MWh)']


if __name__ == '__main__':
    is_test = True
    # This example will do a parametric study on design variables spec in design problem    
    run_name = get_example_path_root() + 'testing_optimization'      # Name of optimization run
    write_to_csv = True                    # Writes result to both a pandas dataframe and a csv file (True), else just a pandas dataframe

    # Cases to run with technologies to include
    cases = {
        'pv_batt': ['pv', 'battery'],
        # 'tower': ['tower'],
        # 'tower_pv': ['tower', 'pv'],
        # 'tower_pv_batt': ['tower', 'pv', 'battery'],
        # 'trough': ['trough'],
        # 'trough_pv_batt': ['trough', 'pv', 'battery']
        }

    # Optimization parameters
    N_calls = 3                   # Number of optimization calls
    N_init_points = 4              # Number of evaluations of func with initialization points
    N_processors = 4               # Number of processors available for parallelization

    for case in cases.keys():
        techs = cases[case]
        prob = DesignProblem(techs, is_test = is_test)
        # Driver configuration
        driver_config = dict(n_proc=N_processors, cache_dir=run_name + '/' + case, reconnect_cache=False, write_csv=write_to_csv)
        driver = OptimizationDriver(prob.create_problem, **driver_config)

        # More information on skopt : https://scikit-optimize.github.io/stable/ 
        # Base configuration for skopt minimizer
        opt_config = dict(dimensions=[(0., 1.)]*prob.get_problem_dimen(),
                            n_calls=N_calls + N_init_points,
                            verbose=False,
                            n_initial_points=N_init_points,
                            )

        opt_config['noise'] = 1E-9  # tell Gaussian minimizer the objective is deterministic

        out = driver.optimize([gp_minimize], [opt_config], [minimize_nom_loce])
        
