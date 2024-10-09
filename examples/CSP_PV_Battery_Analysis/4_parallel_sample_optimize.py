import csv
import numpy as np
import pyDOE2 as pyDOE
from skopt import gp_minimize, forest_minimize, gbrt_minimize

from alt_dev.optimization_driver_alt import OptimizationDriver
from examples.CSP_PV_Battery_Analysis.simulation_init import DesignProblem, get_example_path_root


def maxBCR(result):
    "String comes from hybrid_simulation_outputs keys"
    return -result['Hybrid Benefit cost Ratio (-)']

def minimize_real_lcoe(result):
    "String comes from hybrid_simulation_outputs keys"
    return result['Hybrid Real Levelized Cost of Energy ($/MWh)']

if __name__ == "__main__":

    is_test = True
    # This example will do a parametric study on design variables spec in design problem    
    run_name = get_example_path_root() + 'testing_sample_optimization'      # Name of optimization run
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

    # Method
    sample_design = True                # Do sampling of design space
    optimize_design = True              # Run optimization
    output_cache = False                # Reconnect and output cache file

    # Sampling parameters
    save_samples = True            # Save samples values
    sampling_method = 'lhs'        # 'fullfact'= Full factorial, 'lhs' = Latin hypercube sampling
    N_levels = 2                   # 'fullfact': Sets the number of levels for all deminsions in full factorial sampling
    N_samples = 4                  # 'lhs': Sets the total number of samples in the latin hypercube sampling

    # Optimization parameters
    N_calls = 2                   # Number of optimization calls

    # Performance parameters
    N_smb = 1         # Number of small batches
    N_processors = 4   # Number of processors available for parallelization

    for case in cases.keys():
        techs = cases[case]
        prob = DesignProblem(techs, is_test=is_test)

        if sample_design:
            # Driver configuration
            driver_config = dict(n_proc=N_processors, cache_dir=run_name + '/' + case, reconnect_cache=False, write_csv=write_to_csv)
            driver = OptimizationDriver(prob.create_problem, **driver_config)

            # More information about sampling: https://pythonhosted.org/pyDOE/index.html
            if sampling_method is 'fullfact':
                # Full factorial parametric sweep
                levels = np.array([N_levels] * prob.get_problem_dimen())
                design = pyDOE.fullfact(levels)
                levels[levels == 1] = 2
                samples = design / (levels - 1)
            elif sampling_method is 'lhs':
                # Latin Hypercube Sampling of design space
                samples = pyDOE.lhs(prob.get_problem_dimen(), criterion='cm', samples=N_samples)

            # Saves sampling values for post-process analysis
            if save_samples:
                with open(driver.options['cache_dir'] + '/' + case + '_' + sampling_method + '_samples.csv', 'w', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerows(samples)

            # Breaking sampling into small batches to reduce impacts of memory leak
            N_samples_per_batch = len(samples)/N_smb
            for i in range(N_smb):
                start = int(i*N_samples_per_batch)
                stop = int((i+1)*N_samples_per_batch)
                sb_samples = samples[start:stop]

                # Execute Candidates
                driver.parallel_sample(sb_samples, design_name=case)
                print(driver.cache_info)

                # Start a new driver class (for memory leak) with reconnect
                driver_config['reconnect_cache'] = True
                driver_config['cache_dir'] = driver.options['cache_dir']
                driver = OptimizationDriver(prob.create_problem, **driver_config)
                
        if optimize_design:
            # Driver config
            try:
                driver
            except NameError:
                driver_config = dict(n_proc=N_processors, cache_dir=run_name + '/' + case, reconnect_cache=True,  write_csv=write_to_csv)
                driver = OptimizationDriver(prob.create_problem, **driver_config)

            # Get sampled design points from cache (initial points)
            bounds = [(a, b) for a, b in zip(driver.problem.lower_bounds, driver.problem.upper_bounds)]
            c = [k for k in driver.cache if isinstance(k, list)]
            x0 = [[(x[1] - a) / (b-a) for x, (a, b) in zip(k, bounds)] for k in c]

            # Get 'y' values of designs from cache (objective value)
            obj = [maxBCR(driver.cache[k]) for k in c]

            # Base configuration for skopt minimizer
            base_config = dict(dimensions=[(0., 1.)]*prob.get_problem_dimen(),
                               n_calls=N_calls,
                               x0=x0,
                               verbose=False,
                               n_initial_points=0,
                               acq_func='LCB',
                               kappa=0.25  # higher for more exploration, lower for less (default = 1.96)
                               )

            # List of minimizer functions
            optimizers = [gp_minimize, forest_minimize, gbrt_minimize]
            opt_configs = [base_config.copy() for i in range(len(optimizers))]
            opt_configs[0]['noise'] = 1E-9  # tell Gaussian minimizer the objective is deterministic

            # Edit config (for each objective)
            for i in range(len(optimizers)):
                opt_configs[i]['y0'] = obj

            # list of objective functions (accepting a problem result dictionary as input)
            objectives = [maxBCR]*len(optimizers)

            # parallel execute optimizers on the objectives of interest
            out = driver.parallel_optimize(optimizers, opt_configs, objectives)

        if output_cache:
            # Driver config
            driver_config = dict(n_proc=N_processors, cache_dir=run_name + '/' + case, reconnect_cache=True, write_csv=write_to_csv)
            driver = OptimizationDriver(prob.create_problem, **driver_config)
            driver.write_cache()