import csv
import numpy as np
import pyDOE2 as pyDOE

from alt_dev.optimization_driver_alt import OptimizationDriver

from simulation_init import DesignProblem, get_example_path_root

if __name__ == '__main__':
    is_test = True
    # This example will do a parametric study on design variables spec in design problem    
    run_name = get_example_path_root + 'training_test'      # Name of sampling run
    write_to_csv = True             # Writes result to both a pandas dataframe and a csv file (True), else just a pandas dataframe

    # Cases to run with technologies to include
    cases = {
        'pv_batt': ['pv', 'battery'],
        # 'tower': ['tower'],
        # 'tower_pv': ['tower', 'pv'],
        # 'tower_pv_batt': ['tower', 'pv', 'battery'],
        # 'trough': ['trough'],
        # 'trough_pv_batt': ['trough', 'pv', 'battery']
        }

    # Sampling parameters
    save_samples = True            # Save samples values
    sampling_method = 'fullfact'   # 'fullfact'= Full factorial, 'lhs' = Latin hypercube sampling
    N_levels = 2                   # 'fullfact': Sets the number of levels for all deminsions in full factorial sampling
    N_samples = 10                  # 'lhs': Sets the total number of samples in the latin hypercube sampling

    # Performance parameters
    N_smb = 2         # Number of small batches (limit 'tower' samples to less than 100)
    N_processors = 4   # Number of processors available for parallelization

    for case in cases.keys():
        techs = cases[case]
        prob = DesignProblem(techs, is_test = is_test)
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
            
