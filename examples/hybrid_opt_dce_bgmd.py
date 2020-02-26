"""
A prototype application of the distributed cross-entropy method to the wind optimization problem.
In this basic implementation, the number of turbines is fixed and the generative distribution is uncorrelated.

TODO:
 + Add boundary constraints / penalties
 + Add proximity constraints
 + Better order turbine locations
 + Investigate turbine number as an attribute
 + Investigate modeling parameter covariances
 + Investigate other distribution types
 + Investigate parameter transformations
 + Add solar
 + Add storage
 + Add cabling, etc
 + investigate organic approach
"""

# matplotlib.use('tkagg')
import json
import sys
import time
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from defaults.flatirons_site import (
    Site,
    )
from examples.command_line_config import command_line_config
from examples.command_line_config.run_utils import (
    setup_run,
    plot_from_logger,
    )
from hybrid.site_info import SiteInfo
from examples.wind_opt.wind_opt_BGMD_CEM import WindOptBGMD_CEM
from examples.wind_opt.wind_opt_BGMD_GA import WindOptBGMD_GA
from examples.wind_opt.wind_opt_BGMD_SPSA import WindOptBGMD_SPSA
from examples.hybrid_opt.hybrid_optimization_problem import HybridOptimizationProblem

# import shapely
# sys.path.append('../examples/flatirons')
# import func_tools

# mpl.use('module://backend_interagg')
# sys.path.append('../')
# import wind.opt_tools
# os.environ["OPENBLAS_MAIN_FREE"] = "1"
from examples.hybrid_opt.hybrid_optimization_problem_bgmd import HybridOptimizationProblemBGMD
from optimization.data_logging.data_recorder import DataRecorder
from optimization.data_logging.record_logger import RecordLogger
from optimization.data_logging.JSON_lines_record_logger import JSONLinesRecordLogger

np.set_printoptions(precision=2, threshold=10000, linewidth=240)

"""
TODO:
 + general purpose command line config library
 + make run configurable from command line
 + make log filename configurable
 + set pop size, num evaluations
 + able to choose which optimizer to use
 + able to configure the optimizer
 + notebook for aggregating and plotting the results
"""


def run(default_config: {}) -> None:
    config, recorder = setup_run(default_config)
    
    max_evaluations = config['max_evaluations']
    optimizer_selection = config['optimizer']
    optimizer_config = config['optimizer_config']
    
    site_info = SiteInfo(Site)
    inner_problem = HybridOptimizationProblem(site_info, config['num_turbines'], config['system_size'])
    problem = HybridOptimizationProblemBGMD(inner_problem)
    
    optimizer = None
    if optimizer_selection == 'GA':
        optimizer = WindOptBGMD_GA(problem, recorder=recorder, **optimizer_config)
    elif optimizer_selection == 'CEM':
        optimizer = WindOptBGMD_CEM(problem, recorder=recorder, **optimizer_config)
    elif optimizer_selection == 'SPSA':
        optimizer = WindOptBGMD_SPSA(problem, recorder=recorder, **optimizer_config)
    else:
        raise Exception('Unknown optimizer: "' + optimizer_selection + '"')
    
    figure = plt.figure(1)
    ax = figure.add_subplot(111)
    plt.grid()
    plt.tick_params(which='both', labelsize=15)
    plt.xlabel('x (m)', fontsize=15)
    plt.ylabel('y (m)', fontsize=15)
    site_info.plot()
    
    perf, best = optimizer.best_solution()
    perf = problem.objective(best) if perf is None else perf
    
    # solution_path.append((opt.driver.get_num_iterations(), opt.driver.get_num_evaluations(), perf))
    print(-1, ' ', perf)
    
    # opt.plot_distribution(ax, (0, 0, 0), .25)
    optimizer.problem.plot_candidate(optimizer.best_solution()[1], (1.0, 0, 0), .2)
    prev = optimizer.best_solution()[1]
    while optimizer._driver.get_num_evaluations() < max_evaluations:
        print('step start')
        optimizer.step()
        print('step end')
        
        proportion = min(1.0, optimizer._driver.get_num_evaluations() / max_evaluations)
        g = 1.0 * proportion
        b = 1.0 - g
        # a = .05 + .95 * g
        a = .5
        # opt.plot((0, g, r))
        color = (b, g, b)
        # opt.plot_distribution(ax, color, .1 * a)
        perf, best = optimizer.best_solution()
        perf = problem.objective(best) if perf is None else perf
        problem.plot_candidate(best, color, .3)
        # if prev is not None:
        #     for t in range(problem.num_turbines):
        #         plt.plot(
        #             [prev[t], best[t]],-
        #             [prev[problem.num_turbines + t], best[problem.num_turbines + t]],
        #             color=color,
        #             linestyle='-',
        #             alpha=.2 + .8 * a)
        prev = best
        print(optimizer._driver.get_num_iterations(), ' ', optimizer._driver.get_num_evaluations(), perf)
    
    print('best: ', optimizer.best_solution().__repr__())
    optimizer.problem.plot_candidate(optimizer.best_solution()[1], (0, 0, 0), 1.0)
    plt.show(block=False)
    
    plot_from_logger(optimizer.recorder, 'iteration', 'score')
    plot_from_logger(optimizer.recorder, 'num_evaluations', 'score')
    
    plt.show()
    optimizer.close()


default_config = {
    'run_name':         'test',
    'system_size':      20,     # MW
    'num_turbines':     10,     # default size is 1.5 MW
    'max_evaluations':  2000,
    'optimizer':        'CEM',
    'optimizer_config': {
        }
    }

run(default_config)
