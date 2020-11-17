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
import matplotlib.pyplot as plt
import numpy as np

from tools.optimization import (
    setup_run,
    DataRecorder,
    OptimizationDriver
    )
from hybrid.sites import SiteInfo, flatirons_site

from examples.optimization.wind_opt.wind_optimization_problem import WindOptimizationProblem
from examples.optimization.wind_opt.wind_parametrization import WindParametrization

# import shapely
# sys.path.append('../examples/flatirons')
# import func_tools

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
    config, output_path, run_name = setup_run(default_config)
    recorder = DataRecorder.make_data_recorder(output_path)
    
    max_evaluations = config['max_evaluations']
    optimizer_config = config['optimizer_config']
    
    site_info = SiteInfo(flatirons_site)
    inner_problem = WindOptimizationProblem(site_info, config['num_turbines'])
    problem = WindParametrization(inner_problem)
    
    optimizer = OptimizationDriver(problem, recorder=recorder, **optimizer_config)
    
    figure = plt.figure(1)
    ax = figure.add_subplot(111)
    plt.grid()
    plt.tick_params(which='both', labelsize=15)
    plt.xlabel('x (m)', fontsize=15)
    plt.ylabel('y (m)', fontsize=15)
    site_info.plot()
    
    score, evaluation, best_solution = optimizer.central_solution()
    score, evaluation = problem.objective(best_solution) if score is None else score
    
    # solution_path.append((opt.driver.get_num_iterations(), opt.driver.get_num_evaluations(), perf))
    print(-1, ' ', score)
    
    # opt.plot_distribution(ax, (0, 0, 0), .25)
    optimizer.problem.plot_candidate(best_solution, (1.0, 0, 0), .2)
    
    prev = optimizer.best_solution()[1]
    while optimizer.num_evaluations() < max_evaluations:
        print('step start')
        optimizer.step()
        print('step end')
        
        proportion = min(1.0, optimizer.num_evaluations() / max_evaluations)
        g = 1.0 * proportion
        b = 1.0 - g
        # a = .05 + .95 * g
        a = .5
        # opt.plot((0, g, r))
        color = (b, g, b)
        # opt.plot_distribution(ax, color, .1 * a)
        score, eval, best = optimizer.best_solution()
        score = problem.objective(best) if score is None else score
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
        print(optimizer.num_iterations(), ' ', optimizer.num_evaluations(), score)
    
    print('best: ', optimizer.best_solution().__repr__())
    optimizer.problem.plot_candidate(optimizer.best_solution()[2], (0, 0, 0), 1.0)
    plt.show(block=False)
    
    # plot_from_logger(optimizer.recorder, 'iteration', 'score')
    # plot_from_logger(optimizer.recorder, 'num_evaluations', 'score')
    
    plt.show()
    optimizer.close()


default_config = {
    'name':             'test',
    'num_turbines':     20,
    'max_evaluations':  20,
    'optimizer_config': {
        'method':               'CEM',
        'nprocs':               1,
        'generation_size':      5,
        'selection_proportion': .5,
        'prior_scale':          1.0,
        }
    }

run(default_config)
