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

from defaults.flatirons_site import (
    Site,
    )
from examples.command_line_config.run_utils import (
    setup_run,
    plot_from_logger,
    )
from hybrid.site_info import SiteInfo
from examples.wind_opt.wind_opt_DCEM import WindOptDCEM
from examples.wind_opt.wind_opt_GA import WindOptGA
from examples.wind_opt.wind_opt_SPSA import WindOptSPSA
from examples.wind_opt.wind_optimization_problem import WindOptimizationProblem

np.set_printoptions(precision=2, threshold=10000, linewidth=240)


def run(default_config: {}) -> None:
    config, recorder = setup_run(default_config)

    max_evaluations = config['max_evaluations']
    optimizer_selection = config['optimizer']
    optimizer_config = config['optimizer_config']
        
    site_info = SiteInfo(Site)
    problem = WindOptimizationProblem(site_info, config['num_turbines'])

    optimizer = None
    if optimizer_selection == 'GA':
        optimizer = WindOptGA(problem, recorder=recorder, **optimizer_config)
    elif optimizer_selection == 'SPSA':
        optimizer = WindOptSPSA(problem, recorder=recorder, **optimizer_config)
    elif optimizer_selection == 'DCEM':
        optimizer = WindOptDCEM(problem, recorder=recorder, **optimizer_config)
    else:
        raise Exception('Unknown optimizer: "' + optimizer_selection + '"')
    
    # optimizer = WindOptGA(problem)
    # optimizer = WindOptSPSA(problem)
    # optimizer = WindOptDCEM(problem)
    # optimizer = WindOptDCEMC(problem)
    # optimizer = WindOptIDCEM(problem)
    # optimizer = WindOptIDCEMC(problem)
    # optimizer = WindOptDCE(site, 20, 100, .1, 1.0)
    # optimizer = WindOptKFDCE(site, 10, 200, .95, sensor_noise=1000, dynamic_noise=0.0)
    # optimizer = WindOptKFPDCE(site, 20, 100, .1,
    #                     mu_variance=100000,
    #                     mu_sensor_noise=10,
    #                     mu_dynamic_noise=100.0,
    #                     sigma_variance=100000,
    #                     sigma_sensor_noise=10,
    #                     sigma_dynamic_noise=100.0)
    # optimizer = WindOptIDCEM(site, 20, 100, 10, 1.1)
    # optimizer = WindOptIWDCEM(site, 20, 100, 300, .1)
    # optimizer = WindOptIPDCEM(site, 20, 100, 10, .5)
    # optimizer = WindOptParticleGradientOptimizer(site, 20, 100, 10, 1.0)
    # optimizer.plot((.5, .5, .5))

    figure = plt.figure(1)
    ax = figure.add_subplot(111)
    plt.grid()
    plt.tick_params(which='both', labelsize=15)
    plt.xlabel('x (m)', fontsize=15)
    plt.ylabel('y (m)', fontsize=15)
    site_info.plot()
    
    perf, best = optimizer.best_solution()
    perf = problem.objective(best) if perf is None else perf
    
    print(-1, ' ', perf)
    
    optimizer.plot_distribution(ax, (0, 0, 0), .25)
    optimizer.problem.plot_candidate(optimizer.best_solution()[1], (1.0, 0, 0), 1.0)
    
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
        # optimizer.plot((0, g, r))
        color = (b, g, b)
        optimizer.plot_distribution(ax, color, .1 * a)
        perf, best = optimizer.best_solution()
        perf = problem.objective(best) if perf is None else perf
        problem.plot_candidate(best, color)
        if prev is not None:
            for t in range(problem.num_turbines):
                plt.plot(
                    [prev[t], best[t]],
                    [prev[problem.num_turbines + t], best[problem.num_turbines + t]],
                    color=color,
                    linestyle='-',
                    alpha=.2 + .8 * a)
        prev = best
        print(optimizer._driver.get_num_iterations(), ' ', optimizer._driver.get_num_evaluations(), perf)
    
    optimizer.problem.plot_candidate(optimizer.best_solution()[1], (0, 0, 0), 1.0)
    plt.show(block=False)
    # plt.show()

    plot_from_logger(optimizer.recorder, 'iteration', 'score')
    plot_from_logger(optimizer.recorder, 'num_evaluations', 'score')
    
    plt.show()
    optimizer.close()


default_config = {
    'run_name':         'test',
    'num_turbines':     20,
    'max_evaluations':  5000,
    'optimizer':        'DCEM',
    'optimizer_config': {
        }
    }

run(default_config)
