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
# import shapely

# sys.path.append('../examples/flatirons')
# import func_tools

from defaults.defaults_data import (
    Site,
    )
from examples.wind_opt.site_info import SiteInfo
from examples.wind_opt.wind_opt_DCEMC import WindOptDCEMC
from examples.wind_opt.wind_optimization_problem import WindOptimizationProblem

# mpl.use('module://backend_interagg')
# sys.path.append('../')
# import wind.opt_tools

np.set_printoptions(precision=2, threshold=10000, linewidth=240)


# class WindOptKFDCE(WindOpt):
#
#     def __init__(self,
#                  site_info: SiteInfo,
#                  num_turbines: int,
#                  generation_size: int,
#                  selection_proportion: float,
#                  sensor_noise: float = 0.0,
#                  dynamic_noise: float = 0.0):
#         super().__init__(site_info, num_turbines)
#
#         dimensions = [None] * (num_turbines * 2)
#         for i, dist in enumerate(self.get_starting_distributions()):
#             dimensions[i] = KFDCEM.KFDimension(
#                 dist[0],
#                 dist[1],
#                 sensor_noise,
#                 dynamic_noise)
#             dimensions[num_turbines + i] = KFDCEM.KFDimension(
#                 dist[2],
#                 dist[3],
#                 sensor_noise,
#                 dynamic_noise)
#
#         self.optimizer = KFDCEM(
#             dimensions,
#             generation_size,
#             selection_proportion)
#
#
# class WindOptKFPDCE(WindOpt):
#
#     def __init__(self,
#                  site_info: SiteInfo,
#                  num_turbines: int,
#                  generation_size: int,
#                  selection_proportion: float,
#                  mu_variance: float,
#                  mu_sensor_noise: float,
#                  mu_dynamic_noise: float,
#                  sigma_variance: float,
#                  sigma_sensor_noise: float,
#                  sigma_dynamic_noise: float):
#         super().__init__(site_info, num_turbines)
#
#         dimensions = [None] * (num_turbines * 2)
#         for i, dist in enumerate(self.get_starting_distributions()):
#             dimensions[i] = KFDCEM.KFParameterDimension(
#                 KFDCEM.KFDimension(
#                     dist[0],
#                     mu_variance,
#                     mu_sensor_noise,
#                     mu_dynamic_noise),
#                 KFDCEM.KFDimension(
#                     dist[1],
#                     sigma_variance,
#                     sigma_sensor_noise,
#                     sigma_dynamic_noise))
#             dimensions[num_turbines + i] = KFDCEM.KFParameterDimension(
#                 KFDCEM.KFDimension(
#                     dist[2],
#                     mu_variance,
#                     mu_sensor_noise,
#                     mu_dynamic_noise),
#                 KFDCEM.KFDimension(
#                     dist[3],
#                     sigma_variance,
#                     sigma_sensor_noise,
#                     sigma_dynamic_noise))
#
#         self.optimizer = KFDCEM(
#             dimensions,
#             generation_size,
#             selection_proportion)


def run_dce_test():
    # plt.interactive(False)
    # plt.figure()
    # plt.plot([1, 2, 3], [4, 5, 6])
    # plt.show()
    #
    figure = plt.figure(1)
    ax = figure.add_subplot(111)
    # fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    
    iters = 50
    
    num_turbines = 20
    site_info = SiteInfo(Site)
    problem = WindOptimizationProblem(site_info, num_turbines)
    
    # opt = WindOptDCEM(problem)
    opt = WindOptDCEMC(problem)
    # opt = WindOptIDCEM(problem)
    # opt = WindOptIDCEMC(problem)
    # opt = WindOptDCE(site, 20, 100, .1, 1.0)
    # opt = WindOptKFDCE(site, 10, 200, .95, sensor_noise=1000, dynamic_noise=0.0)
    # opt = WindOptKFPDCE(site, 20, 100, .1,
    #                     mu_variance=100000,
    #                     mu_sensor_noise=10,
    #                     mu_dynamic_noise=100.0,
    #                     sigma_variance=100000,
    #                     sigma_sensor_noise=10,
    #                     sigma_dynamic_noise=100.0)
    # opt = WindOptIDCEM(site, 20, 100, 10, 1.1)
    # opt = WindOptIWDCEM(site, 20, 100, 300, .1)
    # opt = WindOptIPDCEM(site, 20, 100, 10, .5)
    # opt = WindOptParticleGradientOptimizer(site, 20, 100, 10, 1.0)
    # opt.plot((.5, .5, .5))
    
    plt.grid()
    plt.tick_params(which='both', labelsize=15)
    plt.xlabel('x (m)', fontsize=15)
    plt.ylabel('y (m)', fontsize=15)
    site_info.plot()
    
    performance = []
    
    perf = problem.objective(opt.best())
    performance.append(perf)
    print(-1, ' ', perf)
    
    opt.plot_distribution(ax, (0, 0, 0), .25)
    opt.problem.plot_candidate(opt.best(), (1.0, 0, 0), 1.0)
    
    prev = opt.best()
    for i in range(iters):
        opt.step()
        g = i * 1.0 / iters
        b = 1.0 - g
        a = .05 + .95 * g
        # opt.plot((0, g, r))
        color = (b, g, b)
        opt.plot_distribution(ax, color, .1 * a)
        best = opt.best()
        perf = problem.objective(best)
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
        
        performance.append(perf)
        print(i, ' ', perf)
    # opt.plot((0, 0, 0))
    
    opt.problem.plot_candidate(opt.best(), (0, 0, 0), 1.0)
    
    plt.show(block=False)
    plt.figure(2)
    plt.plot(performance)
    plt.xlabel('iteration', fontsize=15)
    plt.ylabel('power', fontsize=15)
    plt.show()


run_dce_test()
