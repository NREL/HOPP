import matplotlib.pyplot as plt
import numpy as np
import os
from typing import (
    Dict,
    Tuple,
    )

import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from examples.solar_wind_opt.solar_wind_opt_DCEM import SolarWindOptDCEM
from examples.solar_wind_opt.solar_wind_optimization_problem import SolarWindOptimizationProblem
from examples.wind_opt.site_info import SiteInfo
from parameters.parameter_data import get_input_output_data
from defaults.defaults_data import (
    wind_windsingleowner,
    PV_pvsingleowner,
    Site,
    )
from hybrid.systems_behavior import (
    run_wind_models,
    use_empirical_turbine_powercurve_params,
    )
from hybrid.scenario import Scenario

from hybrid.solar_wind.shadow_cast import get_unshaded_areas_on_site
from hybrid.solar_wind.solar_wind_opt import *


# from examples.wind_opt_dce import (
#     SiteInfo,
#     )


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
    problem = SolarWindOptimizationProblem(site_info, 2e3 * 20, num_turbines=10, turbine_kw=2e3)
    
    opt = SolarWindOptDCEM(problem)
    # opt = WindOptDCEMC(problem)
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
                    [prev['turb_x'][t], best['turb_x'][t]],
                    [prev['turb_y'][t], best['turb_y'][t]],
                    color=color,
                    linestyle='-',
                    alpha=.2 + .8 * a)
        prev = best
        
        performance.append(perf)
        print(i, ' ', perf)
    plt.plot((0, 0, 0))
    
    opt.problem.plot_candidate(opt.best(), (0, 0, 0), 1.0)
    
    plt.show(block=False)
    plt.figure(2)
    plt.plot(performance)
    plt.xlabel('iteration', fontsize=15)
    plt.ylabel('power', fontsize=15)
    plt.show()


run_dce_test()

# class SolarWindOpt(WindOpt):
#
#     def __init__(self, site_info: SiteInfo, system_capacity: int):
#         """
#         :param site_info:
#         :param system_capacity: kW
#         """
#         # set num_turbines to 0 since it's unknown and meant to be determined but could handle differently in the
#         future
#         super(SolarWindOpt, self).__init__(site_info, 0)
#         self.system_kw: int = system_capacity
#
#         # set up template mask to reuse during shadow calculation
#         self.site_info: SiteInfo = site_info
#         self.polygon: Polygon = Polygon(self.site_info.vertices)
#         poly_bounds = self.polygon.bounds
#         site_dim = (int(poly_bounds[2] - poly_bounds[0]), int(poly_bounds[3] - poly_bounds[1]))
#         site_mask = np.ones(site_dim)
#
#         if os.path.isfile("examples/site_mask_template.npy"):
#             self.site_mask_orig = np.load("examples/site_mask_template.npy")
#         else:
#             for i in range(site_dim[0]):
#                 for j in range(site_dim[1]):
#                     site_mask[i, j] = self.polygon.contains(Point(i, j))
#             self.site_mask_orig = site_mask
#             np.save("examples/site_mask_template", self.site_mask_orig)
#         self.site_area = np.count_nonzero(self.site_mask_orig)
#
#         # setup scenario to run wind and pvwatts (simple solar model)
#         def run_pvwatts(systems):
#             pvmodel = systems['Pvwatts']['Pvwattsv5']
#             pvmodel.execute()
#
#         systems = {'Wind': run_wind_models, 'Pvwatts': run_pvwatts}
#         defaults = {
#             'Wind':    {'Windpower': wind_windsingleowner},
#             'Pvwatts': {'Pvwattsv5': "PVWattsSingleOwner"}
#             }
#         input_data, output_data = get_input_output_data(systems)
#         self.scenario = Scenario(defaults, input_data, output_data, systems)
#
#         # initialize some settings of the Pvwatts model which are not default
#         model = self.scenario.systems['Pvwatts']['Pvwattsv5']
#         model.SystemDesign.array_type = 2
#         model.SystemDesign.tilt = 0
#         model.SystemDesign.module_type = 1
#         model.LocationAndResource.solar_resource_file = PV_pvsingleowner['SolarResource']['solar_resource_file']
#
#         use_empirical_turbine_powercurve_params()
#
#     # def to_configuration_object(self, candidate):
#     #     result = {}
#     #     # result['turb_x'] = candidate[]
#
#     def objective(self, candidate):
#         """
#
#         :param candidate: dictionary
#             'turb_kw': kW rating of single turbine
#             'turb_x': x coordinates of turbines
#             'turb_y': y coordinates of turbines
#             'gcr': between 0 and 0.65 (industry standard)
#         :return:
#         """
#
#         turb_y = candidate['turb_y']
#         turb_x = candidate['turb_x']
#         turb_kw = candidate['turb_kw']
#         gcr = candidate['gcr']
#
#         penalty = 0
#         scale = .1
#
#         max_unpenalized_distance = 0
#         num_turbines = int(len(turb_y))
#
#         border = self.site_info.border
#         for i in range(num_turbines):
#             point = Point(turb_x[i], turb_y[i])
#             distance = self.site_info.polygon.distance(point)
#             if distance > max_unpenalized_distance:
#                 penalty = penalty + (distance - max_unpenalized_distance)
#
#             closest_valid_point = border.interpolate(border.project(point))
#             turb_x[i] = closest_valid_point.x
#             turb_y[i] = closest_valid_point.y
#
#         penalty = scale * penalty * penalty
#
#         # check if enough space for given pv size
#         windfarm_kw = (len(turb_y) * turb_kw)
#         pv_kw = self.system_kw - windfarm_kw
#         if pv_kw < 0:
#             raise ValueError("too much windpower")
#
#         pv_area = calculate_PV_required_area(pv_kw, gcr)
#         # total_area = PolygonArea(self.site_info.vertices)  # 1,170,406 m^2
#         shade_mask = get_unshaded_areas_on_site(
#             self.site_mask_orig.copy(),
#             turb_x,
#             turb_y,
#             calculate_turbine_radius(turb_kw),
#             threshold=.005,
#             plot_bool=False)
#         unshaded_area = np.count_nonzero(shade_mask)
#         if unshaded_area < pv_area:
#             # not good as it won't fit, need to apply some penalty
#             penalty += 234234
#
#         # modify scenario with candidate values
#         pvmodel = self.scenario.systems['Pvwatts']['Pvwattsv5']
#         pvmodel.SystemDesign.gcr = gcr
#         pvmodel.SystemDesign.system_capacity = pv_kw
#         windmodel = self.scenario.systems['Wind']['Windpower']
#         windmodel.Farm.wind_farm_xCoordinates = turb_x
#         windmodel.Farm.wind_farm_yCoordinates = turb_y
#         windmodel.Farm.system_capacity = windfarm_kw
#
#         score = -optimize_solar_wind_AEP(self.scenario)
#
#         # penalty = 0
#         # num_turbines = int(candidate.shape[0] / 2)
#         # for i in range(num_turbines):
#         #     point = Point(candidate[i], candidate[num_turbines + i])
#         #     penalty = penalty + self.site_info.polygon.distance(point)
#         # penalty = 100 * penalty
#
#         return score - penalty, score
#
#
# def run():
#     site = SiteInfo(Site)
#     opt = SolarWindOpt(site, 20000)
#     cand = {
#         'turb_x':  [0.0, 440.0, 880.0, 1320.0, 1760.0],
#         'turb_y':  [0.0, 0.0, 0.0, 0.0, 0.0],
#         'turb_kw': 2000,
#         'gcr':     0.6
#         }
#     print(opt.objective(cand))
#
#
# run()
