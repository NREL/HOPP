import math
import os
from typing import (
    Tuple,
    )

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point

from defaults.defaults_data import (
    PV_pvsingleowner,
    wind_windsingleowner,
    )
from examples.wind_opt.site_info import SiteInfo
from examples.wind_opt.wind_optimization_problem import WindOptimizationProblem
from hybrid.scenario import Scenario
from hybrid.solar_wind.shadow_cast import get_unshaded_areas_on_site
from hybrid.solar_wind.solar_wind_opt import *
from hybrid.systems_behavior import (
    run_wind_models,
    use_empirical_turbine_powercurve_params,
    )
from parameters.parameter_data import get_input_output_data


class SolarWindOptimizationProblem(WindOptimizationProblem):
    
    def __init__(self,
                 site_info: SiteInfo,
                 system_capacity: float,
                 num_turbines: float = 10,
                 turbine_kw: float = 2e3,
                 ) -> None:
        # set num_turbines to 0 since it's unknown and meant to be determined but could handle differently in the future
        super().__init__(site_info, num_turbines)
        
        self.system_kw: float = system_capacity
        self.turbine_kw: float = turbine_kw
        
        # set up template mask to reuse during shadow calculation
        poly_bounds = self.site_info.polygon.bounds
        site_dim = (int(poly_bounds[2] - poly_bounds[0]), int(poly_bounds[3] - poly_bounds[1]))
        site_mask = np.ones(site_dim)
        
        if os.path.isfile("examples/site_mask_template.npy"):
            self.site_mask_orig = np.load("examples/site_mask_template.npy")
        else:
            
            # def evaluate_row(i:int)->None:
            #     for j in range(site_dim[1]):
            #         site_mask[i, j] = self.site_info.polygon.contains(Point(i, j))
            #
            # evaluations = joblib.Parallel(n_jobs=-1, batch_size='auto', prefer='processes')(
            #     joblib.delayed(evaluate_row)(i) for i in range(site_dim[0]))
            
            for i in range(site_dim[0]):
                for j in range(site_dim[1]):
                    site_mask[i, j] = self.site_info.polygon.contains(Point(i, j))
            self.site_mask_orig = site_mask
            np.save(os.path.abspath(os.path.dirname(__file__)) + "/../site_mask_template", self.site_mask_orig)
        self.site_area = np.count_nonzero(self.site_mask_orig)
        
        # setup scenario to run wind and pvwatts (simple solar model)
        def run_pvwatts(systems):
            pvmodel = systems['Pvwatts']['Pvwattsv5']
            pvmodel.execute()
        
        self.systems = {'Wind': run_wind_models, 'Pvwatts': run_pvwatts}
        wind_defs = wind_windsingleowner
        wind_defs['Resource'].pop('wind_resource_filename')
        wind_defs['Resource']['wind_resource_data'] = self.site_info.wind_resource.data
        self.defaults = {
            'Wind':    {'Windpower': wind_windsingleowner},
            'Pvwatts': {'Pvwattsv5': "PVWattsSingleOwner"}
            }
        self.input_data, self.output_data = get_input_output_data(self.systems)
        
    def make_scenario(self) -> Scenario:
        """
        Unfortunately the Scenario object can not be pickled (even with dill) due to the SAM objects stored inside it.
        So, in order to support parallel evaluations, a Scenario is constructed for each objective function call.
        An optimization that reduces this to one construction per thread is possible but requires deeper integration
        with the parallel execution environment.
        """
        scenario = Scenario(self.defaults, self.input_data, self.output_data, self.systems)

        # initialize some settings of the Pvwatts model which are not default
        model = scenario.systems['Pvwatts']['Pvwattsv5']
        model.SystemDesign.array_type = 2
        model.SystemDesign.tilt = 0
        model.SystemDesign.module_type = 1
        model.LocationAndResource.solar_resource_data = self.site_info.solar_resource.data

        use_empirical_turbine_powercurve_params()
        return scenario
    
    def make_conforming_candidate_and_get_penalty(self, candidate: dict) -> Tuple[dict, float, float]:
        conforming_candidate: dict = candidate.copy()
        
        gcr = conforming_candidate['gcr']
        turb_y = conforming_candidate['turb_y']
        turb_x = conforming_candidate['turb_x']
        # turb_kw = conforming_candidate['turb_kw']
        turb_kw = self.turbine_kw
        
        # make gcr fit in a valid range
        gcr = max(.2, min(1.0, gcr))
        conforming_candidate['gcr'] = gcr
        
        # if the number of turbines exceeds the max capacity, remove some
        windfarm_kw = len(turb_y) * turb_kw
        if windfarm_kw > self.system_kw:
            max_turbines = math.floor(self.system_kw / turb_kw)
            turb_x = turb_x[0:max_turbines]
            turb_y = turb_y[0:max_turbines]
        
        num_turbines: int = int(len(turb_y))
        
        boundary = self.site_info.boundary
        l1_error: float = 0.0
        l2_error: float = 0.0
        for i in range(num_turbines):
            point = Point(turb_x[i], turb_y[i])
            
            distance = self.site_info.polygon.distance(point)
            l1_error += distance
            l2_error += distance * distance
            
            closest_valid_point = boundary.interpolate(boundary.project(point))
            turb_x[i] = closest_valid_point.x
            turb_y[i] = closest_valid_point.y
        
        return conforming_candidate, l1_error, l2_error
    
    def make_conforming_candidate(self, candidate: dict) -> dict:
        return self.make_conforming_candidate_and_get_penalty(candidate)[0]
    
    def objective(self, candidate) -> float:
        """
        :param candidate: dictionary
            'turb_kw': kW rating of single turbine
            'turb_x': x coordinates of turbines
            'turb_y': y coordinates of turbines
            'gcr': between 0 and 0.65 (industry standard)
        :return:
        """
        
        scale: float = .01
        max_unpenalized_distance: float = 0.0
        num_turbines: int = len(candidate['turb_y'])
        
        conforming_candidate, l1_error, l2_error = self.make_conforming_candidate_and_get_penalty(candidate)
        penalty = scale * max(0.0, l1_error - max_unpenalized_distance)
        
        turb_y = conforming_candidate['turb_y']
        turb_x = conforming_candidate['turb_x']
        # turb_kw = conforming_candidate['turb_kw']
        turb_kw = self.turbine_kw
        gcr = conforming_candidate['gcr']
        
        # pv_area = calculate_PV_required_area(pv_kw, gcr)
        # total_area = PolygonArea(self.site_info.vertices)  # 1,170,406 m^2
        shade_mask = get_unshaded_areas_on_site(
            self.site_mask_orig.copy(),
            turb_x,
            turb_y,
            calculate_turbine_radius(turb_kw),
            threshold=.005,
            plot_bool=False)
        unshaded_area = np.count_nonzero(shade_mask)
        max_solar_capacity = calculate_PV_capacity(unshaded_area, gcr)
        
        wind_kw = len(turb_y) * turb_kw
        remaining_capacity = self.system_kw - wind_kw
        pv_capacity = min(remaining_capacity, max_solar_capacity)
        
        # modify scenario with candidate values
        scenario = self.make_scenario()
        pvmodel = scenario.systems['Pvwatts']['Pvwattsv5']
        pvmodel.SystemDesign.gcr = gcr
        pvmodel.SystemDesign.system_capacity = pv_capacity
        windmodel = scenario.systems['Wind']['Windpower']
        windmodel.Farm.wind_farm_xCoordinates = turb_x
        windmodel.Farm.wind_farm_yCoordinates = turb_y
        windmodel.Farm.system_capacity = wind_kw
        
        score = -optimize_solar_wind_AEP(scenario)
        
        return score - penalty
    
    def plot_candidate(self, candidate, color=(0, 1, 0), alpha=.5) -> None:
        num_turbines: int = len(candidate['turb_y'])
        plt.plot(candidate['turb_x'], candidate['turb_y'], 'o', color=color, alpha=alpha)
    
    # --- The original formulation of the objective function: ---
    # def objective(self, candidate):
    #     """
    #
    #     :param candidate: dictionary
    #         'turb_kw': kW rating of single turbine
    #         'turb_x': x coordinates of turbines
    #         'turb_y': y coordinates of turbines
    #         'gcr': between 0 and 0.65 (industry standard)
    #     :return:
    #     """
    #
    #     turb_y = candidate['turb_y']
    #     turb_x = candidate['turb_x']
    #     turb_kw = candidate['turb_kw']
    #     gcr = candidate['gcr']
    #
    #     penalty = 0
    #     scale = .1
    #
    #     max_unpenalized_distance = 0
    #     num_turbines = int(len(turb_y))
    #
    #     border = self.site_info.border
    #     for i in range(num_turbines):
    #         point = Point(turb_x[i], turb_y[i])
    #         distance = self.site_info.polygon.distance(point)
    #         if distance > max_unpenalized_distance:
    #             penalty = penalty + (distance - max_unpenalized_distance)
    #
    #         closest_valid_point = border.interpolate(border.project(point))
    #         turb_x[i] = closest_valid_point.x
    #         turb_y[i] = closest_valid_point.y
    #
    #     penalty = scale * penalty * penalty
    #
    #     # check if enough space for given pv size
    #     windfarm_kw = (len(turb_y) * turb_kw)
    #     pv_kw = self.system_kw - windfarm_kw
    #     if pv_kw < 0:
    #         raise ValueError("too much windpower")
    #
    #     pv_area = calculate_PV_required_area(pv_kw, gcr)
    #     # total_area = PolygonArea(self.site_info.vertices)  # 1,170,406 m^2
    #     shade_mask = get_unshaded_areas_on_site(
    #         self.site_mask_orig.copy(),
    #         turb_x,
    #         turb_y,
    #         calculate_turbine_radius(turb_kw),
    #         threshold=.005,
    #         plot_bool=False)
    #     unshaded_area = np.count_nonzero(shade_mask)
    #     if unshaded_area < pv_area:
    #         # not good as it won't fit, need to apply some penalty
    #         penalty += 234234
    #
    #     # modify scenario with candidate values
    #     pvmodel = self.scenario.systems['Pvwatts']['Pvwattsv5']
    #     pvmodel.SystemDesign.gcr = gcr
    #     pvmodel.SystemDesign.system_capacity = pv_kw
    #     windmodel = self.scenario.systems['Wind']['Windpower']
    #     windmodel.Farm.wind_farm_xCoordinates = turb_x
    #     windmodel.Farm.wind_farm_yCoordinates = turb_y
    #     windmodel.Farm.system_capacity = windfarm_kw
    #
    #     score = -optimize_solar_wind_AEP(self.scenario)
    #
    #     # penalty = 0
    #     # num_turbines = int(candidate.shape[0] / 2)
    #     # for i in range(num_turbines):
    #     #     point = Point(candidate[i], candidate[num_turbines + i])
    #     #     penalty = penalty + self.site_info.polygon.distance(point)
    #     # penalty = 100 * penalty
    #
    #     return score - penalty, score
