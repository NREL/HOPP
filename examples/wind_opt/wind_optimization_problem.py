from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import Point

from examples.wind_opt.site_info import SiteInfo
from defaults.defaults_data import (
    wind_windsingleowner,
    )
from hybrid.scenario import Scenario
from hybrid.wind import opt_tools
from parameters.parameter_data import get_input_output_data


class WindOptimizationProblem:
    
    def __init__(self,
                 site_info: SiteInfo,
                 num_turbines: int,
                 ) -> None:
        self.site_info: SiteInfo = site_info
        self.num_turbines: int = num_turbines
    
    def make_conforming_candidate_and_get_penalty(self, candidate: np.ndarray) -> Tuple[np.ndarray, float, float]:
        num_turbines: int = int(len(candidate) / 2)
        conforming_candidate: np.ndarray = np.array(candidate)
        boundary = self.site_info.boundary
        l1_error: float = 0.0
        l2_error: float = 0.0
        for i in range(num_turbines):
            point = Point(candidate[i], candidate[num_turbines + i])
            
            distance = self.site_info.polygon.distance(point)
            l1_error += distance
            l2_error += distance * distance
            
            closest_valid_point = boundary.interpolate(boundary.project(point))
            conforming_candidate[i] = closest_valid_point.x
            conforming_candidate[num_turbines + i] = closest_valid_point.y
        
        return (conforming_candidate, l1_error, l2_error)
    
    def make_conforming_candidate(self, candidate: np.ndarray) -> np.ndarray:
        return self.make_conforming_candidate_and_get_penalty(candidate)[0]
    
    def objective(self, candidate) -> float:
        scale = .01
        
        num_turbines = int(len(candidate) / 2)
        max_unpenalized_distance = 200 * num_turbines
        
        # move turbines that are outside the boundary slightly inside the boundary
        conforming_candidate, l1_error, l2_error = self.make_conforming_candidate_and_get_penalty(candidate)
        penalty = scale * max(0, l1_error - max_unpenalized_distance)
        
        def run_wind_model(systems):
            windmodel = systems['Wind']['Windpower']
            windmodel.Farm.system_capacity = max(windmodel.Turbine.wind_turbine_powercurve_powerout) \
                                             * len(windmodel.Farm.wind_farm_xCoordinates)
            windmodel.execute()
        
        systems = {'Wind': run_wind_model}
        defaults = {'Wind': {'Windpower': wind_windsingleowner}}
        input_data, output_data = get_input_output_data(systems)
        scenario = Scenario(defaults, input_data, output_data, systems)
        
        score = -opt_tools.optimize_wind_AEP(conforming_candidate, scenario)
        
        # if penalty > 0:
        #     print('penalty: ', penalty)
        
        return score - penalty  # , score
    
    def plot_candidate(self, candidate, color=(0, 1, 0), alpha=.5) -> None:
        num_turbines: int = int(len(candidate) / 2)
        plt.plot(candidate[0:num_turbines], candidate[num_turbines:], 'o', color=color, alpha=alpha)
