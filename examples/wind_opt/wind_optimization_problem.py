from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point

from defaults.flatirons_site import (
    wind_windsingleowner,
    )
from examples.optimization_problem import OptimizationProblem
from hybrid.site_info import SiteInfo
from hybrid.scenario import Scenario
from hybrid.wind import opt_tools
from parameters.parameter_data import get_input_output_data


class WindOptimizationProblem(OptimizationProblem):
    
    def __init__(self,
                 site_info: SiteInfo,
                 num_turbines: int = 20,
                 min_spacing: float = 200.0,  # [m]
                 penalty_scale: float = .1,
                 max_unpenalized_distance: float = 0.0,  # [m]
                 ) -> None:
        super().__init__(site_info, num_turbines, min_spacing)
        self.penalty_scale: float = penalty_scale
        self.max_unpenalized_distance: float = max_unpenalized_distance
        
        def run_wind_model(systems):
            windmodel = systems['Wind']['Windpower']
            windmodel.Farm.system_capacity = \
                max(windmodel.Turbine.wind_turbine_powercurve_powerout) * len(windmodel.Farm.wind_farm_xCoordinates)
            windmodel.execute()
        
        systems = {'Wind': run_wind_model}
        defaults = {'Wind': {'Windpower': wind_windsingleowner}}
        # use wind data rather than file name
        resource_defaults = defaults['Wind']['Windpower']['Resource']
        if 'wind_resource_filename' in resource_defaults:
            resource_defaults.pop('wind_resource_filename')
        resource_defaults['wind_resource_data'] = self.site_info.wind_resource.data
        input_data, output_data = get_input_output_data(systems)
        self.scenario = Scenario(defaults, input_data, output_data, systems)
    
    def make_conforming_candidate_and_get_penalty(self, candidate: np.ndarray) -> Tuple[np.ndarray, float, float]:
        num_turbines: int = int(candidate.size / 2)
        
        conforming_candidate: np.ndarray = np.array(candidate)
        # conforming_positions = []
        site_polygon = self.site_info.polygon
        boundary = site_polygon.boundary
        valid_region = site_polygon.buffer(1e-8)
        error: float = 0.0
        squared_error: float = 0.0
        
        # move all turbines inside the site boundary
        for i in range(num_turbines):
            point = Point(candidate[i], candidate[num_turbines + i])
            distance = valid_region.distance(point)
            
            if distance > 0:
                point = boundary.interpolate(boundary.project(point))
                
                error += distance
                squared_error += distance * distance
            
            conforming_candidate[i] = point.x
            conforming_candidate[i + num_turbines] = point.y
        
        return conforming_candidate, error, squared_error
    
    def make_conforming_candidate(self, candidate: np.ndarray) -> np.ndarray:
        return self.make_conforming_candidate_and_get_penalty(candidate)[0]
    
    def objective(self, candidate: np.ndarray) -> float:
        num_turbines = int(candidate.size / 2)
        
        # move turbines that are outside the boundary slightly inside the boundary
        conforming_candidate, error, squared_error = self.make_conforming_candidate_and_get_penalty(candidate)
        penalty = max(0.0, self.penalty_scale * max(0.0, squared_error - self.max_unpenalized_distance))
        score = -opt_tools.optimize_wind_AEP(list(conforming_candidate), self.scenario)
        
        # if penalty > 0:
        #     print('penalty: ', penalty)
        
        return score - penalty  # , score
    
    def plot_candidate(self, candidate, color=(0, 1, 0), alpha=.5) -> None:
        num_turbines: int = int(len(candidate) / 2)
        plt.plot(candidate[0:num_turbines], candidate[num_turbines:], 'o', color=color, alpha=alpha)
