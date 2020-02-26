from enum import (
    IntEnum,
    auto,
    )
from pprint import pprint
from typing import (
    Tuple,
    Optional,
    )

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point

from defaults.defaults_data import (
    wind_windsingleowner,
    )
from examples.wind_opt import turbine_layout_tools
from examples.wind_opt.site_info import (
    SiteInfo,
    LineString,
    )
from examples.wind_opt.wind_optimization_problem import WindOptimizationProblem
from hybrid.scenario import Scenario
from hybrid.wind import opt_tools
from optimization.candidate_converter.candidate_converter import CandidateConverter
from parameters.parameter_data import get_input_output_data
from math import *


class NPArrayBackedObject:
    
    def __init__(
            self,
            array: np.ndarray,
            index: {str, int},
            ) -> None:
        self._array: np.ndarray = array
        self._index: {str, int} = index
    
    def __getattr__(self, key):
        return self._array[self._index[key]]
    
    def __setattr__(self, key, value):
        self._array[self._index[key]] = value
    
    @staticmethod
    def make_index(keys: [str]) -> {str, int}:
        return {key: i for i, key in enumerate(keys)}


# BGM_index = NPArrayBackedObject.make_index(['d', 'offset', 'theta', 'dx', 'dy', 'b'])


class BGMCandidate:
    
    def __init__(
            self,
            d: float = 0.0,  # border spacing [m]
            border_offset: float = 0.0,  # border spacing offset (0, 1)
            theta: float = 0.0,  # inner grid rotation (-pi, pi) [radians]
            dx: float = 0.0,  # grid column spacing [m]
            dy: float = 0.0,  # grid row spacing [m]
            b: float = 0.0,  # (0,1) (phase_offset  = b * dx)
            ) -> None:
        self.d: float = d
        self.border_offset: float = border_offset
        self.theta: float = theta
        self.dx: float = dx
        self.dy: float = dy
        self.b: float = b
    


class WindOptimizationProblemBGM:
    
    def __init__(
            self,
            inner_problem: WindOptimizationProblem,
            min_border_spacing: float = 150,  # [m]
            dx_min: float = 100,
            dx_max: float = 1000,
            dy_min: float = 100,
            dy_max: float = 1000,
            ) -> None:
        self.inner_problem: WindOptimizationProblem = inner_problem
        
        exterior_length = self.inner_problem.site_info.polygon.exterior.length
        self.min_border_spacing: float = max(
            min_border_spacing,
            exterior_length / self.inner_problem.num_turbines)
        self.max_border_spacing = exterior_length
        
        self.dx_min: float = dx_min
        self.dx_max: float = dx_max
        self.dy_min: float = dy_min
        self.dy_max: float = dy_max
    
    def make_inner_candidate_from_parameters(
            self,
            parameters: BGMCandidate,
            ) -> [Point]:
        max_num_turbines: int = self.inner_problem.num_turbines
        bounding_shape = self.inner_problem.site_info.polygon
        exterior = bounding_shape.exterior
        
        sites = []
        if parameters.d * parameters.border_offset < exterior.length:
            # border_spacing = parameters.d * (exterior.length - self.min_border_spacing) + self.min_border_spacing
            border_sites = turbine_layout_tools.get_evenly_spaced_points_along_border(
                exterior, parameters.d, parameters.border_offset, max_num_turbines)
            sites.extend(border_sites)
        
        delta = max_num_turbines - len(sites)
        if delta > 0:
            grid_center = bounding_shape.centroid
            grid_sites = turbine_layout_tools.create_grid(
                bounding_shape,
                grid_center,
                parameters.theta,
                parameters.dx,
                parameters.dy,
                parameters.b)
            
            # sort grid sites by distance to center
            grid_sites.sort(key=lambda site: LineString([site, grid_center]).length)
            sites.extend(grid_sites[0:delta])
        
        num_turbines = len(sites)
        inner_candidate = np.empty(num_turbines * 2)
        for i, p in enumerate(sites):
            inner_candidate[i] = p.x
            inner_candidate[num_turbines + i] = p.y
        
        return inner_candidate
    
    def make_conforming_candidate_and_get_penalty(self, candidate: BGMCandidate) \
            -> Tuple[BGMCandidate, float, float]:
        error: float = 0.0
        squared_error: float = 0.0
        
        conforming_candidate = BGMCandidate()
        conforming_candidate.d = min(self.max_border_spacing, max(self.min_border_spacing, candidate.d))
        conforming_candidate.border_offset = max(0.0, min(1.0, candidate.border_offset))
        conforming_candidate.theta = (candidate.theta + pi) % (2 * pi) - pi
        conforming_candidate.dx = max(self.dx_min, min(self.dx_max, candidate.dx))
        conforming_candidate.dy = max(self.dy_min, min(self.dy_max, candidate.dy))
        conforming_candidate.b = max(0.0, min(1.0, candidate.b))
        
        return conforming_candidate, error, squared_error
    
    def objective(self, parameters: BGMCandidate) -> float:
        
        inner_candidate = self.make_inner_candidate_from_parameters(parameters)
        return self.inner_problem.objective(inner_candidate)
    
    def plot_candidate(self, candidate, *args, **kwargs) -> None:
        self.inner_problem.plot_candidate(self.make_inner_candidate_from_parameters(candidate), *args, **kwargs)
