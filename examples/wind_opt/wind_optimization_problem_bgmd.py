from enum import (
    IntEnum,
    auto,
    )
import pprint
from typing import (
    Tuple,
    Callable,
    Type
    )

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point

from examples.wind_opt import turbine_layout_tools
from hybrid.site_info import (
    SiteInfo,
    LineString,
    )
from examples.optimization_problem import OptimizationProblem
from optimization.optimizer.dimension.dimension_info import DimensionInfo

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
'''
 d: ratio of maximum turbines on border -> border spacing (45% suggested)
 border_offset: border spacing offset as ratio of border spacing  (0, 1)
 theta: inner grid rotation (-pi, pi) [radians]
 b: inner grid phase offset (0,1) (phase_offset  = b * dx) (20% suggested)
 aspect: grid column spacing as ratio of max and min spacing, given d and theta => dx, dy
       -> but how do we know what the max and min spacing is?
       -> min and max aspects may be lines -> could test it on these lines
           -> but, all turbines might not fit in a single line
-> BG paper suggests dy = 4*dx when possible, uses fixed number of rows and columns
-> BG paper adjusts dy and b to meet requirements set by d, theta, and dx

+ could do binary/exp search to meet requirements
+ what about using just theta, b, and dx and then filling the site by setting dy?
    + could use binary search to find dy
        -> use largest dy that fits target # of turbines
        -> eliminate excess turbines from center out? or boundary in?
'''


class BGMDCandidate:
    
    def __init__(
            self,
            # ratio_turbines: int = 0.0,
            border_ratio: float = 0.0,  # ratio of maximum turbines on border -> border spacing (45% suggested)
            border_offset: float = 0.0,  # border spacing offset as ratio of border spacing  (0, 1)
            theta: float = 0.0,  # inner grid rotation (-pi, pi) [radians]
            # dx: float = 0.0,  # grid column spacing [m]
            grid_aspect: float = 0.0,  # grid aspect ratio [cols / rows]
            b: float = 0.0,  # inner grid phase offset (0,1) (phase_offset  = b * dx) (20% suggested)
            ) -> None:
        # self.ratio_turbines: float = ratio_turbines
        self.border_ratio: float = border_ratio
        self.border_offset: float = border_offset
        self.theta: float = theta
        # self.dx: float = dx
        self.grid_aspect: float = grid_aspect
        self.b: float = b
    
    def __repr__(self) -> str:
        return pprint.pformat({
            # 'ratio_turbines':  self.ratio_turbines,
            'border_ratio':  self.border_ratio,
            'border_offset': self.border_offset,
            'theta':         self.theta,
            # 'dx':            self.dx,
            'grid_aspect':   self.grid_aspect,
            'b':             self.b,
            })
    
    def __str__(self) -> str:
        return self.__repr__()


class WindOptimizationProblemBGMD:
    
    def __init__(
            self,
            inner_problem: OptimizationProblem,
            ) -> None:
        self.candidate_type = BGMDCandidate
        self.inner_problem: OptimizationProblem = inner_problem
        self.exterior_length = self.inner_problem.site_info.polygon.exterior.length
        self.max_turbines_on_border = min(self.inner_problem.num_turbines,
                                          int(floor(self.exterior_length / self.inner_problem.min_spacing)))

    def generate_prior(
            self,
            dimension_type: Type,
            callback_one: Callable = lambda x: x,
            callback_two: Callable = lambda x: x,
            callback_three: Callable = lambda x: x
            ) -> BGMDCandidate:
        prior = self.candidate_type()
        prior.border_ratio = dimension_type(callback_one(.45), callback_two(.5), callback_three())
        prior.border_offset = dimension_type(callback_one(.5), callback_two(1), callback_three())
        prior.theta = dimension_type(callback_one(0), callback_two(2 * pi), callback_three())
        prior.grid_aspect = dimension_type(callback_one(0.0), callback_two(2), callback_three())
        prior.b = dimension_type(callback_one(.2), callback_two(.4), callback_three())
        return prior

    def make_inner_candidate_from_parameters(
            self,
            parameters: BGMDCandidate,
            ) -> [Point]:
        max_num_turbines: int = self.inner_problem.num_turbines
        # target_num_turbines :int = int(round(max_num_turbines * parameters.ratio_turbines))
        
        bounding_shape = self.inner_problem.site_info.polygon
        exterior = bounding_shape.exterior
        
        num_turbines_on_border = int(round(self.max_turbines_on_border * parameters.border_ratio))
        num_turbines_in_grid = max_num_turbines - num_turbines_on_border
        
        # is there enough spacing to maintain minimum separation?
        
        border_sites = []
        if num_turbines_on_border > 0:
            d = self.exterior_length / num_turbines_on_border
            border_sites = turbine_layout_tools.get_evenly_spaced_points_along_border(
                exterior, d, parameters.border_offset * d, num_turbines_on_border)
        
        interior_space = bounding_shape.buffer(-self.inner_problem.min_spacing)
        
        # we need to find the largest dx and dy that achieves num_turbines_in_grid
        grid_aspect = exp(parameters.grid_aspect)
        high = min(
            self.inner_problem.site_info.max_distance,
            max(
                self.inner_problem.min_spacing,
                self.inner_problem.site_info.max_distance / grid_aspect))
        
        low = max(
            self.inner_problem.min_spacing,
            self.inner_problem.min_spacing / grid_aspect)
        
        best = None
        for i in range(20):
            mid = ((high - low) / 2.0) + low
            grid_center = bounding_shape.centroid
            
            dx = mid
            dy = mid * grid_aspect
            grid_sites = turbine_layout_tools.create_grid(
                interior_space,
                grid_center,
                parameters.theta,
                dx,
                dy,
                parameters.b,
                2 * num_turbines_in_grid)
            
            # print(parameters.grid_aspect, num_turbines_in_grid, dx, dy, i, mid, len(grid_sites))
            if len(grid_sites) < num_turbines_in_grid:  # spacing is too large
                high = mid
                best = (mid, grid_sites) if best is None else best
            else:  # spacing is too small
                low = mid
                best = (mid, grid_sites)
        
        dx, grid_sites = best
        
        if len(grid_sites) > num_turbines_in_grid:
            grid_sites.sort(key=lambda site: LineString([site, grid_center]).length)
            grid_sites = grid_sites[0:num_turbines_in_grid]
        
        sites = border_sites
        sites.extend(grid_sites)
        
        num_turbines = len(sites)
        # print('num turbines: ', num_turbines)
        # print(str(parameters))
        # print([(p.x, p.y) for p in sites])
        inner_candidate = np.empty(num_turbines * 2)
        for i, p in enumerate(sites):
            inner_candidate[i] = p.x
            inner_candidate[num_turbines + i] = p.y
        
        return inner_candidate
    
    @staticmethod
    def clamp(value, error, minimum, maximum):
        if value > maximum:
            error += value - maximum
            value = maximum
        elif value < minimum:
            error += minimum - value
            value = minimum
        return value, error
    
    def make_conforming_candidate_and_get_penalty(self, candidate: BGMDCandidate) \
            -> Tuple[BGMDCandidate, float, float]:
        conforming_candidate = BGMDCandidate()
        
        parameter_error: float = 0.0
        
        clamp = WindOptimizationProblemBGMD.clamp
        # conforming_candidate.ratio_turbines, parameter_error = clamp(candidate.ratio_turbines, parameter_error,
        # 0.0, 1.0)
        conforming_candidate.border_ratio, parameter_error = clamp(candidate.border_ratio, parameter_error, 0.0, 1.0)
        conforming_candidate.border_offset, parameter_error = clamp(candidate.border_offset, parameter_error, 0.0, 1.0)
        conforming_candidate.theta = candidate.theta
        # conforming_candidate.dx, parameter_error = \
        #     clamp(candidate.dx, parameter_error, self.inner_problem.min_spacing,
        #           self.inner_problem.site_info.max_distance)
        # conforming_candidate.grid_aspect, parameter_error = clamp(candidate.grid_aspect, parameter_error, 0.1, 10.0)
        conforming_candidate.grid_aspect, parameter_error = \
            clamp(candidate.grid_aspect, parameter_error, log(.2), log(10))
        conforming_candidate.b, parameter_error = clamp(candidate.b, parameter_error, 0.0, 1.0)
        
        return conforming_candidate, parameter_error, parameter_error ** 2
    
    def objective(self, parameters: BGMDCandidate) -> float:
        inner_candidate = self.make_inner_candidate_from_parameters(parameters)
        return self.inner_problem.objective(inner_candidate)
    
    def plot_candidate(self, candidate, *args, **kwargs) -> None:
        self.inner_problem.plot_candidate(self.make_inner_candidate_from_parameters(candidate), *args, **kwargs)
