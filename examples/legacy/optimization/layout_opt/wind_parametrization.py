from math import *
from typing import (
    Tuple,
    Type,
    )

import numpy as np
from shapely.geometry import Point

from hopp.simulation.technologies.layout.layout_tools import clamp
from hopp.simulation.technologies.layout.wind_layout_tools import (
    get_best_grid,
    get_evenly_spaced_points_along_border,
    )
from examples.optimization.layout_opt.parametrized_optimization_problem import ParametrizedOptimizationProblem
from hopp.tools.optimization import (
    Candidate,
    ProblemParametrization
    )

'''
 aspect: grid column spacing as ratio of max and min spacing, given d and grid_angle => intrarow_spacing,
 interrow_spacing
       -> but how do we know what the max and min spacing is?
       -> min and max aspects may be lines -> could test it on these lines
           -> but, all turbines might not fit in a single line
-> BG paper suggests interrow_spacing = 4*intrarow_spacing when possible, uses fixed number of rows and columns
-> BG paper adjusts interrow_spacing and row_phase_offset to meet requirements set by d, grid_angle,
and intrarow_spacing
'''


class BGMDCandidate(Candidate):
    """
    Parameters to be optimized for WindOptimizationProblemBGMD
    """
    
    def __init__(self,
                 border_ratio: float = 0.0,
                 border_offset: float = 0.0,
                 grid_angle: float = 0.0,
                 grid_aspect: float = 0.0,
                 row_phase_offset: float = 0.0,
                 ) -> None:
        """
        :param border_ratio: ratio of maximum turbines on border -> border spacing (45% suggested)
        :param border_offset: border spacing offset as ratio of border spacing  (0, 1)
        :param grid_angle: inner grid rotation (-pi, pi) [radians]
        :param grid_aspect: grid aspect ratio [cols / rows]
        :param row_phase_offset: inner grid phase offset (0,1) (phase_offset  = row_phase_offset * intrarow_spacing)
                                 (20% suggested)
        """
        super().__init__()
        self.border_ratio: float = border_ratio
        self.border_offset: float = border_offset
        self.grid_angle: float = grid_angle
        self.grid_aspect: float = grid_aspect
        self.row_phase_offset: float = row_phase_offset


class WindParametrization(ProblemParametrization):
    """
    Parametrizes the wind layout optimization problem using the Boundary Grid Method Distributed
    """
    
    def __init__(
            self,
            inner_problem: ParametrizedOptimizationProblem,
            ) -> None:
        """
        Sets up site information and turbine placement constraints from the layout problem

        The site must be a Polygon (i.e. a single polygon)
        :param inner_problem: wind layout optimization problem
        """
        super().__init__(inner_problem, ParametrizedOptimizationProblem, BGMDCandidate)
        self.exterior_length = self.inner_problem.site_info.polygon.exterior.length
        self.max_turbines_on_border = min(self.inner_problem.num_turbines,
                                          int(floor(self.exterior_length / self.inner_problem.min_spacing)))
    
    def get_prior_params(self,
                         distribution_type: Type,
                         ) -> dict:
        """
        Returns the parameters for each parameter's distribution for a given distribution type
        :param distribution_type: str identifier ("Gaussian", "Bernoulli", ...)
        :return: dictionary of parameters
        """
        if distribution_type.__name__ is "Gaussian":
            priors = {
                "border_ratio":     {
                    "mu":    .45,
                    "sigma": .5
                    },
                "border_offset":    {
                    "mu":    0.5,
                    "sigma": 1
                    },
                "grid_angle":       {
                    "mu":    0,
                    "sigma": 2 * pi
                    },
                "grid_aspect":      {
                    "mu":    2,
                    "sigma": 2
                    },
                "row_phase_offset": {
                    "mu":    0.2,
                    "sigma": 0.4
                    }
                }
            return priors
        else:
            raise NotImplementedError
    
    def make_inner_candidate_from_parameters(self,
                                             parameters: BGMDCandidate,
                                             ) -> [Point]:
        """
        Transforms parameters into inner problem candidate (i.e. a set of wind turbine coordinates)

        1. Place boundary turbines
            -> number of turbines placed on the boundary is determined by the wind farm perimeter and ratio of
            border to grid turbines while preserving minimum desired turbine spacing

        2. Place interior turbines using binary search to find interrow_spacing
            -> use largest interrow_spacing that fits target # of turbines

        :param parameters:
        :return: candidate to wind turbine layout problem
        """
        if isinstance(parameters, float):
            x = 0
        max_num_turbines: int = self.inner_problem.num_turbines
        min_spacing: float = self.inner_problem.min_spacing
        # target_num_turbines :int = int(round(max_num_turbines * parameters.ratio_turbines))
        
        bounding_shape = self.inner_problem.site_info.polygon
        exterior = bounding_shape.exterior
        
        num_turbines_on_border = int(round(self.max_turbines_on_border * parameters.border_ratio))
        num_turbines_in_grid = max_num_turbines - num_turbines_on_border
        
        # is there enough spacing to maintain minimum separation?
        
        turbine_locations: [Point] = []
        
        # place border turbines
        if num_turbines_on_border > 0:
            d = self.exterior_length / num_turbines_on_border
            turbine_locations.extend(
                get_evenly_spaced_points_along_border(
                    bounding_shape.exterior,
                    d,
                    parameters.border_offset,
                    max_num_turbines - len(turbine_locations),
                    ))
        
        # place interior grid turbines
        interior_space = bounding_shape.buffer(-self.inner_problem.min_spacing)
        intrarow_spacing, grid_sites = get_best_grid(
            interior_space,
            interior_space.centroid,
            parameters.grid_angle,
            parameters.grid_aspect,
            parameters.row_phase_offset,
            np.sqrt(bounding_shape.area),
            min_spacing,
            max_num_turbines - len(turbine_locations),
            )
        turbine_locations.extend(grid_sites)
        
        inner_candidate = self.inner_problem.candidate_type(turbine_locations)
        
        return 0, inner_candidate
    
    def make_conforming_candidate_and_get_penalty(self,
                                                  candidate: BGMDCandidate
                                                  ) -> Tuple[BGMDCandidate, float, float]:
        """
        Modifies a candidate's parameters so that it falls within range
        :param candidate: optimization candidate
        :return: conforming candidate, parameter error values
        """
        conforming_candidate = BGMDCandidate()
        
        parameter_error: float = 0.0
        
        conforming_candidate.border_ratio, parameter_error = clamp(candidate.border_ratio, parameter_error, 0.0, 1.0)
        conforming_candidate.border_offset, parameter_error = clamp(candidate.border_offset, parameter_error, 0.0, 1.0)
        conforming_candidate.grid_angle = candidate.grid_angle
        conforming_candidate.grid_aspect, parameter_error = \
            clamp(candidate.grid_aspect, parameter_error, 1e-3, log(10))
        conforming_candidate.row_phase_offset, parameter_error = clamp(candidate.row_phase_offset, parameter_error, 0.0,
                                                                       1.0)
        
        return conforming_candidate, parameter_error, parameter_error ** 2
    
    def plot_candidate(self,
                       candidate,
                       *args,
                       **kwargs) -> None:
        self.inner_problem.plot_candidate(self.make_inner_candidate_from_parameters(candidate)[1], *args, **kwargs)
