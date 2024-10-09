from __future__ import annotations
from math import *
from typing import Type

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import (
    MultiPolygon,
    Point,
    Polygon,
    )
from shapely.geometry.base import BaseGeometry

from hopp.utilities.log import opt_logger as logger
from hopp.simulation.technologies.layout.layout_tools import (
    clamp,
    make_polygon_from_bounds,
    )
from hopp.simulation.technologies.layout.wind_layout_tools import (
    get_best_grid,
    get_evenly_spaced_points_along_border,
    subtract_turbine_exclusion_zone
    )

from hopp.tools.optimization import (
    Candidate,
    ProblemParametrization
    )

from hopp.simulation.technologies.layout.pv_layout_tools import find_best_solar_size
from examples.optimization.layout_opt.hybrid_optimization_problem import (
    HybridOptimizationProblem,
    HybridSimulationVariables,
    )
from hopp.simulation.technologies.layout.plot_tools import plot_shape


class HybridCandidate(Candidate):
    """
    Parameter inputs to be optimized for a HybridOptimizationProblem
    """
    
    def __init__(self,
                 border_spacing: float = 0.0,
                 border_offset: float = 0.0,
                 grid_angle: float = 0.0,
                 grid_aspect_power: float = 0.0,
                 row_phase_offset: float = 0.0,
                 solar_x_position: float = 0.0,
                 solar_y_position: float = 0.0,
                 solar_aspect_power: float = 0.0,
                 solar_gcr: float = 0.0,
                 solar_s_buffer: float = 0.0,
                 solar_x_buffer: float = 0.0,
                 ) -> None:
        """
        :param border_spacing: spacing along border = (1 + border_spacing) * min spacing
        :param border_offset: turbine border spacing offset as ratio of border spacing  (0, 1)
        :param grid_angle: turbine inner grid rotation (0, pi) [radians]
        :param grid_aspect_power: grid aspect ratio [cols / rows] = 2^grid_aspect_power
        :param row_phase_offset: inner grid phase offset (0,1)  (20% suggested)
        :param solar_x_position: ratio of solar's x coords to site width (0, 1)
        :param solar_y_position: ratio of solar's y coords to site height (0, 1)
        :param solar_aspect_power: aspect ratio of solar to site width = 2^solar_aspect_power
        :param solar_gcr: gcr ratio of solar patch
        :param solar_s_buffer: south side buffer ratio (0, 1)
        :param solar_x_buffer: east and west side buffer ratio (0, 1)
        """
        super().__init__()
        self.border_spacing: float = border_spacing
        self.border_offset: float = border_offset
        self.grid_angle: float = grid_angle
        self.grid_aspect_power: float = grid_aspect_power
        self.row_phase_offset: float = row_phase_offset
        self.solar_x_position: float = solar_x_position
        self.solar_y_position: float = solar_y_position
        self.solar_aspect_power: float = solar_aspect_power
        self.solar_gcr: float = solar_gcr
        self.solar_s_buffer: float = solar_s_buffer
        self.solar_x_buffer: float = solar_x_buffer


class HybridParametrization(ProblemParametrization):
    """
    Parametrizes the Hybrid (solar plus wind) layout optimization problem using the Boundary Grid Method
    """
    
    def __init__(self,
                 inner_problem: HybridOptimizationProblem,
                 ) -> None:
        """
        The site must be a Polygon (i.e. a single polygon)
        :param inner_problem: wind layout optimization problem
        """
        super().__init__(inner_problem, HybridOptimizationProblem, HybridCandidate)
        logger.info("Created HybridOptimizationProblemBGRM")
    
    def get_losses(self,
                   candidate: HybridCandidate,
                   ):
        """
        Returns inner problem performance of parametrized candidate
        :param candidate:
        :return: performance
        """
        conforming_candidate, conformation_penalty, _ = self.make_conforming_candidate_and_get_penalty(candidate)
        penalty, inner_candidate = self.make_inner_candidate_from_parameters(conforming_candidate)
        score, evaluation, wind_score, solar_score, wake_losses, gcr_losses, flicker_losses = \
            self.inner_problem.compute_objective(inner_candidate[0])
        score -= penalty
        score -= conformation_penalty
        return score, evaluation, wind_score, solar_score, wake_losses, gcr_losses, flicker_losses

    def get_prior_params(self,
                         distribution_type: Type
                         ) -> dict:
        """
        Returns the parameters for each parameter's distribution for a given distribution type
        :param distribution_type: str identifier ("Gaussian", "Bernoulli", ...)
        :return: dictionary of parameters
        """

        if distribution_type.__name__ == "Gaussian":
            priors = {
                "border_spacing":     {
                    "mu":    5,
                    "sigma": 5
                    },
                "border_offset":      {
                    "mu":    0.5,
                    "sigma": 2
                    },
                "grid_angle":         {
                    "mu":    pi / 2,
                    "sigma": pi,
                    },
                "grid_aspect_power":  {
                    "mu":    0,
                    "sigma": 3
                    },
                "row_phase_offset":   {
                    "mu":    0.5,
                    "sigma": .5
                    },
                "solar_x_position":   {
                    "mu":    .5,
                    "sigma": .5
                    },
                "solar_y_position":   {
                    "mu":    .5,
                    "sigma": .5
                    },
                "solar_aspect_power": {
                    "mu":    0,
                    "sigma": 3
                    },
                "solar_gcr":          {
                    "mu":    .5,
                    "sigma": .5
                    },
                "solar_s_buffer":     {
                    "mu":    4,
                    "sigma": 4
                    },
                "solar_x_buffer":     {
                    "mu":    4,
                    "sigma": 4
                    }
                }
            return priors
        else:
            raise NotImplementedError
    
    def make_inner_candidate_from_parameters(
            self,
            parameters: HybridCandidate,
            ) -> tuple[float, tuple[HybridSimulationVariables, Polygon, BaseGeometry]]:
        """
        Transforms parameters into inner problem candidate (i.e. a set of wind turbine coordinates)

        1. Place the section of solar panels
            -> height and width; x and y position; east, west and south buffers

        2. Place turbines according to Wind BGMD

        :param parameters:
        :return: candidate to hybrid layout problem
        """
        logger.info("Starting inner candidate: {}".format(vars(parameters)))
        
        '''
        - x or y position does not change actual solar placement
            - want to get a solution with x and y as centered as possible
                - get bounds of solar placement and bounds of solar region
                    - compute distance from most centered point

        - larger buffer has no effect because it goes out of bounds
            - want a solution with the smallest buffer possible
            - get bounds of buffer region and buffer in bounds
                - compute distance from smallest buffer possible
        '''
        
        penalty = 0.0
        max_num_turbines: int = self.inner_problem.num_turbines
        
        site_shape = self.inner_problem.site_info.polygon
        min_spacing = self.inner_problem.min_spacing
        
        # place solar area
        site_sw_bound = np.array([site_shape.bounds[0], site_shape.bounds[1]])
        site_ne_bound = np.array([site_shape.bounds[2], site_shape.bounds[3]])
        site_bounds_size = site_ne_bound - site_sw_bound
        
        solar_center = site_sw_bound + site_bounds_size * \
                       np.array([parameters.solar_x_position, parameters.solar_y_position])
        
        # place solar
        max_solar_width = self.inner_problem.module_width * self.inner_problem.max_num_modules \
                          / self.inner_problem.min_strand_length
        
        solar_aspect = np.exp(parameters.solar_aspect_power)
        solar_x_size, num_modules, strands, solar_region, solar_bounds = \
            find_best_solar_size(
                self.inner_problem.max_num_modules,
                self.inner_problem.min_strand_length,
                site_shape,
                solar_center,
                0.0,
                self.inner_problem.module_width,
                self.inner_problem.module_height,
                parameters.solar_gcr,
                solar_aspect,
                self.inner_problem.module_width,
                max_solar_width,
                )
        
        solar_x_buffer_length = min_spacing * (1 + parameters.solar_x_buffer)
        solar_s_buffer_length = min_spacing * (1 + parameters.solar_s_buffer)
        solar_buffer_shape = make_polygon_from_bounds(
            solar_bounds[0] - np.array([solar_x_buffer_length, solar_s_buffer_length]),
            solar_bounds[1] + np.array([solar_x_buffer_length, 0]))
        
        def get_bounds_center(shape):
            bounds = shape.bounds
            return Point(.5 * (bounds[0] + bounds[2]), .5 * (bounds[1] + bounds[3]))
        
        def get_excess_buffer_penalty(buffer, solar_region, bounding_shape):
            penalty = 0.0
            buffer_intersection = buffer.intersection(bounding_shape)
            
            shape_center = get_bounds_center(buffer)
            intersection_center = get_bounds_center(buffer_intersection)
            
            shape_center_delta = \
                np.abs(np.array(shape_center.coords) - np.array(intersection_center.coords)) / site_bounds_size
            shape_center_penalty = np.sum(shape_center_delta ** 2)
            penalty += shape_center_penalty
            
            bounds = buffer.bounds
            intersection_bounds = buffer_intersection.bounds
            
            west_excess = intersection_bounds[0] - bounds[0]
            south_excess = intersection_bounds[1] - bounds[1]
            east_excess = bounds[2] - intersection_bounds[2]
            north_excess = bounds[3] - intersection_bounds[3]
            
            solar_bounds = solar_region.bounds
            actual_aspect = (solar_bounds[3] - solar_bounds[1]) / \
                            (solar_bounds[2] - solar_bounds[0])
            
            aspect_error = fabs(np.log(actual_aspect) - np.log(solar_aspect))
            penalty += aspect_error ** 2
            
            # excess buffer, minus minimum size
            # excess buffer is how much extra there is, but we must not penalise minimum sizes
            #
            # excess_x_buffer = max(0.0, es - min_spacing)
            # excess_y_buffer = max(0.0, min(ee, ew) - min_spacing)
            
            # if buffer has excess, then we need to penalize any excess buffer length beyond the minimum
            
            minimum_s_buffer = max(solar_s_buffer_length - south_excess, min_spacing)
            excess_x_buffer = (solar_s_buffer_length - minimum_s_buffer) / min_spacing
            penalty += excess_x_buffer ** 2
            
            minimum_w_buffer = max(solar_x_buffer_length - west_excess, min_spacing)
            minimum_e_buffer = max(solar_x_buffer_length - east_excess, min_spacing)
            excess_y_buffer = (solar_x_buffer_length - max(minimum_w_buffer, minimum_e_buffer)) / min_spacing
            penalty += excess_y_buffer ** 2

            return penalty
        
        penalty += get_excess_buffer_penalty(solar_buffer_shape, solar_region, site_shape)
        
        solar_buffer_region = site_shape.intersection(solar_buffer_shape)
        wind_shape = site_shape.difference(solar_buffer_shape)  # compute valid wind layout shape
        
        # place border turbines
        turbine_positions: list[Point] = []
        if not isinstance(wind_shape, MultiPolygon):
            wind_shape = MultiPolygon([wind_shape, ])
        
        border_spacing = (parameters.border_spacing + 1) * min_spacing
        for bounding_shape in wind_shape.geoms:
            turbine_positions.extend(
                get_evenly_spaced_points_along_border(
                    bounding_shape.exterior,
                    border_spacing,
                    parameters.border_offset,
                    max_num_turbines - len(turbine_positions),
                    ))
        
        valid_wind_shape = subtract_turbine_exclusion_zone(min_spacing, wind_shape, turbine_positions)
        
        # place interior grid turbines
        max_num_interior_turbines = max_num_turbines - len(turbine_positions)
        grid_aspect = np.exp(parameters.grid_aspect_power)
        intrarow_spacing, grid_sites = get_best_grid(
            valid_wind_shape,
            wind_shape.centroid,
            parameters.grid_angle,
            grid_aspect,
            parameters.row_phase_offset,
            min_spacing * 10000,
            min_spacing,
            max_num_interior_turbines,
            )
        turbine_positions.extend(grid_sites)
        
        inner_candidate = self.inner_problem.candidate_type(turbine_positions,
                                                            ((parameters.solar_gcr, num_modules, strands),))

        return penalty, (inner_candidate, solar_buffer_shape, solar_region)
    
    def make_conforming_candidate_and_get_penalty(self,
                                                  candidate: HybridCandidate
                                                  ) -> tuple[HybridCandidate, float, float]:
        """
        Modifies a candidate's parameters so that it falls within range
        :param candidate: optimization candidate
        :return: conforming candidate, parameter error values
        """
        conforming_candidate = HybridCandidate()
        
        parameter_error: float = 0.0
        
        conforming_candidate.border_spacing, parameter_error = \
            clamp(candidate.border_spacing,
                  parameter_error,
                  0.0,
                  100)
        conforming_candidate.border_offset, parameter_error = \
            clamp(candidate.border_offset, parameter_error, 0.0, 1.0)
        conforming_candidate.grid_angle, parameter_error = \
            clamp(candidate.grid_angle, parameter_error, 0, pi)
        conforming_candidate.grid_aspect_power, parameter_error = \
            clamp(candidate.grid_aspect_power, parameter_error, -4, 4)
        conforming_candidate.row_phase_offset, parameter_error = \
            clamp(candidate.row_phase_offset, parameter_error, 0.0, 1.0)
        conforming_candidate.solar_x_position, parameter_error = \
            clamp(candidate.solar_x_position, parameter_error, 0.0, 1.0)
        conforming_candidate.solar_y_position, parameter_error = \
            clamp(candidate.solar_y_position, parameter_error, 0.0, 1.0)
        conforming_candidate.solar_gcr, parameter_error = \
            clamp(candidate.solar_gcr, parameter_error, .2, .9)
        conforming_candidate.solar_aspect_power, parameter_error = \
            clamp(candidate.solar_aspect_power, parameter_error, -4, 4)
        conforming_candidate.solar_s_buffer, parameter_error = \
            clamp(candidate.solar_s_buffer, parameter_error, 0.0, 9.0)
        conforming_candidate.solar_x_buffer, parameter_error = \
            clamp(candidate.solar_x_buffer, parameter_error, 0.0, 9.0)
        
        return conforming_candidate, parameter_error, parameter_error ** 2
    
    def plot_candidate(
            self,
            parameters: HybridCandidate,
            figure,
            axes,
            turbine_color,
            solar_color,
            border_color=(0, 0, 0),
            alpha=1.0,
            linewidth=4.0,
            ) -> None:
        penalty, (inner_candidate, solar_buffer_shape, solar_bounds_shape) = \
            self.make_inner_candidate_from_parameters(parameters)
        
        site_shape = self.inner_problem.site_info.polygon
        plot_shape(figure, axes, site_shape, '--', color=border_color, alpha=.95, linewidth=linewidth / 2)
        
        turb_pos_x = inner_candidate.turb_pos_x
        turb_pos_y = inner_candidate.turb_pos_y
        
        for n in range(len(turb_pos_x)):
            x, y = turb_pos_x[n], turb_pos_y[n]
            circle = plt.Circle(
                (x, y),
                linewidth * 10,
                color=turbine_color,
                alpha=alpha,
                fill=True,
                linewidth=linewidth,
                )
            axes.add_artist(circle)
            # axes.plot(x, y, '+', color=turbine_color, alpha=alpha, linewidth=linewidth)
            # circle = plt.Circle(
            #     (x, y),
            #     self.inner_problem.min_spacing,
            #     color=turbine_color,
            #     alpha=alpha,
            #     fill=False,
            #     linewidth=linewidth,
            #     )
            # axes.add_artist(circle)
        
        # plot_turbines(inner_candidate.turb_pos_x, inner_candidate.turb_pos_y, turbine_color, 1.0)
        # plot_solar_strands(figure, axes, inner_candidate.solar_areas, '-', color=solar_color, alpha=alpha)
        
        plot_shape(figure, axes, solar_bounds_shape, '-', color=solar_color, alpha=alpha)
        plot_shape(figure, axes, solar_buffer_shape, '--', color=solar_color, alpha=alpha)
        
        # plot_shape(solar_s_buffer_shape, '--', **kwargs)
        # plot_shape(solar_shape, '-', **kwargs)
