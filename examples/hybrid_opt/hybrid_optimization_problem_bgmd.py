import pprint
from typing import (
    Tuple,
    Callable,
    Type
    )

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
from shapely.ops import split

from examples.wind_opt import turbine_layout_tools
from examples.wind_opt.wind_optimization_problem_bgmd import WindOptimizationProblemBGMD
from examples.hybrid_opt.hybrid_optimization_problem import HybridOptimizationProblem
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


class HybridBGMDCandidate:
    
    def __init__(
            self,
            border_ratio: float = 0.0,  # ratio of maximum turbines on border -> border spacing (45% suggested)
            border_offset: float = 0.0,  # turbine border spacing offset as ratio of border spacing  (0, 1)
            theta: float = 0.0,  # turbine inner grid rotation (-pi, pi) [radians]
            # dx: float = 0.0,  # turbine inner grid column spacing [m]
            grid_aspect: float = 0.0,  # turbine inner grid aspect ratio [cols / rows]
            b: float = 0.0,  # turbine inner grid phase offset (0,1) (phase_offset  = b * dx) (20% suggested)
            vert_offset: float = 0.5,   # height of turbine inner grid and lower border as ratio of layout height (0, 1)
            # panel_dist: float = 0.0,    # min distance between a turbine and a row of solar panels, ratio of dx (0, 1)
            gcr: float = 0.0,  # ground coverage ratio of a set of rows of solar panels (0, 1)
    ) -> None:
        self.border_ratio: float = border_ratio
        self.border_offset: float = border_offset
        self.theta: float = theta
        # self.dx: float = dx
        self.grid_aspect: float = grid_aspect
        self.b: float = b
        self.vert_offset: float = vert_offset
        self.gcr: float = gcr
    
    def __repr__(self) -> str:
        return pprint.pformat({
            # 'ratio_turbines':  self.ratio_turbines,
            'border_ratio':  self.border_ratio,
            'border_offset': self.border_offset,
            'theta':         self.theta,
            # 'dx':            self.dx,
            'grid_aspect':   self.grid_aspect,
            'b':             self.b,
            'vert_offset':   self.vert_offset,
            'gcr':           self.gcr
            })
    
    def __str__(self) -> str:
        return self.__repr__()


class HybridOptimizationProblemBGMD(WindOptimizationProblemBGMD):
    module_power = .321     # kw
    module_width = 0.992    # m
    module_height = 1.488   # m
    road_area_multiplier = 1.2
    """
    Trial 1: Determine the number of turbines that'll fit given the boundary grid parameters, then fit the remaining
    system capacity in solar panels in the remaining area respecting gcr while minimizing distance from turbines
    """
    
    def __init__(
            self,
            inner_problem: HybridOptimizationProblem,
            ) -> None:
        super().__init__(inner_problem)
        self.inner_problem: HybridOptimizationProblem = inner_problem
        self.candidate_type = HybridBGMDCandidate
        self.exterior_length = self.inner_problem.site_info.polygon.exterior.length
        self.max_turbines_on_border = min(self.inner_problem.num_turbines,
                                          int(floor(self.exterior_length / self.inner_problem.min_spacing)))

    def generate_prior(
            self,
            dimension_type: Type,
            callback_one: Callable = lambda x: x,
            callback_two: Callable = lambda x: x,
            callback_three: Callable = lambda x: x
            ) -> HybridBGMDCandidate:
        prior = self.candidate_type()
        prior.border_ratio = dimension_type(callback_one(.45), callback_two(.5), callback_three())
        prior.border_offset = dimension_type(callback_one(.5), callback_two(1), callback_three())
        prior.theta = dimension_type(callback_one(0), callback_two(2 * pi), callback_three())
        prior.grid_aspect = dimension_type(callback_one(0.0), callback_two(2), callback_three())
        prior.b = dimension_type(callback_one(.2), callback_two(.4), callback_three())
        prior.vert_offset = dimension_type(callback_one(.25), callback_two(.2), callback_three())
        prior.gcr = dimension_type(callback_one(.3), callback_two(.2), callback_three())
        return prior

    @staticmethod
    def get_turbine_flicker_loss(diams_from_wind):
        """
        really rough approximation, temporary
        :param diams_from_wind: m
        :return:
        """
        return max(min(0.0575 * np.log(diams_from_wind) + 0.9683, 1.), 0)

    def get_required_solar_area(self, solar_size, gcr):
        n_modules = int(solar_size / self.module_power)
        module_area = n_modules * self.module_height * self.module_width
        return module_area / gcr * self.road_area_multiplier

    def find_solar_area_in_grid(self, start_x, start_y, required_mod_width):
        # TODO: figure this part out to find a section of area in between turbines in the inner grid
        # inside which at least one column of modules can fit (the width of which is the required width)
        col_x = 0
        col_y = 0
        col_height = 0
        col_width = 0
        if col_width > required_mod_width:
            pass
        return col_x, col_y, col_height, col_width

    def make_inner_candidate_from_parameters(
            self,
            parameters: HybridBGMDCandidate,
            ) -> [Point]:
        max_num_turbines: int = self.inner_problem.num_turbines
        # target_num_turbines :int = int(round(max_num_turbines * parameters.ratio_turbines))

        # move boundary according to vert offset
        layout_polygon = self.inner_problem.site_info.polygon
        site_height = (layout_polygon.bounds[3] - layout_polygon.bounds[1]) * parameters.vert_offset
        site_splitter = LineString([(layout_polygon.bounds[0], site_height), (layout_polygon.bounds[2], site_height)])
        result_shapes = split(layout_polygon, site_splitter)

        solar_reserve_indices = [i for i in range(len(result_shapes))
                                 if result_shapes[i].bounds[3] < site_height + 1e-3
                                 and result_shapes[i].area > self.inner_problem.min_spacing]
        solar_reserve = MultiPolygon([result_shapes[i] for i in solar_reserve_indices])
        bounding_shape = MultiPolygon([result_shapes[i] for i in range(len(result_shapes))
                                       if i not in solar_reserve_indices
                                       and result_shapes[i].area > self.inner_problem.min_spacing ** 2])

        # exterior = bounding_shape.exterior
        # for i in solar_reserve:
        #     x, y = i.exterior.xy
        #     plt.plot(x, y, 'b--')
        # for i in bounding_shape:
        #     x, y = i.exterior.xy
        #     plt.plot(x, y, 'g--')
        # plt.show()

        num_turbines_on_border = int(round(self.max_turbines_on_border * parameters.border_ratio))
        num_turbines_in_grid = max_num_turbines - num_turbines_on_border

        # is there enough spacing to maintain minimum separation?

        border_sites = []
        if num_turbines_on_border > 0:
            for poly in bounding_shape:
                d = poly.exterior.length / num_turbines_on_border
                border_sites += turbine_layout_tools.get_evenly_spaced_points_along_border(
                    poly.exterior, d, parameters.border_offset * d, num_turbines_on_border)

        if len(border_sites) != num_turbines_on_border:
            raise ValueError("HybridOptimizationProblemBGMD could not fit requested number of turbines in boundary")

        total_exterior_length = [poly.exterior.length for poly in bounding_shape]
        num_turbines_in_poly = [int(i / sum(total_exterior_length) * num_turbines_in_grid) for i in total_exterior_length]
        num_turbines_in_poly[-1] = num_turbines_in_grid - sum(num_turbines_in_poly[0:-1])

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

        dx = None
        dy = None
        best = None
        sites = border_sites
        for i in range(40):
            inner_sites = []

            mid = ((high - low) / 2.0) + low
            dx = mid
            dy = mid * grid_aspect
            for n, poly in enumerate(bounding_shape):
                interior_space = poly.buffer(-self.inner_problem.min_spacing)
                num_turbs_required = num_turbines_in_poly[n]

                grid_center = poly.centroid

                grid_sites = turbine_layout_tools.create_grid(
                    interior_space,
                    grid_center,
                    parameters.theta,
                    dx,
                    dy,
                    parameters.b,
                    2 * num_turbs_required)
                inner_sites += grid_sites

            if len(inner_sites) > num_turbines_in_grid:
                inner_sites.sort(key=lambda site: LineString([site, grid_center]).length)
                inner_sites = inner_sites[0:num_turbines_in_grid]

            if len(inner_sites) == num_turbines_in_grid:
                sites.extend(inner_sites)
                break

            # print(parameters.grid_aspect, num_turbines_in_grid, dx, dy, i, mid, len(grid_sites))
            if len(inner_sites) < num_turbines_in_grid:  # spacing is too large
                high = mid
                best = (mid, inner_sites) if best is None else best
            else:  # spacing is too small
                low = mid
                best = (mid, inner_sites)

            dx, inner_sites = best

        if len(sites) != self.inner_problem.num_turbines:
            raise ValueError("HybridOptimizationProblemBGMD could not fit requested number of turbines in grid")

        num_turbines = len(sites)
        # print('num turbines: ', num_turbines)
        # print(str(parameters))
        # print([(p.x, p.y) for p in sites])
        inner_candidate = np.empty(num_turbines * 2 + 1)
        for i, p in enumerate(sites):
            inner_candidate[i] = p.x
            inner_candidate[num_turbines + i] = p.y

        # calculate where to fit remaining solar capacity
        gcr = min(max(parameters.gcr, .1), .99)
        required_solar_area = self.get_required_solar_area(self.inner_problem.solar_size_kw, gcr)
        mod_width_with_gcr = self.module_width / np.sqrt(gcr)

        diams_from_turbs = 11
        # spacing between columns of solar panels set by gcr
        reserve_area = 0
        grid_area = 0

        solar_cols_x = []
        solar_cols_y = []
        solar_cols_h = []
        solar_cols_w = []
        solar_cols_n = []

        while diams_from_turbs > 0:
            grid_area = 0
            reserve_area = sum([poly.area for poly in solar_reserve])
            if reserve_area > required_solar_area:
                reserve_area = required_solar_area
                break

            # start trying to fit columns of solar by moving right through site
            start_x = bounding_shape.bounds[0]
            start_y = bounding_shape.bounds[1]
            max_x = bounding_shape.bounds[2]
            while start_x < max_x:
                col_x, col_y, col_height, col_width = self.find_solar_area_in_grid(start_x, start_y, self.module_width)
                # if no viable placement for a new column is found
                if col_x is None or col_x > max_x:
                    break
                start_x = col_x + col_width

                n_cols = int(col_width/self.module_width)
                col_width = 0
                if n_cols == 1:
                    col_width = self.module_width
                # if there's more than one column in each area, make room for gcr consideration
                elif n_cols > 1:
                    n_cols = int(col_width / mod_width_with_gcr)
                    col_width = n_cols * mod_width_with_gcr

                grid_area += col_width * col_height
                # could move the col_x and col_y so that the column is centered, but doesn't make a difference except
                # except in plotting

                solar_cols_x.append(col_x)
                solar_cols_y.append(col_y)
                solar_cols_h.append(col_height)
                solar_cols_w.append(col_width)
                solar_cols_n.append(n_cols)

                if grid_area + reserve_area > required_solar_area:
                    grid_area = required_solar_area - reserve_area
                    break

        if grid_area + reserve_area < required_solar_area:
            raise ValueError("HybridOptimizationProblemBGMD could not fit enough solar")

        gcr_multiplier = self.inner_problem.solar_gcr_loss_multiplier[int(gcr * 100) - 1]
        flicker_multiplier = self.get_turbine_flicker_loss(diams_from_turbs)
        grid_area_single_row = 0
        for i, n in enumerate(solar_cols_n):
            if n == 1:
                grid_area_single_row += solar_cols_w[i] * solar_cols_h[i]

        area_with_gcr_loss = (reserve_area + grid_area_single_row) * gcr_multiplier
        area_with_flicker_loss = (grid_area - grid_area_single_row) * flicker_multiplier
        losses = (area_with_gcr_loss + area_with_flicker_loss) / required_solar_area

        inner_candidate[-1] = losses

        return inner_candidate, (solar_reserve, solar_cols_x, solar_cols_y, solar_cols_w, solar_cols_h)
    
    @staticmethod
    def clamp(value, error, minimum, maximum):
        if value > maximum:
            error += value - maximum
            value = maximum
        elif value < minimum:
            error += minimum - value
            value = minimum
        return value, error
    
    def make_conforming_candidate_and_get_penalty(self, candidate: HybridBGMDCandidate) \
            -> Tuple[HybridBGMDCandidate, float, float]:
        conforming_candidate = HybridBGMDCandidate()

        parameter_error: float = 0.0
        
        clamp = HybridOptimizationProblemBGMD.clamp
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
        conforming_candidate.vert_offset = candidate.vert_offset
        conforming_candidate.gcr = candidate.gcr
        return conforming_candidate, parameter_error, parameter_error ** 2
    
    def objective(self, parameters: HybridBGMDCandidate) -> float:
        """
        Candidate contains turbine coordinates for first 2 * self.num_turbine entries, then pv loss ratio
        :param candidate:
        :return:
        """
        # print("enter obj", parameters)
        inner_candidate = self.make_inner_candidate_from_parameters(parameters)[0]
        # print("fin obj", parameters)
        return self.inner_problem.objective(inner_candidate)
    
    def plot_candidate(self, candidate, *args, **kwargs) -> None:
        # plot turbines
        wind_layout, solar_layout = self.make_inner_candidate_from_parameters(candidate)
        self.inner_problem.plot_candidate(wind_layout, *args, **kwargs)
        # plot solar
        for poly in solar_layout[0]:
            x, y = poly.exterior.xy
            plt.plot(x, y)
        for i in range(len(solar_layout[1])):
            x = solar_layout[1][i]
            y = solar_layout[2][i]
            w = solar_layout[3][i]
            h = solar_layout[4][i]
            col = Polygon(((x, y),
                           (x, y + w),
                           (x + h, y + w),
                           (x + h, w)))
            x, y = col.exterior.xy
            plt.plot(x, y)
