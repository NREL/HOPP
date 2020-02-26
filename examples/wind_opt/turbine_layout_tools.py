import random
from math import *
from typing import Optional

from shapely.affinity import *
from shapely.geometry import *
from shapely.geometry.base import *

import matplotlib.pyplot as plt
import numpy as np


def get_evenly_spaced_points_along_border(
        boundary: BaseGeometry,
        spacing: float,
        offset: float = 0.0,
        max_number: Optional[int] = None,
        ) -> [Point]:
    length = boundary.length
    result = []
    d = 0.0
    while d <= length and (max_number is None or len(result) < max_number):
        result.append(boundary.interpolate(offset + d))
        d += spacing
    # print('shape: ', [c for c in shape.coords])
    # print('boundary: ', [(p.x, p.y) for p in result])
    return result


def create_grid(
        bounding_shape: BaseGeometry,
        center: Point,
        theta: float,
        dx: float,
        dy: float,
        b: float,
        max_sites: int = None,
        ) -> [Point]:
    '''
    :param offset:
    :param theta:
    :param dx:
    :param dy:
    :param b:
    :return:
    '''
    theta = (theta + pi) % (2 * pi) - pi  # reset theta to (-pi, pi)
    site_bounds = bounding_shape.bounds
    bounding_box_line = LineString([(site_bounds[0], site_bounds[1]), (site_bounds[2], site_bounds[3])])
    base_line = LineString([(-bounding_box_line.length, 0), (bounding_box_line.length, 0)])
    line_length = base_line.length
    # center = bounding_box_line.centroid
    base_line = translate(base_line, center.x, center.y)
    base_line = rotate(base_line, theta, use_radians=True)
    row_offset = rotate(Point(0, dy), theta)
    
    grid_positions: [Point] = []
    row_number: int = 0
    phase_offset: float = b * dx
    for row_number in range(ceil((line_length / 2) / dy)):
        def add_points_along_line(row_number):
            # generate lines according to the grid
            line = translate(base_line, row_number * row_offset.x, row_number * row_offset.y)
            
            # generate points along that line with the right phase offset
            start_offset = (phase_offset * row_number) % dx
            x = start_offset
            while x <= line_length:
                pos = line.interpolate(x)
                if bounding_shape.contains(pos):
                    grid_positions.append(pos)
                x += dx

        if max_sites and len(grid_positions) > max_sites:
            break
            
        add_points_along_line(row_number)
        if row_number > 0:
            add_points_along_line(-row_number)
    
    # sorted(grid_positions, lambda pos: LineString(pos, center).length)
    return grid_positions


"""
The number of turbines placed on the boundary is determined by the wind farm perimeter and turbine rotor
diameter. If the perimeter is large enough, 45% of the wind turbines are placed on the boundary. In some
cases, the wind farm perimeter is small, and would result in turbines that are too closely spaced if 45%
were placed around the boundary. In this case, the number of boundary turbines is reduced until the minimum
desired turbine spacing in the wind farm is preserved. No matter how many turbines are placed around the
boundary, they are always spaced equally traversing the perimeter, and all of the remaining turbines are
placed in the inner grid.

The number of rows, columns, and their organization in the grid is determined with
the following procedure. First, dy is set to be four times dx, b is set such that turbines are offset
twenty degrees from those in adjacent rows, and θ is initialized randomly. Then, dx is varied with θ
remaining constant, and dy and b changing to fulfill the requirements prescribed in the initialization
definition, until the correct number of turbines are within the wind farm boundary. During optimization,
each of the grid variables can change individually, however the discrete values remain fixed.


Options for making feasible solutions with BGM and DFO:
    + penalize turbines out of bounds
        - does not prevent infeasible solutions
        - requires tuning
        + provides a smooth surface to descend into a good solution
    + ignore turbines out of bounds
        - discontinuities in parameter space
        + always generates a feasible solution
    + both ignore turbines and penalize
        - may not give adequate support to some configurations
"""

#
#
# def create_from_boundary_grid(num_turbines):
#     rotor_diameter = 50
#     site_shape = self.polygon
#     boundary = site_shape.exterior
#     site_bounds = site_shape.bounds
#
#     minimum_spacing = 3 * rotor_diameter  # 3-5x RD is recommended
#     max_boundary_allocation = .45
#     exterior_length = boundary.length
#     num_exterior_turbines = \
#         floor(min(
#             max_boundary_allocation * num_turbines,
#             exterior_length / minimum_spacing))
#     boundary_spacing = exterior_length / num_exterior_turbines
#     boundary_positions = self.get_evenly_spaced_points_along_border(boundary, boundary_spacing)
#
#     interior_turbines = num_turbines - num_exterior_turbines
#
#     theta = pi * (2 * random.random() - 1)
#
#     dx = 3 * rotor_diameter
#     dy = 4 * dx
#     b = 20 / 360
#
#     # (minx, miny, maxx, maxy)
#     # bounding_box_line = LineString([(site_bounds[0], site_bounds[1]), (site_bounds[2], site_bounds[3])])
#     # base_line = LineString([(-bounding_box_line.length, 0), (bounding_box_line.length, 0)])
#     # line_length = base_line.length
#     # center = site_shape.centroid
#     # base_line = affinity.translate(base_line, center.x, center.y)
#     # base_line = affinity.rotate(base_line, theta, use_radians=True)
#     # row_offset = affinity.rotate(Point(0, dy), theta)
#     #
#     grid_positions = self.create_grid(site_shape, site_shape.centroid, theta, dx, dy, b)
#
#     positions = grid_positions.copy()
#     positions.extend(boundary_positions)
#     return positions
