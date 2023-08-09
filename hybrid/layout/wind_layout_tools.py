import numpy as np
from typing import Optional

from shapely.affinity import rotate, translate
from shapely.geometry import Point, LineString, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep
from shapely.ops import unary_union

from hybrid.layout.layout_tools import binary_search_float


def get_evenly_spaced_points_along_border(boundary: BaseGeometry,
                                          spacing: float,
                                          offset: float = 0.0,
                                          max_number: Optional[int] = None,
                                          ) -> [Point]:
    """
    Spaced equally traversing the perimeter
    :param boundary: a boundary line
    :param spacing: distance between points
    :param offset: shifting where to start placing points
    :param max_number: max points
    :return: list of Points
    """
    length = boundary.length - spacing
    result = []
    d = 0.0
    starting_pt = offset * boundary.length
    while d <= length and (max_number is None or len(result) < max_number):
        result.append(boundary.interpolate(starting_pt + d))
        d += spacing
    # print('site_shape: ', [c for c in site_shape.coords])
    # print('boundary: ', [(p.x, p.y) for p in result])
    return result


def make_grid_lines(site_shape: BaseGeometry,
                    center: Point,
                    grid_angle: float,
                    interrow_spacing: float
                    ) -> [LineString]:
    """
    Place parallel lines inside a site
    :param site_shape: Polygon
    :param center: where to center the grid
    :param grid_angle: in degrees where 0 is east
    :param interrow_spacing: distance between lines
    :return: list of lines
    """
    if site_shape.is_empty:
        return []
    
    grid_angle = (grid_angle + np.pi) % (2 * np.pi) - np.pi  # reset grid_angle to (-pi, pi)
    bounds = site_shape.bounds
    
    bounding_box_line = LineString([(bounds[0], bounds[1]), (bounds[2], bounds[3])])
    base_line = LineString([(-bounding_box_line.length, 0), (bounding_box_line.length, 0)])
    line_length = base_line.length
    
    base_line = rotate(base_line, -grid_angle, use_radians=True)
    base_line = translate(base_line, center.x, center.y)
    
    row_offset = Point(
        interrow_spacing * np.cos(-grid_angle + np.pi / 2),
        interrow_spacing * np.sin(-grid_angle + np.pi / 2))
    
    grid_lines: [LineString] = []
    num_rows_per_side: int = int(np.ceil((line_length / 2) / interrow_spacing) + 1)
    for row_number in range(-num_rows_per_side, num_rows_per_side + 1):
        line = translate(base_line, row_number * row_offset.x, row_number * row_offset.y)
        grid_lines.append(line)
    
    return grid_lines


def create_grid(site_shape: BaseGeometry,
                center: Point,
                grid_angle: float,
                intrarow_spacing: float,
                interrow_spacing: float,
                row_phase_offset: float,
                max_sites: int = None,
                ) -> [Point]:
    """
    Get a list of coordinates placed along a grid inside a site boundary
    :param site_shape: Polygon
    :param center: where to center the grid
    :param grid_angle: in degrees where 0 is north, increasing clockwise
    :param intrarow_spacing: distance between turbines along same row
    :param interrow_spacing: distance between rows
    :param row_phase_offset: offset of turbines along row from one row to the next
    :param max_sites: max number of turbines
    :return: list of coordinates
    """
    grid_lines: [LineString] = make_grid_lines(
        site_shape,
        center,
        grid_angle,
        interrow_spacing
        )
    phase_offset: float = row_phase_offset * intrarow_spacing
    
    prepared_site = prep(site_shape)
    grid_positions: [Point] = []
    for row_number, grid_line in enumerate(grid_lines):
        length = grid_line.length
        
        # generate points along that line with the right phase offset
        x: float = (phase_offset * row_number) % intrarow_spacing
        while x <= length:
            position = grid_line.interpolate(x)
            if prepared_site.contains(position):
                if max_sites and len(grid_positions) >= max_sites:
                    break
                grid_positions.append(position)
            x += intrarow_spacing
        if max_sites and len(grid_positions) >= max_sites:
            break
    
    return grid_positions


def get_best_grid(site_shape: BaseGeometry,
                  center: Point,
                  grid_angle: float,
                  grid_aspect: float,
                  row_phase_offset: float,
                  max_spacing: float,
                  min_spacing: float,
                  max_sites: int,
                  ) -> (float, [Point]):
    """
    Finds the least dense grid layout that fits max_sites into it, and if that isn't possible it finds the grid that
    fits the most turbines into the site_shape.
    
    Respects min_spacing and max_spacing limits.
    :param site_shape: Polygon
    :param center: where to center the grid
    :param grid_angle: in degrees where 0 is north, increasing clockwise
    :param grid_aspect: ratio [cols / rows]
    :param row_phase_offset: offset of turbines along row from one row to the next
    :param max_spacing: max spacing between turbines
    :param min_spacing: min spacing
    :param max_sites: max number of turbines
    :return intrarow spacing and list of grid coordinates
    """
    best: (int, float, [Point]) = (0, max_spacing, [])
    
    if max_sites > 0:
        prepared_site = prep(site_shape)
        
        def grid_objective(intrarow_spacing: float) -> float:
            nonlocal best
            interrow_spacing = intrarow_spacing * grid_aspect
            grid_sites = create_grid(
                # site_shape.buffer(-intrarow_spacing),
                site_shape,
                center,
                grid_angle,
                intrarow_spacing,
                interrow_spacing,
                row_phase_offset,
                max_sites)
            num_sites = len(grid_sites)
            
            delta_sites = num_sites - best[0]
            if delta_sites > 0 or delta_sites == 0 and intrarow_spacing > best[1]:
                best = (num_sites, intrarow_spacing, grid_sites)
            
            if num_sites < max_sites:
                return 1  # less than the max: decrease spacing
            return -1  # greater than or equal to the max: increase spacing
        
        if site_shape.area > np.pi * (min_spacing ** 2):
            maximum_chord = max_distance(site_shape)
            
            max_intrarow_spacing = min(max_spacing, max_spacing * grid_aspect, maximum_chord)
            min_intrarow_spacing = min(max(min_spacing, min_spacing * grid_aspect), max_intrarow_spacing)
            
            interrow_offset, _ = binary_search_float(
                grid_objective,
                min_intrarow_spacing,
                max_intrarow_spacing,
                max_iters=64,
                threshold=1e-1)
    return best[1], best[2]


def max_distance(site_shape: BaseGeometry) -> float:
    """
    :param site_shape:
    :return: an upper bound on the maximum distance any two points in the site_shape could be from each other
    """
    if site_shape.area < 1e-3:
        return 0.0
    bounds = site_shape.bounds
    return Point(bounds[0], bounds[1]).distance(Point(bounds[2], bounds[3]))


def move_turbines_within_boundary(turb_pos_x: list,
                                  turb_pos_y: list,
                                  boundary: Polygon,
                                  valid_region: Polygon
                                  ) -> (np.ndarray, float):
    """
    :param turb_pos_x: list of x coordinates
    :param turb_pos_y: list of y coordinates
    :param boundary: site boundary
    :param valid_region: region to move turbines into
    :return: adjusted x and y coordinates
    """
    squared_error: float = 0.0
    num_turbines = len(turb_pos_x)
    
    for i in range(num_turbines):
        point = Point(turb_pos_x[i], turb_pos_y[i])
        distance = valid_region.distance(point)
        
        if distance > 0:
            point = boundary.interpolate(boundary.project(point))
            
            squared_error += distance ** 2
        
        turb_pos_x[i] = point.x
        turb_pos_y[i] = point.y
    
    return turb_pos_x, turb_pos_y, squared_error


def subtract_turbine_exclusion_zone(min_spacing: float,
                                    source_shape: BaseGeometry,
                                    turbine_positions: [Point],
                                    ) -> BaseGeometry:
    """
    Subtract the min spacing around each turbine from a site polygon
    :param min_spacing: minimum distance around turbine
    :param source_shape: site polygon
    :param turbine_positions: Points of the turbines within the source_shape
    :return: modified shape with the circles around the turbines removed
    """
    if len(turbine_positions) <= 0:
        return source_shape
    return source_shape.difference(
        unary_union([turbine.buffer(min_spacing) for turbine in turbine_positions]))

"""
The number of turbines placed on the boundary is determined by the wind farm perimeter and turbine rotor
diameter. If the perimeter is large enough, 45% of the wind turbines are placed on the boundary. In some
cases, the wind farm perimeter is small, and would result in turbines that are too closely spaced if 45%
were placed around the boundary. In this case, the number of boundary turbines is reduced until the minimum
desired turbine spacing in the wind farm is preserved. No matter how many turbines are placed around the
boundary, they are always spaced equally traversing the perimeter, and all of the remaining turbines are
placed in the inner grid.

The number of rows, columns, and their organization in the grid is determined with
the following procedure. First, interrow_spacing is set to be four times intrarow_spacing, row_phase_offset is set
such that turbines are offset
twenty degrees from those in adjacent rows, and θ is initialized randomly. Then, intrarow_spacing is varied with θ
remaining constant, and interrow_spacing and row_phase_offset changing to fulfill the requirements prescribed in the
initialization
definition, until the correct number of turbines are within the wind farm boundary. During optimization,
each of the grid variables can change individually, however the discrete values remain fixed.

"""
