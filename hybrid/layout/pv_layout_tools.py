from typing import List
import warnings
from math import floor
from shapely.geometry import MultiLineString, GeometryCollection, MultiPoint

import PySAM.Pvwattsv8 as pvwatts
import PySAM.Windpower as windpower
from shapely.prepared import (
    PreparedGeometry,
    )

from hybrid.layout.layout_tools import *
from hybrid.sites import SiteInfo
from hybrid.layout.wind_layout_tools import *


def find_best_gcr(
        max_num_modules: int,
        min_strand_length: int,
        site_shape: BaseGeometry,
        center: Point,
        phase: float,
        module_width: float,
        module_height: float,
        min_gcr: float = 0.0,
        max_gcr: float = 1.0,
        ) -> Tuple[float, int, List[Tuple[int, float, Polygon]]]:
    """
    Finds the least dense (lowest gcr) layout that fits max_num_modules. If that isn't possible, it finds the densest,
    highest gcr that fits as many modules as possible.
    """
    best: Tuple[float, int, List[(int, float, Polygon)]] = (0.0, 0, [])
    prepared_site = prep(site_shape)
    
    def objective(gcr: float) -> float:
        nonlocal best
        num_modules, strands = \
            place_solar_strands(
                max_num_modules,
                min_strand_length,
                site_shape,
                center,
                phase,
                gcr,
                module_width,
                module_height,
                prepared_site=prepared_site
                )
        
        delta_modules = num_modules - best[1]
        if delta_modules > 0 or (delta_modules == 0 and best[0] > gcr):
            best = gcr, num_modules, strands
        
        if num_modules < max_num_modules:
            # if the number of modules is less than the max, search denser, larger gcrs
            return -1
        return 1
    
    gcr, _ = binary_search_float(
        objective,
        min_gcr,
        max_gcr,
        max_iters=32,
        threshold=1e-4)
    
    return best


def find_best_solar_size(
        max_num_modules: int,
        min_strand_length: int,
        site_shape: BaseGeometry,
        center: np.ndarray,
        phase: float,
        module_width: float,
        module_height: float,
        gcr,
        aspect,
        min_size,
        max_size
        ) -> Tuple[float, int, List[Tuple[int, float, Polygon]], np.ndarray]:
    """
    Finds the smallest size that fits max_num_modules. If that isn't possible, it fits as many modules as it can.
    """
    best: Tuple[float, int, List[(int, float, Polygon)], np.ndarray] = (0.0, 0, [], Point(0, 0).buffer(.01), np.zeros(2))
    prepared_site = prep(site_shape)
    
    def objective(x_length: float) -> float:
        nonlocal best
        size = np.array([x_length, aspect * x_length])
        sw_bound = center - size / 2
        ne_bound = center + size / 2
        solar_bounds = (sw_bound, ne_bound)
        bounds_shape = make_polygon_from_bounds(sw_bound, ne_bound)
        valid_region = bounds_shape.intersection(site_shape)
        
        num_modules, strands = \
            place_solar_strands(
                max_num_modules,
                min_strand_length,
                valid_region,
                Point(center),
                phase,
                gcr,
                module_width,
                module_height,
                prepared_site=prepared_site
                )
        
        delta_modules = num_modules - best[1]
        if delta_modules > 0 or (delta_modules == 0 and best[0] > x_length):
            best = x_length, num_modules, strands, valid_region, solar_bounds
        
        if num_modules < max_num_modules:
            # if the number of modules is less than the max, search larger sizes
            return -1
        # if we fit the max modules, find the smallest size that fits them all
        return 1
    
    binary_search_float(
        objective,
        min_size,
        max_size,
        max_iters=32,
        threshold=1e-1)
    
    return best


def place_solar_strands(max_num_modules: int,
                        min_strand_length: int,
                        site_shape: BaseGeometry,
                        center: Point,
                        phase_offset: float,
                        gcr: float,
                        module_width: float,
                        module_height: float,
                        prepared_site: Optional[PreparedGeometry] = None,
                        ) -> Tuple[int, List[Tuple[int, float, LineString]]]:
    """
    Places rows of solar strands within the given site where each strand is described by:
        - num_modules: number of solar panels
        - length:
        - segment: a LineString
    """
    # spacing between subrows of solar panels set by gcr
    interrow_spacing = module_width / np.sqrt(gcr)
    raw_phase_offset = phase_offset * interrow_spacing
    
    # print('place_solar_strands', max_num_modules, min_strand_length, phase_offset, gcr, module_width, module_height,
    #       interrow_spacing, raw_phase_offset)
    grid_center = translate(center, xoff=raw_phase_offset)
    
    # prep_site = prep(site_shape)
    
    grid_lines = make_grid_lines(
        site_shape,
        translate(center, xoff=raw_phase_offset),
        np.pi / 2,  # N-S orientation
        interrow_spacing
        )
    
    # for segment in grid_lines:
    #     pyplot.plot([point[0] for point in segment.coords], [point[1] for point in segment.coords], 'b')
    
    prepared_site = prep(site_shape) if prepared_site is None else prepared_site
    
    # generate a valid (but possibly suboptimal) strand list
    module_site = Polygon([(0, 0), (module_width, 0), (module_width, module_height), (0, module_height)])
    strands: List[(int, float, Polygon)] = []
    num_modules_remaining: int = max_num_modules
    for row_number, grid_line in enumerate(grid_lines):
        if num_modules_remaining < min_strand_length:
            break
        
        if not prepared_site.intersects(grid_line):
            continue
        
        if not site_shape.intersects(grid_line):
            continue
        intersection_result = site_shape.intersection(grid_line)
        
        if isinstance(intersection_result, GeometryCollection):
            intersections = list(intersection_result.geoms)
        else:
            intersections = [intersection_result]
        
        lines = []
        for intersection in intersections:
            if isinstance(intersection, MultiLineString):
                lines.extend(intersection.geoms)
            else:
                lines.append(intersection)
        
        for segment in lines:
            length = segment.length
            num_modules = min(num_modules_remaining, floor(length / module_height))
            # print('seg ', segment.length, num_modules)
            if num_modules >= min_strand_length:
                strands.append((num_modules, length, segment))
                num_modules_remaining -= num_modules
                # plt.plot([point[0] for point in segment.coords], [point[1] for point in segment.coords], '.b')
    
    return (max_num_modules - num_modules_remaining), strands


def get_flicker_loss_multiplier(flicker_data: Tuple[float, np.ndarray, np.ndarray, np.ndarray],
                                turbine_coords_x: list,
                                turbine_coords_y: list,
                                turbine_diameter: float,
                                module_dimensions: Tuple[float, float],
                                primary_strands: List[Tuple[int, float, Polygon]]=None,
                                module_points: MultiPoint=None):
    """
    Aggregated loss multiplier of solar output in primary strands due to turbine flicker
    :param flicker_data: (turbine diameter used in flicker modeling,
                          indicies of location of turbine,
                          2-D array containing flicker loss multiplier at x, y coordinates (0-1, 0 is no loss),
                          x_coordinates of grid,
                          y_coordinates of grid)
    :param turbine_coords_x: list of turbine locations x coordinates
    :param turbine_coords_y: list of turbine locations y coordinates
    :param turbine_diameter: the diameter of turbines in meters
    :param module_dimensions: tuple of module width & height in meters
    :param primary_strands: list of (num_modules, length, shapely.geometry.String) of strands of solar panels
    :param module_points: MultiPoint object with module locations
    :return: loss multiplier
    """
    if primary_strands is None and module_points is None:
        raise ValueError("Either `primary_strands` or `module_points` must be provided.")
    elif primary_strands is not None and module_points is None:
        if len(primary_strands) == 0:
            return 1
        mode = "strands"
        total_power = sum([row[0] for row in primary_strands])  # assume each module has unit power output
    elif primary_strands is None and module_points is not None:
        total_power = len(module_points)
        if total_power == 0:
            return 1
        mode = "points"
    else:
        raise ValueError("Only one of `primary_strands` and `module_points` must be provided.")
    
    turb_diam = flicker_data[0]
    # TODO: implement different flicker table for larger turbine
    # if abs(turb_diam - turbine_diameter) > 10:
    #     raise NotImplementedError("Scaling of flicker look up table to different turbine diameters not implemented yet")
    
    turb_index = flicker_data[1]
    heatmap = flicker_data[2]
    x_coords, y_coords = flicker_data[3], flicker_data[4]
    
    x_min, x_max = x_coords[0], x_coords[-1]
    y_min, y_max = y_coords[0], y_coords[-1]
    turb_x, turb_y = x_coords[turb_index[0]], y_coords[turb_index[1]]
    active_area_around_turbine = Polygon(((x_min, y_min),
                                          (x_min, y_max),
                                          (x_max, y_max),
                                          (x_max, y_min)))
    gridcell_width = x_coords[1] - x_coords[0]
    gridcell_height = y_coords[1] - y_coords[0]
    
    flicker_power = total_power
    num_turbines = len(turbine_coords_x)
    for n in range(num_turbines):
        t_x = turbine_coords_x[n]
        t_y = turbine_coords_y[n]
        dx, dy = t_x - turb_x, t_y - turb_y
        active_area_translated = translate(active_area_around_turbine, dx, dy)

        modules = []
        if mode == 'strands':
            with warnings.catch_warnings():
                # if intersection is empty will get warnings, turn those off
                warnings.simplefilter("ignore")
                active_segments = map(active_area_translated.intersection, [row[2] for row in primary_strands])
                for i, s in enumerate(active_segments):
                    if not s.is_empty:
                        # figure out the orientation of the modules, whether the module_distance is laid out by width or by height
                        length_per_module = primary_strands[0][1] / primary_strands[0][0] 
                        module_distance = module_dimensions[np.argmin([abs(d - length_per_module) for d in module_dimensions])]

                        distances = np.arange(0, s.length * (1 + 1e-6), module_distance)
                        modules += [s.interpolate(distance) for distance in distances]
        if mode == "points":
            modules = active_area_translated.intersection(module_points)
            if modules.is_empty:
                modules = []

        if len(modules) == 0:
            continue

        mods_x = np.array([p.x for p in modules])
        mods_y = np.array([p.y for p in modules])
        mods_dx_from_t = mods_x - t_x
        mods_dy_from_t = mods_y - t_y

        # map from dist(module, turbine t) to dist(heatmap grid coordinate, turbine in flicker model)
        x_coords_ind = ((mods_dx_from_t - x_min) / gridcell_width).round().astype(int)
        y_coords_ind = ((mods_dy_from_t - y_min) / gridcell_height).round().astype(int)
        flicker_val = heatmap[y_coords_ind, x_coords_ind]
        flicker_power -= sum(flicker_val)

    return flicker_power / total_power


def calculate_max_hybrid_aep(site_info: SiteInfo,
                             num_turbines: int,
                             solar_capacity_kw: float
                             ) -> dict:
    """
    Calculates the max total pv and solar annual energy output by assuming no wake, gcr or flicker losses.
    All other factors and losses are not adjusted because they remain constant throughout the optimization
    :return: dictionary of "wind", "solar" and "total" max AEPs
    """
    upper_bounds = dict()
    
    # wind
    wind_model = windpower.default("WindPowerSingleOwner")
    wind_model.Resource.wind_resource_data = site_info.wind_resource.data
    
    wind_params_orig = wind_model.export()
    wind_model.Farm.wind_farm_xCoordinates = np.zeros(num_turbines)
    wind_model.Farm.wind_farm_yCoordinates = np.zeros(num_turbines)
    wind_model.Farm.system_capacity = num_turbines * max(wind_model.Turbine.wind_turbine_powercurve_powerout)
    wind_model.Farm.wind_farm_wake_model = 3  # constant wake loss model which we can set to 0
    wind_model.Losses.wake_int_loss = 0
    wind_model.execute(0)
    
    # solar
    solar_model = pvwatts.default("PVWattsSingleOwner")
    solar_model.SolarResource.solar_resource_data = site_info.solar_resource.data
    solar_model.SystemDesign.array_type = 2  # single-axis tracking
    
    solar_params_orig = solar_model.export()
    solar_model.SystemDesign.gcr = 0.01  # lowest possible gcr
    solar_model.SystemDesign.system_capacity = solar_capacity_kw
    solar_model.execute(0)
    
    upper_bounds['wind'] = wind_model.Outputs.annual_energy / 1000
    upper_bounds['solar'] = solar_model.Outputs.annual_energy / 1000
    upper_bounds['total'] = upper_bounds['wind'] + upper_bounds['solar']
    
    # restore original parameters
    wind_model.assign(wind_params_orig)
    solar_model.assign(solar_params_orig)
    
    return upper_bounds
