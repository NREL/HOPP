from typing import NamedTuple, Optional, Union
import numpy as np
from shapely.geometry import Point, Polygon
import PySAM.Pvwattsv8 as pv_simple
import PySAM.Pvsamv1 as pv_detailed

from hybrid.log import hybrid_logger as logger
from hybrid.sites import SiteInfo
from hybrid.layout.pv_module import get_module_attribs
from hybrid.layout.plot_tools import plot_shape
from hybrid.layout.layout_tools import make_polygon_from_bounds
from hybrid.layout.pv_layout_tools import find_best_solar_size
from hybrid.layout.pv_design_utils import *


class PVGridParameters(NamedTuple):
    """
    x_position: ratio of solar's x coords to site width (0, 1)
    y_position: ratio of solar's y coords to site height (0, 1)
    aspect_power: aspect ratio of solar to site width = 2^solar_aspect_power
    gcr: gcr ratio of solar patch
    s_buffer: south side buffer ratio (0, 1)
    x_buffer: east and west side buffer ratio (0, 1)
    """
    x_position: float
    y_position: float
    aspect_power: float
    gcr: float
    s_buffer: float
    x_buffer: float


class PVSimpleParameters(NamedTuple):
    """
    gcr: gcr ratio of solar patch
    """
    gcr: float


class PVLayout:
    """

    """

    def __init__(self,
                 site_info: SiteInfo,
                 solar_source: Union[pv_simple.Pvwattsv8, pv_detailed.Pvsamv1],
                 parameters: Optional[PVGridParameters] = None,
                 min_spacing: float = 100.
                 ):
        self.site: SiteInfo = site_info
        self._system_model: Union[pv_simple.Pvwattsv8, pv_detailed.Pvsamv1] = solar_source
        self.min_spacing = min_spacing

        module_attribs = get_module_attribs(self._system_model)
        self.module_power: float = module_attribs['P_mp_ref']
        self.module_width: float = module_attribs['width']
        self.module_height: float = module_attribs['length']
        self.modules_per_string: int = get_modules_per_string(self._system_model)

        inverter_attribs = get_inverter_attribs(self._system_model)
        self.inverter_power: float = inverter_attribs['P_ac']

        # solar array layout variables
        self.parameters = parameters

        # grid layout design values
        self.strands: list = []
        self.solar_region: Polygon = Polygon()
        self.buffer_region: Polygon = Polygon()
        self.excess_buffer: float = 0
        self.flicker_loss = 0
        self.num_modules = 0

    def _set_system_layout(self):
        if self.parameters:
            if isinstance(self._system_model, pv_simple.Pvwattsv8):
                self._system_model.SystemDesign.gcr = self.parameters.gcr
            elif isinstance(self._system_model, pv_detailed.Pvsamv1):
                self._system_model.SystemDesign.subarray1_gcr = self.parameters.gcr
        if type(self.parameters) == PVGridParameters:
            target_solar_kw = self.module_power * self.num_modules
            if isinstance(self._system_model, pv_simple.Pvwattsv8):
                self._system_model.SystemDesign.system_capacity = target_solar_kw
            elif isinstance(self._system_model, pv_detailed.Pvsamv1):
                n_strings, system_capacity, n_inverters = align_from_capacity(
                    system_capacity_target=target_solar_kw,
                    dc_ac_ratio=self.get_dc_ac_ratio(),
                    modules_per_string=self.modules_per_string,
                    module_power=self.module_power,
                    inverter_power=get_inverter_attribs(self._system_model)['P_ac'],
                )
                self._system_model.SystemDesign.subarray1_nstrings = n_strings
                self._system_model.SystemDesign.system_capacity = system_capacity
                self._system_model.SystemDesign.inverter_count = n_inverters

            logger.info(f"Solar Layout set for {self.module_power * self.num_modules} kw")
        self._system_model.AdjustmentFactors.constant = self.flicker_loss * 100  # percent

    def compute_pv_layout(self,
                        solar_kw: float,
                        parameters: PVGridParameters = None):
        if not parameters:
            return

        site_sw_bound = np.array([self.site.polygon.bounds[0], self.site.polygon.bounds[1]])
        site_ne_bound = np.array([self.site.polygon.bounds[2], self.site.polygon.bounds[3]])
        site_bounds_size = site_ne_bound - site_sw_bound

        solar_center = site_sw_bound + site_bounds_size * \
                       np.array([parameters.x_position, parameters.y_position])

        # place solar
        num_modules = int(np.floor(solar_kw / self.module_power))
        max_solar_width = self.module_width * num_modules \
                          / self.modules_per_string

        if max_solar_width < self.module_width:
            self.buffer_region = make_polygon_from_bounds(np.array([0, 0]), np.array([0, 0]))
            self.solar_region = make_polygon_from_bounds(np.array([0, 0]), np.array([0, 0]))
            self.strands = []
            self._set_system_layout()
            return

        solar_aspect = np.exp(parameters.aspect_power)
        solar_x_size, self.num_modules, self.strands, self.solar_region, solar_bounds = \
            find_best_solar_size(
                num_modules,
                self.modules_per_string,
                self.site.polygon,
                solar_center,
                0.0,
                self.module_width,
                self.module_height,
                parameters.gcr,
                solar_aspect,
                self.module_width,
                max_solar_width,
            )

        solar_x_buffer_length = self.min_spacing * (1 + parameters.x_buffer)
        solar_s_buffer_length = self.min_spacing * (1 + parameters.s_buffer)
        self.buffer_region = make_polygon_from_bounds(
            solar_bounds[0] - np.array([solar_x_buffer_length, solar_s_buffer_length]),
            solar_bounds[1] + np.array([solar_x_buffer_length, 0]))

        def get_bounds_center(shape):
            bounds = shape.bounds
            return Point(.5 * (bounds[0] + bounds[2]), .5 * (bounds[1] + bounds[3]))

        def get_excess_buffer(buffer, solar_region, bounding_shape):
            excess_buffer = 0.0
            buffer_intersection = buffer.intersection(bounding_shape)

            if buffer_intersection.area > 1e-3:
                shape_center = get_bounds_center(buffer)
                intersection_center = get_bounds_center(buffer_intersection)
                shape_center_delta = \
                    np.abs(np.array(shape_center.coords) - np.array(intersection_center.coords)) / site_bounds_size
                total_shape_center_delta = np.sum(shape_center_delta ** 2)
                excess_buffer += total_shape_center_delta

            bounds = buffer.bounds
            intersection_bounds = buffer_intersection.bounds

            if len(intersection_bounds) > 0:
                west_excess = intersection_bounds[0] - bounds[0]
                south_excess = intersection_bounds[1] - bounds[1]
                east_excess = bounds[2] - intersection_bounds[2]
                north_excess = bounds[3] - intersection_bounds[3]
            else:
                west_excess = south_excess = east_excess = north_excess = 0

            solar_bounds = solar_region.bounds
            actual_aspect = (solar_bounds[3] - solar_bounds[1]) / \
                            (solar_bounds[2] - solar_bounds[0])

            aspect_error = np.abs(np.log(actual_aspect) - np.log(solar_aspect))
            excess_buffer += aspect_error ** 2

            # excess buffer, minus minimum size
            # excess buffer is how much extra there is, but we must not penalise minimum sizes
            #
            # excess_x_buffer = max(0.0, es - min_spacing)
            # excess_y_buffer = max(0.0, min(ee, ew) - min_spacing)

            # if buffer has excess, then we need to penalize any excess buffer length beyond the minimum

            minimum_s_buffer = max(solar_s_buffer_length - south_excess, self.min_spacing)
            excess_x_buffer = (solar_s_buffer_length - minimum_s_buffer) / self.min_spacing
            excess_buffer += excess_x_buffer ** 2

            minimum_w_buffer = max(solar_x_buffer_length - west_excess, self.min_spacing)
            minimum_e_buffer = max(solar_x_buffer_length - east_excess, self.min_spacing)
            excess_y_buffer = (solar_x_buffer_length - max(minimum_w_buffer, minimum_e_buffer)) / self.min_spacing
            excess_buffer += excess_y_buffer ** 2

            return excess_buffer

        self.excess_buffer = get_excess_buffer(self.buffer_region, self.solar_region, self.site.polygon)

        self._set_system_layout()

        return self.excess_buffer

    def set_layout_params(self,
                          solar_kw: float,
                          params: Union[PVGridParameters, PVSimpleParameters]):
        self.parameters = params
        if type(params) == PVGridParameters:
            self.compute_pv_layout(solar_kw, params)
        elif type(params) == PVSimpleParameters:
            self._set_system_layout()

    def set_system_capacity(self,
                            size_kw):
        """
        Changes system capacity in the existing layout
        """
        if type(self.parameters) == PVGridParameters:
            self.compute_pv_layout(size_kw, self.parameters)
            if abs(self._system_model.SystemDesign.system_capacity - size_kw) > 1e-3 * size_kw:
                logger.warn(f"Could not fit {size_kw} kw into existing PV layout parameters of {self.parameters}")

    def set_flicker_loss(self,
                         flicker_loss_multipler: float):
        self.flicker_loss = flicker_loss_multipler
        self._set_system_layout()

    def get_dc_ac_ratio(self):
        return self._system_model.value('system_capacity') / \
               (self._system_model.value('inverter_count') * self.inverter_power)

    def plot(self,
             figure=None,
             axes=None,
             solar_color='darkorange',
             site_border_color='k',
             site_alpha=0.95,
             linewidth=4.0
             ):
        if not figure and not axes:
            figure, axes = self.site.plot(figure, axes, site_border_color, site_alpha, linewidth)

        plot_shape(figure, axes, self.solar_region, '-', color=solar_color)
        plot_shape(figure, axes, self.site.polygon.intersection(self.buffer_region), '--', color=solar_color)
