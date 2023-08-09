from __future__ import annotations
from typing import Union, NamedTuple
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.affinity import scale
import PySAM.Windpower as windpower

from hybrid.sites import SiteInfo
from hybrid.log import hybrid_logger as logger
from hybrid.layout.wind_layout_tools import (
    get_best_grid,
    get_evenly_spaced_points_along_border,
    subtract_turbine_exclusion_zone
    )


class WindBoundaryGridParameters(NamedTuple):
    """
    border_spacing: spacing along border = (1 + border_spacing) * min spacing
    border_offset: turbine border spacing offset as ratio of border spacing  (0, 1)
    grid_angle: turbine inner grid rotation (0, pi) [radians]
    grid_aspect_power: grid aspect ratio [cols / rows] = 2^grid_aspect_power
    row_phase_offset: inner grid phase offset (0,1)  (20% suggested)
    """
    border_spacing: float
    border_offset: float
    grid_angle: float
    grid_aspect_power: float
    row_phase_offset: float


class WindCustomParameters(NamedTuple):
    """
    direct user input of the x and y coordinates
    """

    layout_x: list
    layout_y: list


class WindLayout:
    """

    """
    def __init__(self,
                 site_info: SiteInfo,
                 wind_source: windpower.Windpower,
                 layout_mode: str,
                 parameters: Union[WindBoundaryGridParameters, WindCustomParameters, None],
                 min_spacing: float = 200.,
                 ):
        """

        """
        self.site: SiteInfo = site_info
        self._system_model: windpower.Windpower = wind_source
        self.min_spacing = max(min_spacing, self._system_model.value("wind_turbine_rotor_diameter") * 2)

        if layout_mode not in ('boundarygrid', 'grid', 'custom'):
            raise ValueError('Options for `layout_mode` are: "boundarygrid", "grid", "custom"')
        self._layout_mode = layout_mode

        # layout design parameters
        self.parameters = parameters

        # turbine layout values
        self.turb_pos_x = self._system_model.value("wind_farm_xCoordinates")
        self.turb_pos_y = self._system_model.value("wind_farm_yCoordinates")

    def _get_system_config(self):
        self.min_spacing = max(self.min_spacing, self._system_model.value("wind_turbine_rotor_diameter") * 2)

    def _set_system_layout(self):
        self._system_model.value("wind_farm_xCoordinates", self.turb_pos_x)
        self._system_model.value("wind_farm_yCoordinates", self.turb_pos_y)

        n_turbines = len(self.turb_pos_x)
        turb_rating = max(self._system_model.value("wind_turbine_powercurve_powerout"))
        self._system_model.value("system_capacity", n_turbines * turb_rating)
        logger.info("Wind Layout set with {} turbines for {} kw system capacity".format(n_turbines,
                                                                                        n_turbines * turb_rating))

    @property
    def rotor_diameter(self):
        return self._system_model.value("wind_turbine_rotor_diameter")

    def reset_boundarygrid(self,
                           n_turbines,
                           parameters: WindBoundaryGridParameters,
                           exclusions: Polygon = None):
        """

        """
        self._get_system_config()

        wind_shape = Polygon(self.site.polygon.exterior)
        if exclusions:
            wind_shape = wind_shape.difference(exclusions)  # compute valid wind layout shape

        # place border turbines
        turbine_positions: list[Point] = []
        if not isinstance(wind_shape, MultiPolygon):
            wind_shape = MultiPolygon([wind_shape, ])

        border_spacing = (parameters.border_spacing + 1) * self.min_spacing
        for bounding_shape in wind_shape.geoms:
            turbine_positions.extend(
                get_evenly_spaced_points_along_border(
                    bounding_shape.exterior,
                    border_spacing,
                    parameters.border_offset,
                    n_turbines - len(turbine_positions),
                ))

        valid_wind_shape = subtract_turbine_exclusion_zone(self.min_spacing, wind_shape, turbine_positions)

        # place interior grid turbines
        max_num_interior_turbines = n_turbines - len(turbine_positions)
        grid_aspect = np.exp(parameters.grid_aspect_power)
        intrarow_spacing, grid_sites = get_best_grid(
            valid_wind_shape,
            wind_shape.centroid,
            parameters.grid_angle,
            grid_aspect,
            parameters.row_phase_offset,
            self.min_spacing * 10000,
            self.min_spacing,
            max_num_interior_turbines,
        )
        turbine_positions.extend(grid_sites)
        xcoords, ycoords = [], []
        for p in turbine_positions:
            xcoords.append(p.x)
            ycoords.append(p.y)

        self.turb_pos_x, self.turb_pos_y = xcoords, ycoords
        self._set_system_layout()

    def reset_grid(self,
                   n_turbines):
        """
        Set the number of turbines. System capacity gets modified as a result.
        Wind turbines will be placed in a grid

        :param n_turbines: int
        """
        self._get_system_config()

        xcoords = []
        ycoords = []
        if not self.site.polygon:
            raise ValueError("WindPlant set_num_turbines_in_grid requires site polygon")

        if n_turbines > 0:
            spacing = np.sqrt(
                self.site.polygon.area / n_turbines) * self.site.polygon.envelope.area / self.site.polygon.area
            spacing = max(spacing, self._system_model.value("wind_turbine_rotor_diameter") * 3)
            coords = []
            while len(coords) < n_turbines:

                envelope = Polygon(self.site.polygon.envelope)
                while len(coords) < n_turbines and envelope.area > spacing * spacing:
                    d = 0
                    sub_boundary = envelope.boundary
                    while d <= sub_boundary.length and len(coords) < n_turbines:
                        coord = sub_boundary.interpolate(d)
                        if self.site.polygon.buffer(1e3).contains(coord):
                            coords.append(coord)
                        d += spacing
                    if len(coords) < n_turbines:
                        envelope = scale(envelope, (envelope.bounds[2] - spacing) / envelope.bounds[2],
                                         (envelope.bounds[3] - spacing) / envelope.bounds[3])
                if len(coords) < n_turbines:
                    spacing *= .95
                    coords = []
            for _, p in enumerate(coords):
                xcoords.append(p.x)
                ycoords.append(p.y)

        self.turb_pos_x, self.turb_pos_y = xcoords, ycoords
        self._set_system_layout()

    def set_layout_params(self,
                          wind_kw,
                          params: Union[WindBoundaryGridParameters, WindCustomParameters, None],
                          exclusions: Polygon = None):
        self.parameters = params
        n_turbines = int(np.floor(wind_kw / max(self._system_model.Turbine.wind_turbine_powercurve_powerout)))
        if self._layout_mode == 'boundarygrid':
            self.reset_boundarygrid(n_turbines, params, exclusions)
        elif self._layout_mode == 'grid':
            self.reset_grid(n_turbines)
        elif self._layout_mode == 'custom':
            self.turb_pos_x, self.turb_pos_y = self.parameters.layout_x, self.parameters.layout_y
            self._set_system_layout()

    def set_num_turbines(self,
                         n_turbines: int):
        """
        Changes number of turbines in the existing layout
        """
        self._get_system_config()

        if self._layout_mode == 'boundarygrid':
            self.reset_boundarygrid(n_turbines, self.parameters)
        elif self._layout_mode == 'grid':
            self.reset_grid(n_turbines)

    def plot(self,
             figure=None,
             axes=None,
             turbine_color='b',
             site_border_color='k',
             site_alpha=0.95,
             linewidth=4.0
             ):
        if not figure and not axes:
            figure, axes = self.site.plot(figure, axes, site_border_color, site_alpha, linewidth)

        turb_pos_x = self._system_model.value("wind_farm_xCoordinates")
        turb_pos_y = self._system_model.value("wind_farm_yCoordinates")

        for n in range(len(turb_pos_x)):
            x, y = turb_pos_x[n], turb_pos_y[n]
            circle = plt.Circle(
                (x, y),
                linewidth * 10,
                color=turbine_color,
                fill=True,
                linewidth=linewidth,
                )
            axes.add_patch(circle)

        return figure, axes
