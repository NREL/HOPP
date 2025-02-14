from __future__ import annotations
from typing import Union, NamedTuple
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.affinity import scale
import PySAM.Windpower as windpower
from attrs import define, field
from typing import Optional
from hopp.utilities.log import hybrid_logger as logger
from hopp.simulation.technologies.layout.wind_layout_tools import (
    get_best_grid,
    get_evenly_spaced_points_along_border,
    subtract_turbine_exclusion_zone,
    make_site_boundary_for_square_grid_layout,
    create_grid,
    check_turbines_in_site,
    adjust_site_for_box_grid_layout
    )
from hopp.utilities.validators import contains, range_val
from hopp.simulation.technologies.sites.site_shape_tools import plot_site_polygon
from hopp.simulation.base import BaseClass
from typing import List

@define
class WindBasicGridParameters:
    """Configuration class for 'basicgrid' wind layout.

    Args:
        row_D_spacing (float, Optional): rotor diameter multiplier for spacing between rows of turbines (y direction).
            Defaults to 5.0.
        turbine_D_spacing (float, Optional): rotor diameter multiplier for spacing between turbines in a row (x direction).
            Defaults to 5.0.
        grid_angle (float, Optional): grid rotation angle in degrees where 0 is North, increasing clockwise. 
            Defaults to 0.0.
        row_phase_offset (float, Optional): offset of turbines along row from one row to the next.
            Value must be between 0 and 1. Defaults to 0.0.
        site_boundary_constrained (bool, Optional): whether to constrain the layout to the site. Defaults to False.
    """

    row_D_spacing: Optional[float] = field(default = 5.0)
    turbine_D_spacing: Optional[float]= field(default = 5.0)
    grid_angle: Optional[float] = field(default = 0.0)
    row_phase_offset: Optional[float] = field(default = 0.0, validator=range_val(0.0, 1.0))
    site_boundary_constrained: Optional[bool] = field(default = False)

@define
class WindBoundaryGridParameters:
    """ Configuration class for 'boundarygrid' wind layout.

    Args:
        border_spacing: 
            spacing along border = (1 + border_spacing) * min spacing
        border_spacing_m:
        min_spacing_m
        border_offset: turbine border spacing offset as ratio of border spacing  (0, 1)
        
        grid_angle: turbine inner grid rotation (0, 180) [degrees]
        grid_aspect_power: grid aspect ratio [cols / rows] = 2^grid_aspect_power
        row_phase_offset: inner grid phase offset (0,1)  (20% suggested)
    """

    #TODO: rename to border_spacing_ratio?
    border_spacing: float = field(default = 0.0)
    #TODO: rename to border_offset_ratio?
    border_offset: float = field(default = 0.0, validator = range_val(0.0, 1.0)) 
    border_spacing_m: Optional[float] = field(default = None)
    min_spacing_m: Optional[float] = field(default = None)

    grid_angle: float = field(default = 0.0, validator = range_val(0.0, 180.0))
    grid_aspect_power: Optional[float] = field(default = None)
    grid_aspect_ratio: Optional[float] = field(default = None)
    row_phase_offset: float = field(default = 0.2, validator = range_val(0.0, 1.0))
    def __attrs_post_init__(self):
        
        if self.grid_aspect_power is not None and self.grid_aspect_ratio is None:
            #NOTE: unsure if this equation is correct given doc strong
            self.grid_aspect_ratio = np.exp(self.grid_aspect_power) 
        if self.grid_aspect_power is None and self.grid_aspect_ratio is None:
            self.grid_aspect_ratio = 1.0
            
        if self.min_spacing_m is not None and self.border_spacing_m is not None:
            self.border_spacing = (self.border_spacing_m/self.min_spacing_m) - 1

@define
class WindCustomParameters:
    """
    direct user input of the x and y coordinates
    """

    layout_x: List[float]
    layout_y: List[float]


@define
class WindLayout(BaseClass):
    """Class to manage wind farm layout.

    Args:
        site_polygon (Polygon | BaseGeometry): site polygon shape.
        _system_model (windpower.Windpower | Floris): pysam wind power object
        layout_mode (str): layout choice:  "boundarygrid", "grid", "custom", "basicgrid"
        parameters (Union[WindBoundaryGridParameters, WindCustomParameters, WindBasicGridParameters, None]): wind
            layout parameters for the corresponding `layout_mode`

        min_spacing_meters (float, Optional): minimum spacing between turbines in meters. Defaults to 0.0.
        max_spacing_meters (float, Optional): maximum spacing between turbines in meters. Defaults to 2e6.
        min_rotor_diameter_multiplier (float, Optional): minimum spacing between turbines as multiplier of rotor diameter. Defaults to 2.0
        max_rotor_diameter_multiplier (float, Optional): maximum spacing between turbines as multiplier of rotor diameter. Defaults to 20.0
        turbine_rating_kW (float, Optional): rating of a single turbine in kW. if not provided, turbine power is estimated from the power-curve.
    """
    site_polygon: Union[Polygon, BaseGeometry] 
    _system_model: windpower.Windpower
    layout_mode: str = field(validator=contains(['boundarygrid', 'grid', 'custom','basicgrid']))
    parameters: Union[WindBoundaryGridParameters, WindCustomParameters, WindBasicGridParameters, None]
    min_spacing_meters: Optional[float] = field(default = 0.0)
    max_spacing_meters: Optional[float] = field(default = 2e6)

    min_rotor_diameter_multiplier: Optional[float] = field(default = 2.0)
    max_rotor_diameter_multiplier: Optional[float] = field(default = 20.0)
    
    turbine_rating_kW: Optional[float] = field(default = None)

    turb_pos_x: List[float] = field(init=False)
    turb_pos_y: List[float] = field(init=False)

    min_spacing: float = field(init = False)
    max_spacing: float = field(init = False)
    
    def __attrs_post_init__(self):
        self.min_spacing = max(
            self.min_spacing_meters, 
            self._system_model.value("wind_turbine_rotor_diameter") * self.min_rotor_diameter_multiplier
        )
        self.max_spacing = max(
            self.max_spacing_meters, 
            self._system_model.value("wind_turbine_rotor_diameter") * self.max_rotor_diameter_multiplier
        )

        # turbine layout values
        self.turb_pos_x = self._system_model.value("wind_farm_xCoordinates")
        self.turb_pos_y = self._system_model.value("wind_farm_yCoordinates")

        if self.layout_mode == 'boundarygrid' and isinstance(self.parameters,dict):
            self.parameters = WindBoundaryGridParameters(**self.parameters)
        elif self.layout_mode == 'basicgrid' and isinstance(self.parameters,dict):
            self.parameters = WindBasicGridParameters(**self.parameters)
        elif self.layout_mode == 'custom' and isinstance(self.parameters,dict):
            self.parameters = WindCustomParameters(**self.parameters)

    def _get_system_config(self):
        self.min_spacing = max(
            self.min_spacing,
            self.min_spacing_meters, 
            self._system_model.value("wind_turbine_rotor_diameter") * self.min_rotor_diameter_multiplier
        )
        self.max_spacing = max(
            self.max_spacing_meters, 
            self._system_model.value("wind_turbine_rotor_diameter") * self.max_rotor_diameter_multiplier
        )


    def _set_system_layout(self):
        self._system_model.value("wind_farm_xCoordinates", self.turb_pos_x)
        self._system_model.value("wind_farm_yCoordinates", self.turb_pos_y)

        n_turbines = len(self.turb_pos_x)
        if self.turbine_rating_kW is None:
            turb_rating = max(self._system_model.value("wind_turbine_powercurve_powerout"))
            self._system_model.value("system_capacity", n_turbines * turb_rating)
        else:
            self._system_model.value("system_capacity", n_turbines * self.turbine_rating_kW)
        logger.info("Wind Layout set with {} turbines for {} kw system capacity".format(n_turbines,
                                                                                        n_turbines * turb_rating))

    @property
    def rotor_diameter(self):
        return self._system_model.value("wind_turbine_rotor_diameter")

    def reset_boundarygrid(self,
                           n_turbines,
                           exclusions: Polygon = None):

        self._get_system_config()

        wind_shape = Polygon(self.site_polygon.exterior)
        if exclusions is not None:
            wind_shape = wind_shape.difference(exclusions)  # compute valid wind layout shape

        # place border turbines
        turbine_positions: list[Point] = []
        if not isinstance(wind_shape, MultiPolygon):
            wind_shape = MultiPolygon([wind_shape, ])

        border_spacing = (self.parameters.border_spacing + 1) * self.min_spacing
        for bounding_shape in wind_shape.geoms:
            turbine_positions.extend(
                get_evenly_spaced_points_along_border(
                    bounding_shape.exterior,
                    border_spacing,
                    self.parameters.border_offset,
                    n_turbines - len(turbine_positions),
                ))

        valid_wind_shape = subtract_turbine_exclusion_zone(self.min_spacing, wind_shape, turbine_positions)

        # place interior grid turbines
        print(f"grid aspect power: {self.parameters.grid_aspect_power}")
        max_num_interior_turbines = n_turbines - len(turbine_positions)
        intrarow_spacing, grid_sites = get_best_grid(
            valid_wind_shape,
            wind_shape.centroid,
            self.parameters.grid_angle,
            self.parameters.grid_aspect_ratio,
            self.parameters.row_phase_offset,
            self.max_spacing,
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
        if not self.site_polygon:
            raise ValueError("WindPlant set_num_turbines_in_grid requires site polygon")

        if n_turbines > 0:
            spacing = np.sqrt(
                self.site_polygon.area / n_turbines) * self.site_polygon.envelope.area / self.site_polygon.area
            spacing = max(spacing, self.min_spacing)
            coords = []
            while len(coords) < n_turbines:

                envelope = Polygon(self.site_polygon.envelope)
                while len(coords) < n_turbines and envelope.area > spacing * spacing:
                    d = 0
                    sub_boundary = envelope.boundary
                    while d <= sub_boundary.length and len(coords) < n_turbines:
                        coord = sub_boundary.interpolate(d)
                        if self.site_polygon.buffer(1e3).contains(coord):
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

    def reset_basic_grid(self,n_turbines):
        
        self._get_system_config()

        interrow_spacing = self.parameters.row_D_spacing*self.rotor_diameter
        intrarow_spacing = self.parameters.turbine_D_spacing*self.rotor_diameter
            
        data = make_site_boundary_for_square_grid_layout(n_turbines,self.rotor_diameter,self.parameters.row_D_spacing,self.parameters.turbine_D_spacing)
        vertices = np.array([np.array(v) for v in data['site_boundaries']['verts']])
        square_bounds = Polygon(vertices)
        grid_position_square = create_grid(square_bounds,
                square_bounds.centroid,
                self.parameters.grid_angle,
                intrarow_spacing,
                interrow_spacing,
                self.parameters.row_phase_offset,
                int(n_turbines),
        )
       
        if self.parameters.site_boundary_constrained:
            # 1) see if turbines are in the site polygon
            xcoords_grid = [point.x for point in grid_position_square]
            ycoords_grid = [point.y for point in grid_position_square]
            x_ingrid,y_ingrid = check_turbines_in_site(xcoords_grid,ycoords_grid,self.site_polygon)
            if len(x_ingrid)==n_turbines:
                self.turb_pos_x, self.turb_pos_y = x_ingrid,y_ingrid
                self._set_system_layout()
                return 
            x,y = adjust_site_for_box_grid_layout(
                self.site_polygon,
                n_turbines,
                interrow_spacing,
                intrarow_spacing,
                self.parameters.row_phase_offset,
                self.parameters.grid_angle
            )
            if len(x)==n_turbines or len(x)>x_ingrid:
                self.turb_pos_x, self.turb_pos_y = x_ingrid,y_ingrid
                self._set_system_layout()
                return 
            else:
                self.reset_grid(n_turbines)
        else:
            xcoords_grid = [point.x for point in grid_position_square]
            ycoords_grid = [point.y for point in grid_position_square]
            self.turb_pos_x, self.turb_pos_y = xcoords_grid,ycoords_grid
            self._set_system_layout()
            
    def set_layout_params(self,
                          wind_kw,
                          params: Optional[WindBoundaryGridParameters],
                          exclusions: Polygon = None):
        if params:
            self.parameters = params
        n_turbines = int(np.floor(wind_kw / max(self._system_model.Turbine.wind_turbine_powercurve_powerout)))
        if self.layout_mode == 'boundarygrid':
            self.reset_boundarygrid(n_turbines, exclusions)
        elif self.layout_mode == 'grid':
            self.reset_grid(n_turbines)
        elif self.layout_mode == 'basicgrid':
            self.reset_basic_grid(n_turbines)
        elif self.layout_mode == 'custom':
            self.turb_pos_x, self.turb_pos_y = self.parameters.layout_x, self.parameters.layout_y
            self._set_system_layout()

    def set_num_turbines(self,
                         n_turbines: int):
        """
        Changes number of turbines in the existing layout
        """
        self._get_system_config()

        if self.layout_mode == 'boundarygrid':
            self.reset_boundarygrid(n_turbines)
        elif self.layout_mode == 'grid':
            self.reset_grid(n_turbines)
        elif self.layout_mode == 'basicgrid':
            self.reset_basic_grid(n_turbines)
        elif self.layout_mode == 'custom':
            self.turb_pos_x, self.turb_pos_y = self.parameters.layout_x, self.parameters.layout_y
            self._set_system_layout()

    def plot(self,
             figure=None,
             axes=None,
             turbine_color='b',
             site_border_color='k',
             site_alpha=0.95,
             linewidth=4.0
             ):
        if not figure and not axes:
            figure, axes = plot_site_polygon(self.site_polygon,figure, axes, site_border_color, site_alpha, linewidth)

        turb_pos_x = self._system_model.value("wind_farm_xCoordinates")
        turb_pos_y = self._system_model.value("wind_farm_yCoordinates")
        for n in range(len(turb_pos_x)):
            x, y = turb_pos_x[n], turb_pos_y[n]
            circle = plt.Circle(
                (x, y),
                radius=self.rotor_diameter/2.0,
                # linewidth=linewidth * 10,
                color=turbine_color,
                fill=True,
                linewidth=linewidth,
                )
            axes.add_patch(circle)

        return figure, axes
