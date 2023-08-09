from typing import Union, Tuple, Optional, List
import datetime
import pytz

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from shapely.affinity import translate
from shapely.geometry import Point
from shapely.geometry import Polygon, MultiPolygon, MultiPoint
from shapely.ops import unary_union
import timezonefinder
from pysolar.solar import *
from pvmismatch import *

from hybrid.layout.pv_module import *


def get_time_zone(lat: float,
                  lon: float
                  ) -> pytz.tzinfo:
    timezone_str = timezonefinder.TimezoneFinder().certain_timezone_at(lat=lat, lng=lon)
    if timezone_str is None:
        raise ValueError("Could not determine the time zone")
    else:
        return pytz.timezone(timezone_str)


def get_sun_pos(lat: float,
                lon: float,
                step_in_minutes: float = 60,
                n: int = 8760,
                start_hr: int = 0,
                steps: Optional[range] = None
                ) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Calculates the sun azimuth & elevation angles at each time step in provided range

    :param lat: latitude, degrees
    :param lon: longitude, degrees
    :param step_in_minutes: the number of minutes between each step
    :param n: number of steps
    :param start_hr: hour of first day of the year
    :param steps: if given, calculate for the timesteps in the range, ignoring `start_hr` and `n`

    :returns: array of sun azimuth, array of sun elevation, datetime of each entry
    """
    if steps:
        start = datetime.datetime(2012, 1, 1, 0, 0, 0, 0, tzinfo=get_time_zone(lat, lon))
        date_generated = [start + datetime.timedelta(minutes=x * step_in_minutes) for x in steps]
    else:
        start = datetime.datetime(2012, 1, 1, start_hr, 0, 0, 0, tzinfo=get_time_zone(lat, lon))
        date_generated = [start + datetime.timedelta(minutes=x * step_in_minutes) for x in range(n)]

    azi_ang = np.zeros(len(date_generated))
    elv_ang = np.zeros(len(date_generated))
    for tt, date in enumerate(date_generated):
        azi_ang[tt] = get_azimuth(lat, lon, date)
        elv_ang[tt] = get_altitude(lat, lon, date)
    return azi_ang, elv_ang, date_generated


def blade_pos_of_rotated_ellipse(radius_x: float,
                                 radius_y: float,
                                 rotation_theta: Union[float, np.ndarray],
                                 blade_theta: Union[float, np.ndarray],
                                 center_x: float,
                                 center_y: float
                                 ) -> Tuple[float, float]:
    """
    Parametric equation for rotated ellipse

    :param radius_x: radius of ellipse along x-axis
    :param radius_y: radius of ellipse along y-axis
    :param rotation_theta: rotation of ellipse in radians
    :param blade_theta: angle of blade in radians
    :param center_x: ellipse center x coordinate
    :param center_y: ellipse center y coordinate
    :returns: (x, y) coordinate of the blade tip along rotated ellipse
    """
    x = radius_x * np.cos(blade_theta) * np.cos(rotation_theta) - \
        radius_y * np.sin(blade_theta) * np.sin(rotation_theta) + center_x
    y = radius_x * np.cos(blade_theta) * np.sin(rotation_theta) + \
        radius_y * np.sin(blade_theta) * np.cos(rotation_theta) + center_y
    return x, y


def get_turbine_shadow_polygons(blade_length: float,
                                blade_angle: Optional[float],
                                azi_ang: float,
                                elv_ang: float,
                                wind_dir,
                                tower_shadow: bool = True,
                                tower_height: Optional[float] = None
                                ) -> Tuple[Union[None, Polygon, MultiPolygon], float]:
    """
    Calculates the (x, y) coordinates of a wind turbine's shadow, which depends on the sun azimuth and elevation.

    The dimensions of the tower and blades are in fixed ratios to the blade_length. The blade angle is the degrees from
    z-axis, whereas the wind direction is where the turbine is pointing towards (if None, north is assumed).

    In spherical coordinates, blade angle is phi and wind direction is theta, with 0 at north, moving clockwise.

    The output shadow polygon is relative to the turbine located at (0, 0).

    :param blade_length: meters, radius in spherical coords
    :param blade_angle: degrees from z-axis, or None to use ellipse as swept area
    :param azi_ang: azimuth degrees, clockwise from north as 0
    :param elv_ang: elevation degrees, from x-y plane as 0
    :param wind_dir: degrees from north, clockwise, determines which direction rotor is facing
    :param tower_shadow: if false, do not include the tower's shadow
    :returns: (shadow polygon, shadow angle from north) if shadow exists, otherwise (None, None)
    """
    # "Shadow analysis of wind turbines for dual use of land for combined wind and solar photovoltaic power generation":
    # the average tower_height=2.5R; average tower_width=R/16; average blade_width=R/16
    blade_width = blade_length / 16
    if tower_height is None:
        tower_height = 2.5 * blade_length
    tower_width = blade_width

    # get shadow info
    sun_elv_rad = np.radians(elv_ang)
    tan_elv_inv = np.tan(sun_elv_rad) ** -1

    shadow_ang = azi_ang - 180.0
    if not wind_dir:
        wind_dir = 0
    if elv_ang <= 0.0:
        shadow_ang = np.nan
    if shadow_ang < 0.0:
        shadow_ang += 360.0

    shadow_tower_length = tower_height * tan_elv_inv
    if shadow_tower_length <= 0.0:
        shadow_tower_length = np.nan

    theta = np.radians(shadow_ang)
    if np.isnan(shadow_tower_length) or np.isnan(theta):
        return None, None

    shadow_length_blade_top = (tower_height + blade_length) * tan_elv_inv
    shadow_length_blade_bottom = (tower_height - blade_length) * tan_elv_inv
    shadow_height_blade = shadow_length_blade_top - shadow_length_blade_bottom
    shadow_width_blade = blade_length * abs(np.cos(np.radians(shadow_ang - wind_dir)))

    # calculate the tower shadow position
    tower_dx = tower_width / 2.0
    tower_dy = shadow_tower_length

    theta_left = np.radians(shadow_ang - 90)
    theta_right = np.radians(shadow_ang + 90)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    base_left_x, base_left_y = tower_dx * np.sin(theta_left), tower_dx * np.cos(theta_left)
    base_rght_x, base_rght_y = tower_dx * np.sin(theta_right), tower_dx * np.cos(theta_right)
    top_rght_x, top_rght_y = tower_dy * sin_theta + base_rght_x, tower_dy * cos_theta + base_rght_y
    top_left_x, top_left_y = tower_dy * sin_theta + base_left_x, tower_dy * cos_theta + base_left_y

    if tower_shadow:
        turbine_shadow = Polygon(((base_left_x, base_left_y),
                                  (base_rght_x, base_rght_y),
                                  (top_rght_x, top_rght_y),
                                  (top_left_x, top_left_y)))
    else:
        turbine_shadow = Polygon()

    # calculate the blade shadows of swept area using parametric eq of general ellipse
    radius_x = shadow_width_blade
    radius_y = shadow_height_blade / 2
    center_x = tower_dy * sin_theta
    center_y = tower_dy * cos_theta

    rot_ang = 360 - shadow_ang + 90
    rotation_theta = np.radians(rot_ang)

    if blade_angle is None:
        degs = np.linspace(0, 2 * np.pi, 50)
        x, y = blade_pos_of_rotated_ellipse(radius_y, radius_x, rotation_theta, degs, center_x, center_y)
        turbine_shadow = unary_union([turbine_shadow, Polygon(zip(x, y))])
    else:
        turbine_blade_angles = (blade_angle, blade_angle + 120, blade_angle - 120)

        for blade_angle in turbine_blade_angles:
            blade_theta = np.radians(blade_angle - 90)
            x, y = blade_pos_of_rotated_ellipse(radius_y, radius_x, rotation_theta, blade_theta, center_x, center_y)

            blade_1_dr = np.radians(blade_angle + 90)
            blade_2_dr = np.radians(blade_angle - 90)

            blade_tip_left_x, blade_tip_left_y = tower_dx * np.cos(blade_1_dr) + center_x, \
                                                 tower_dx * np.sin(blade_1_dr) + center_y
            blade_tip_rght_x, blade_tip_rght_y = tower_dx * np.cos(blade_2_dr) + center_x, \
                                                 tower_dx * np.sin(blade_2_dr) + center_y
            blade_base_rght_x, blade_base_rght_y = tower_dx * np.cos(blade_2_dr) + x, \
                                                   tower_dx * np.sin(blade_2_dr) + y
            blade_base_left_x, blade_base_left_y = tower_dx * np.cos(blade_1_dr) + x, \
                                                   tower_dx * np.sin(blade_1_dr) + y

            turbine_shadow = unary_union([turbine_shadow, Polygon(((blade_tip_left_x, blade_tip_left_y),
                                                                      (blade_tip_rght_x, blade_tip_rght_y),
                                                                      (blade_base_rght_x, blade_base_rght_y),
                                                                      (blade_base_left_x, blade_base_left_y)))])
    return turbine_shadow, shadow_ang


def get_turbine_shadows_timeseries(blade_length: float,
                                   steps: range,
                                   angles_per_step: int,
                                   azi_ang: Union[list, np.ndarray],
                                   elv_ang: Union[list, np.ndarray],
                                   wind_ang: Optional[list] = None,
                                   tower_shadow: bool = True
                                   ) -> List[List[Union[None, Polygon, MultiPolygon]]]:
    """
    Calculate turbine shadows for a number of equally-spaced blade angles per time step.
    Returns a list of turbine shadows per time step, where each entry has a shadow for each angle.

    :param blade_length: meters
    :param steps: which timesteps to calculate
    :param angles_per_step: number of blade angles per timestep
    :param elv_ang: array of elevation angles, degrees
    :param azi_ang: array of azimuth angles, degrees
    :param wind_ang: array of wind direction degrees with 0 as north, degrees
    :param tower_shadow: if false, do not include the tower's shadow

    :returns: list of turbine shadows per time step
    """
    if len(steps) != len(azi_ang) or len(steps) != len(elv_ang):
        raise ValueError("Timesteps provided in 'steps' not equal in length to azimuth and elevation arrays")

    turbine_shadows_per_timestep = []
    if angles_per_step is None:
        angles_range = (None,)
    else:
        step_to_angle = 120 / angles_per_step
        angles_range = [i * step_to_angle for i in range(angles_per_step)]

    for n, step in enumerate(steps):
        if elv_ang[n] < 0:
            turbine_shadows_per_timestep.append(None)
            continue
        shadows = []
        wind_dir = None if wind_ang is None else wind_ang[step]
        for angle in angles_range:
            turbine_shadow, shadow_ang = get_turbine_shadow_polygons(blade_length,
                                                                     angle,
                                                                     azi_ang=azi_ang[n],
                                                                     elv_ang=elv_ang[n],
                                                                     wind_dir=wind_dir,
                                                                     tower_shadow=tower_shadow)
            if turbine_shadow and shadow_ang:
                shadows.append(turbine_shadow)
        turbine_shadows_per_timestep.append(shadows)
    return turbine_shadows_per_timestep


def shadow_cast_over_panel(panel_x: float,
                           panel_y: float,
                           n_mod: int,
                           blade_length: float,
                           blade_angle: float,
                           azi_ang: float,
                           elv_ang: float,
                           wind_dir: float = None
                           ) -> Optional[Tuple[np.ndarray, Polygon]]:
    """
    Calculates which cells in a string of PV panels are shaded. The panel is located at a (panel_x, panel_y) distance
    from the turbine at (0, 0). Shadow shape depends on the sun azimuth and elevation angle.

    The PV panel is assumed to be a 96-cell, 1.488 x 0.992 m panel with 12.4 x 12.4 cm cells, 12x8 cells with
    2, 4, and 2 columns of cells per diode for total of 3 substrings.

    Turbine dimensions depend on blade_length and shape of the blades depend on blade_angle and wind_dir-- see
    get_turbine_shadow_polygons for more details.

    :param panel_x: distance from turbine to bottom-left corner of panels
    :param panel_y: degrees from x-axis to bottom-left corner of panels
    :param n_mod: number of modules in a string ( n x 1 solar array)
    :param blade_length: meters, radius in spherical coords
    :param blade_angle: degrees from xv-plane, 90-inclination/theta in spherical coords
    :param azi_ang: azimuth degrees
    :param elv_ang: elevation degrees
    :param wind_dir: degrees from north, clockwise, determines which dir rotor is facing, azimuth/phi in spherical coord

    :returns: grid of cells where 1 means shaded, turbine shadow polygon
    """

    turbine_shadow: Polygon = Polygon()
    turbine_shadow, shadow_ang = get_turbine_shadow_polygons(blade_length, blade_angle, azi_ang, elv_ang, wind_dir)

    if not turbine_shadow:
        return None

    panel_height = cell_len * cell_cols
    panel_width = cell_len * cell_rows * n_mod

    # generate a mesh of the pv panel assuming turbine at (0, 0)
    x = np.linspace(panel_x, panel_x + panel_width, num=cell_rows * n_mod + 1)
    y = np.linspace(panel_y, panel_y + panel_height, num=cell_cols + 1)
    xv, yv = np.meshgrid(x, y, indexing='xy')
    xc = 0.5 * (x[1:] + x[:-1])
    yc = 0.5 * (y[1:] + y[:-1])
    xvc, yvc = np.meshgrid(xc, yc, indexing='xy')
    shadow = np.zeros(np.shape(xvc))

    for i in range(len(xvc)):
        for j in range(len(xvc[0])):
            point = Point(xvc[i, j], yvc[i, j])
            if turbine_shadow.contains(point):
                shadow[i, j] = 1

    return shadow, turbine_shadow


def create_turbines_in_grid(dx: float,
                            dy: float,
                            theta: Union[float, np.ndarray],
                            n_turbines_per_side: int
                            ) -> Tuple[list, Polygon]:
    """
    Sets up turbines in a grid. Returns a list of the turbine positions and a Polygon including them.

    :param dx: x distance between turbines in grid
    :param dy: y distance
    :param theta: rotation of grid
    :param n_turbines_per_side:
    :return:
    """
    turb_pos = []
    dx_y_offset = dx * np.sin(theta)
    dx_x_offset = dx * np.cos(theta)

    first_row = [(dx_x_offset * i, dx_y_offset * i) for i in range(n_turbines_per_side)]
    turb_pos += first_row

    dy_y_offset = dy * np.sin(theta)
    dy_x_offset = dy * np.cos(theta)
    for r in range(1, n_turbines_per_side):
        turb_pos += [(i - dy_y_offset * r, j + dy_x_offset * r) for i, j in first_row]
    min_x, min_y = np.min(turb_pos, axis=0)
    max_x, max_y = np.max(turb_pos, axis=0)
    site = Polygon(((min_x, min_y),
                    (max_x, min_y),
                    (max_x, max_y),
                    (min_x, max_y)))
    return turb_pos, site


def get_turbine_grid_shadow(shadow_polygons: Union[MultiPolygon, None],
                            turb_pos: list
                            ) -> Optional[List[Union[Polygon, MultiPolygon]]]:
    """
    Calculate shadow polygons for each step in simulation for each turbine in the grid

    :return: list with dimension [step_per_hour, angles_per_step]
    """
    if not shadow_polygons:
        return None
    turbine_grid_shadows = []
    for shadow in shadow_polygons:
        all_turbine_shadows = []
        for t, offset in enumerate(turb_pos):
            translated_shadow = translate(shadow, xoff=offset[0], yoff=offset[1])
            all_turbine_shadows.append(translated_shadow)
        turbine_grid_shadows.append(unary_union(all_turbine_shadows))
    return turbine_grid_shadows


def create_module_cells_mesh(mod_x: float,
                             mod_y: float,
                             mod_width: float,
                             mod_height: float,
                             n_module: int):
    """
    For a string of PV modules, create an array of meshgrids having a point for each cell.

    :param mod_x: x coordinate of corner of panel
    :param mod_y: y coordinate
    :param mod_width: single module's width
    :param mod_height: module's height
    :param n_module: number of modules per string

    :return: n_module array of meshgrids
    """
    module_meshes = []
    for i in range(n_module):
        x = np.linspace(mod_x, mod_x + mod_width, num=cell_rows + 1)
        y = np.linspace(mod_y + mod_height * i, mod_y + mod_height * (i + 1), num=cell_cols + 1)
        xc = 0.5 * (x[1:] + x[:-1])
        yc = 0.5 * (y[1:] + y[:-1])
        module_meshes += [np.meshgrid(xc, yc, indexing='xy')]
    return module_meshes


def shadow_over_module_cells(module_mesh: np.ndarray,
                             turbine_shadow: Union[Polygon, MultiPolygon]):
    """
    For a meshgrid where each point is a cell in a PV module, identify which cells are in the turbine_shadow.

    :param module_mesh: meshgrid
    :param turbine_shadow: polygon

    :return: 2-D array with same coordinates as the PV module with values 0 (unshaded) or 1 (shaded)
    """
    x = module_mesh[0]
    y = module_mesh[1]
    shadow = np.zeros(np.shape(x))
    for i in range(len(x)):
        for j in range(len(x[0])):
            point = Point(x[i, j], y[i, j])
            if turbine_shadow.contains(point):
                shadow[i, j] = 1
    return shadow


def create_pv_string_points(x_coord: float,
                            y_coord: float,
                            mod_width: float,
                            mod_height: float,
                            string_width: float,
                            string_height: float
                            ) -> Tuple[Polygon, np.ndarray]:
    """

    :param x_coord:
    :param y_coord:
    :param mod_width:
    :param mod_height:
    :param string_width:
    :param string_height:

    :return:
    """
    pts = ((x_coord, y_coord),
           (x_coord + string_width, y_coord),
           (x_coord + string_width, y_coord + string_height),
           (x_coord, y_coord + string_height))
    module = Polygon(pts)

    xs_string = np.arange(mod_width / 2, mod_width, mod_width)
    ys_string = np.arange(mod_height / 2 + y_coord, y_coord + string_height, mod_height)

    xxs, yys = np.meshgrid(xs_string, ys_string, sparse=True)
    string_points = MultiPoint(np.transpose([np.tile(xs_string, len(ys_string)),
                                             np.repeat(yys, len(xs_string))]))
    return module, string_points
