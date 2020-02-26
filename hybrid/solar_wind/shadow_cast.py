import multiprocessing as mp
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import cm
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union
from pysolar.solar import *
import datetime
from pytz.reference import Mountain
from scipy import interpolate

cell_len = 0.124
cell_rows = 12
cell_cols = 8


def get_sun_pos(lat, lon, n=8760, step_secs=3600, start_hr=0):
    """
    Calculates the sun azimuth & elevation angles at each time step in provided range
    :param lat: latitude, degrees
    :param lon: longitude, degrees
    :param n: number of periods
    :return: array of sun azimuth, array of sun elevation
    """
    start = datetime.datetime(2012, 1, 1, start_hr, 0, 0, 0, tzinfo=Mountain)
    date_generated = [start + datetime.timedelta(seconds=x * step_secs) for x in range(0, n)]

    # initialize_variables
    azi_ang = np.zeros((n))
    elv_ang = np.zeros((n))

    # loop over all times and calculate elevation angle
    for tt, date in enumerate(date_generated):
        azi_ang[tt] = get_azimuth(lat, lon, date)
        elv_ang[tt] = get_altitude(lat, lon, date)
    return azi_ang, elv_ang


def blade_pos_of_rotated_ellipse(r_vert, r_hor, rotation_theta, blade_theta, Cx, Cy):
    x = r_vert * np.cos(blade_theta) * np.cos(rotation_theta) - r_hor * np.sin(blade_theta) * np.sin(rotation_theta) + Cx
    y = r_vert * np.cos(blade_theta) * np.sin(rotation_theta) + r_hor * np.sin(blade_theta) * np.cos(rotation_theta) + Cy
    return x, y


def get_turbine_shadow_polygons(blade_length, blade_angle, azi_ang, elv_ang, wind_dir) -> [Polygon, float]:
    """
    Calculates the (x, y) coordinates of the corners of the blade shadow assuming turbine at (0, 0)
    :param blade_length: meters, radius in spherical coords
    :param blade_angle: degrees from xy-plane, 90-inclination/theta in spherical coords
    :param azi_ang: azimuth degrees
    :param elv_ang: elevation degrees
    :param wind_dir: degrees from north, clockwise, determines which dir rotor is facing, azimuth/phi in spherical coords
    :return: dictionary with coordinates of 'bottom_left', 'bottom_right', 'top_left', and 'top_right'
    """
    blade_width = blade_length / 16
    tower_height = 2.5*blade_length
    tower_width = blade_width

    # get shadow info
    sun_elv_rad = np.radians(elv_ang)
    tan_elv_inv = np.tan(sun_elv_rad) ** -1

    shadow_ang = azi_ang - 180.0
    if not wind_dir:
        wind_dir = shadow_ang
    if elv_ang <= 0.0:
        shadow_ang = np.nan
    if shadow_ang < 0.0:
        shadow_ang += 360.0

    wind_dir -= 180
    if wind_dir < 0:
        wind_dir += 360

    # shadow_length = (tower_height + blade_length * np.sin(np.radians(blade_angle))) * np.tan(sun_elv_rad) ** -1
    shadow_tower_length = tower_height * tan_elv_inv
    if shadow_tower_length <= 0.0:
        shadow_tower_length = np.nan

    theta = np.radians(shadow_ang)
    if np.isnan(shadow_tower_length) or np.isnan(theta):
        return None

    shadow_length_blade_top = (tower_height + blade_length) * tan_elv_inv
    shadow_length_blade_bottom = (tower_height - blade_length) * tan_elv_inv
    shadow_height_blade = shadow_length_blade_top - shadow_length_blade_bottom
    shadow_width_blade = blade_length * abs(np.cos(shadow_ang - wind_dir))

    # calculate the blade shadow position
    z = tower_height
    htw = blade_width / 2.0
    D = shadow_tower_length

    theta_left = np.radians(shadow_ang - 90)
    theta_right = np.radians(shadow_ang + 90)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    tower_left_xs, tower_left_ys = htw * np.sin(theta_left), htw * np.cos(theta_left)
    tower_rght_xs, tower_rght_ys = htw * np.sin(theta_right), htw * np.cos(theta_right)
    tower_rght_xe, tower_rght_ye = D * sin_theta + tower_rght_xs, D * cos_theta + tower_rght_ys
    tower_left_xe, tower_left_ye = D * sin_theta + tower_left_xs, D * cos_theta + tower_left_ys

    turbine_shadow = Polygon(((tower_left_xs, tower_left_ys),
                            (tower_rght_xs, tower_rght_ys),
                            (tower_rght_xe, tower_rght_ye),
                            (tower_left_xe, tower_left_ye)))

    # blade tip position on shadow of swept area using parametric eq of general ellipse
    R_hor = shadow_width_blade
    R_vert = shadow_height_blade / 2
    Cx = D * sin_theta
    Cy = D * cos_theta

    rot_ang = 360 - shadow_ang + 90
    rotation_theta = np.radians(rot_ang)

    turbine_blade_angles = (blade_angle, blade_angle + 120, blade_angle - 120)

    for blade_angle in turbine_blade_angles:
        blade_theta = np.radians(blade_angle - 90)
        x, y = blade_pos_of_rotated_ellipse(R_vert, R_hor, rotation_theta, blade_theta, Cx, Cy)
        # angles = np.linspace(0, np.pi / 4, 10)
        # plt.clf()
        # plt.plot(x(angles), y(angles))
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.xlim((-90, -50))
        # plt.ylim((70,90))
        # plt.show()

        blade_theta_left = np.radians(blade_angle + 90)
        blade_theta_right = np.radians(blade_angle - 90)
        blade_left_xs, blade_left_ys = htw * np.cos(blade_theta_left) + Cx, htw * np.sin(blade_theta_left) + Cy
        blade_rght_xs, blade_rght_ys = htw * np.cos(blade_theta_right) + Cx, htw * np.sin(blade_theta_right) + Cy
        blade_rght_xe, blade_rght_ye = htw * np.cos(blade_theta_right) + x, htw * np.sin(blade_theta_right) + y
        blade_left_xe, blade_left_ye = htw * np.cos(blade_theta_left) + x, htw * np.sin(blade_theta_left) + y

        turbine_shadow = cascaded_union([turbine_shadow, Polygon(((blade_left_xs, blade_left_ys),
                                                               (blade_rght_xs, blade_rght_ys),
                                                               (blade_rght_xe, blade_rght_ye),
                                                               (blade_left_xe, blade_left_ye)))])
    # if isinstance(turbine_shadow, Polygon):
    #     xb, yb = turbine_shadow.exterior.xy
    #     plt.plot(xb, yb, color='#6699cc', alpha=0.5)
    # else:
    #     for poly in turbine_shadow:
    #         xb, yb = poly.exterior.xy
    #         plt.plot(xb, yb, color='#6699cc', alpha=0.5)
    # plt.show()
    return turbine_shadow, shadow_ang


def shadow_cast_over_panel(panel_x, panel_y, n_mod, n_cols, blade_length, blade_angle, azi_ang, elv_ang, wind_dir=None, plot_obj=None):
    """
    Assumes a 96-cell 1.488 x 0.992 m panel with 12.4 x 12.4 cm cells, 12x8 cells with 2, 4, and 2 columns of cells per
    diode for total of 3 substrings
    :param panel_x: distance from turbine to bottom-left corner of panels
    :param panel_y: degrees from x-axis to bottom-left corner of panels
    :param n_mod: number of modules in a string ( n x 1 solar array)
    :param n_cols:
    :param blade_length: meters, radius in spherical coords
    :param blade_angle: degrees from xv-plane, 90-inclination/theta in spherical coords
    :param azi_ang: azimuth degrees
    :param elv_ang: elevation degrees
    :param wind_dir: degrees from north, clockwise, determines which dir rotor is facing, azimuth/phi in spherical coords
    :param plot_obj:
    :return:
    """

    turbine_shadow: Polygon = Polygon()
    turbine_shadow, shadow_ang = get_turbine_shadow_polygons(blade_length, blade_angle, azi_ang, elv_ang, wind_dir)

    if not turbine_shadow:
        return None

    panel_height = cell_len * cell_cols * n_cols
    panel_width = cell_len * cell_rows * n_mod

    if plot_obj:
        xb, yb = turbine_shadow.exterior.xy
        plot_obj.set_data(xb, yb)

    # generate a mesh of the pv panel assuming turbine at (0, 0)
    # panel_x, panel_y = pv_r * np.cos(np.radians(pv_angle)), pv_r * np.sin(np.radians(pv_angle))
    x = np.linspace(panel_x, panel_x + panel_width, num=cell_rows * n_mod + 1)
    y = np.linspace(panel_y, panel_y + panel_height, num=cell_cols * n_cols + 1)
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

    return xv, yv, shadow


def create_module_cells_mesh(panel_x, panel_y, mod_width, mod_height, nrows):
    module_meshes = []
    for i in range(nrows):
        x = np.linspace(panel_x, panel_x + mod_width, num=cell_rows + 1)
        y = np.linspace(panel_y + mod_height * i, panel_y + mod_height * (i + 1), num=cell_cols + 1)
        xc = 0.5 * (x[1:] + x[:-1])
        yc = 0.5 * (y[1:] + y[:-1])
        module_meshes += [np.meshgrid(xc, yc, indexing='xy')]
    return module_meshes


def shadow_over_module_cells(module_mesh, turbine_shadow):
    x = module_mesh[0]
    y = module_mesh[1]
    shadow = np.zeros(np.shape(x))
    for i in range(len(x)):
        for j in range(len(x[0])):
            point = Point(x[i, j], y[i, j])
            if turbine_shadow.contains(point):
                shadow[i, j] = 1
    return shadow


def shadow_cast_swept_area(tower_height, tower_width, rotor_rad, shadow_info, azi_ang, elv_ang, x, y):
    """
    Calculates the percent incoming radiation around a turbine
    :param tower_height: in meters
    :param tower_width: in meters
    :param rotor_rad: in meters
    :param shadow_info: Dictionary containing 'tower_shadow_weight' and 'rotor_shadow_weight'
    :param azi_ang: azimuth angle
    :param elv_ang: elevation angle
    :param x: array of x-coordinates for mesh grid
    :param y: array of x-coordinates for mesh grid
    :return: 1000 x 1000 array of shadow percent or None when sun is down
    """
    shadow_length_tower = tower_height * np.tan(np.radians(elv_ang)) ** -1
    shadow_length_rotor_top = (tower_height + rotor_rad) * np.tan(np.radians(elv_ang)) ** -1
    shadow_length_rotor_bot = (tower_height - rotor_rad) * np.tan(np.radians(elv_ang)) ** -1

    if shadow_length_tower <= 0.0:
        shadow_length_tower = np.nan
    if shadow_length_rotor_top <= 0.0:
        shadow_length_rotor_top = np.nan
    if shadow_length_rotor_bot <= 0.0:
        shadow_length_rotor_bot = np.nan
    shadow_ang = azi_ang - 180.0
    if elv_ang <= 0.0:
        shadow_ang = np.nan
    if shadow_ang < 0.0:
        shadow_ang += 360.0

    D = shadow_length_tower
    theta = np.radians(shadow_ang)

    if np.isnan(D) and np.isnan(theta):
        return None

    # generate a mesh assuming tower in center
    xc = 0.5 * (x[1:] + x[:-1])
    yc = 0.5 * (y[1:] + y[:-1])
    xyc, yxc = np.meshgrid(xc, yc, indexing='ij')
    shadow = np.zeros(np.shape(xyc))

    distance = np.sqrt(xyc ** 2 + yxc ** 2)
    angle = np.degrees(np.arctan(xyc / yxc))
    angle[yxc < 0.0] = angle[yxc < 0.0] + 180.0
    angle[angle < 0.0] = angle[angle < 0.0] + 360.0

    # calculate turbine shadow on grid
    htw = tower_width / 2.0
    # find the left and right edges of the tower shadow by adding/subtracting 90 degrees from the shadow angle
    tower_left_xs, tower_left_ys = htw * np.sin(np.radians(shadow_ang - 90.0)), htw * np.cos(
        np.radians(shadow_ang - 90.0))
    tower_rght_xs, tower_rght_ys = htw * np.sin(np.radians(shadow_ang+ 90.0)), htw * np.cos(
        np.radians(shadow_ang + 90.0))
    tower_left_xe, tower_left_ye = D * np.sin(theta) + tower_left_xs, D * np.cos(theta) + tower_left_ys
    tower_rght_xe, tower_rght_ye = D * np.sin(theta) + tower_rght_xs, D * np.cos(theta) + tower_rght_ys
    # Find the slopes & intercepts of these lines to find the cells that are between the two
    tower_left_slope = (tower_left_ys - tower_left_ye) / (tower_left_xs - tower_left_xe)
    tower_rght_slope = (tower_rght_ys - tower_rght_ye) / (tower_rght_xs - tower_rght_xe)
    tower_left_int = tower_left_ys - tower_left_slope * tower_left_xs
    tower_rght_int = tower_rght_ys - tower_rght_slope * tower_rght_xs
    tower_axis_slope = (tower_left_ys - tower_rght_ys) / (tower_left_xs - tower_rght_xs)
    # Make is so that the angle is 0 in the direction of the shadow... remove 180 > A > 270
    shadow_angle = angle - shadow_ang
    shadow_angle[shadow_angle < 0.0] += 360.0
    # Find points between the two lines...
    shadow[((yxc >= tower_left_slope * xyc + tower_left_int) & (yxc <= tower_rght_slope * xyc + tower_rght_int)) |
        ((yxc <= tower_left_slope * xyc + tower_left_int) & (yxc >= tower_rght_slope * xyc + tower_rght_int))] \
        = shadow_info['tower_shadow_weight']
    # Find points that are less than the shadow distance
    shadow[distance > shadow_length_tower] = 0.0
    # Find points in the direction of the shadow
    shadow[(shadow_angle > 90.0) & (shadow_angle < 270.0)] = 0.0

    # Define the ellipse!
    g_ell_center = (D * np.sin(theta), D * np.cos(theta))
    g_ell_height = rotor_rad
    g_ell_width = shadow_length_rotor_top - shadow_length_rotor_bot
    ell_angle = np.degrees(np.arctan2(D * np.cos(theta), D * np.sin(theta)))
    # Get the angles of the axes
    cos_angle = np.cos(np.radians(180. - ell_angle))
    sin_angle = np.sin(np.radians(180. - ell_angle))
    # Find the distance of each gridpoint from the ellipse
    exc = xyc - g_ell_center[0]
    eyc = yxc - g_ell_center[1]
    exct = exc * cos_angle - eyc * sin_angle
    eyct = exc * sin_angle + eyc * cos_angle
    # Get the radial distance away from ellipse: cutoff at r = 1
    rad_cc = (exct ** 2 / (g_ell_width / 2.) ** 2) + (eyct ** 2 / (g_ell_height / 2.) ** 2)

    shadow[rad_cc <= 1.0] = 1.0 - (1.0 - shadow[rad_cc <= 1.0]) * (1.0 - shadow_info['rotor_shadow_weight'])
    return shadow


def compute_centered_mask(start_t, n_ts, x, y, tower_height, tower_width, rotor_rad, shadow_info, azi, elv, poa_wght):
    """
    Calculates the percent shadow over a mesh grid over a specified time interval, weighted by
    :param start_t: timestep at which to start
    :param n_ts: number of timesteps in interval
    :param x: array of x coordinates for mesh
    :param y: array of y coordinates for mesh
    :param tower_height: meters
    :param tower_width: meters
    :param rotor_rad: meters
    :param shadow_info: Dictionary containing
            tower_shadow_weight: how much incoming radiation is blocked by tower
            rotor_shadow_weight: how much of the swept area do the blades cover
    :param azi: array of azimuth angles
    :param elv: array of elevation angles
    :param poa_wght: array of normalized poa irradiance to use as weights
    :return:
    """
    mesh_x, mesh_y = len(x) - 1, len(y) - 1
    avg_shadow_mp = np.zeros((mesh_x, mesh_y))

    for timestep in range(start_t, start_t + n_ts):
        centered_mask = shadow_cast_swept_area(tower_height, tower_width, rotor_rad, shadow_info, azi[timestep], elv[timestep], x, y)
        if centered_mask is None:
            continue
        avg_shadow_mp += centered_mask * poa_wght[timestep]

    return avg_shadow_mp


def weighted_avg_masks_by_poa(turb_info, shadow_info, sun_info, lxy):
    """
    Calculates the average percent incoming radiation over a mesh grid, weighted by plane-of-array irradiance
    :param turb_info:
        Dictionary containing:
            tower_height: in meters
            tower_width: in meters
            rotor_rad: in meters
    :param shadow_info:
        Dictionary containing:
            tower_shadow_weight: how much incoming radiation is blocked by tower
            rotor_shadow_weight: how much of the swept area do the blades cover
    :param sun_info:
        Dictionary containing:
            azi_ang: array of sun azimuth angles
            elv_ang: array of sun elevation angles
            irrad: array of irradiance by which to weight to shadows
    :param lxy:
        Tuple of x and y length of mesh grid
    :return: a single 2D array of shadow percent
    """

    if not all([i in turb_info for i in ['tower_height', 'tower_width', 'rotor_rad']]):
        print("Error: turb_info must have tower_height, tower_width and rotor_rad entries")
    tower_height = turb_info['tower_height']
    tower_width = turb_info['tower_width']
    rotor_rad = turb_info['rotor_rad']

    if not all([i in sun_info for i in ['azi_ang', 'elv_ang', 'irrad']]):
        print("Error: sun_info must have azi_ang, elv_ang and irrad entries")
    azi_ang = sun_info['azi_ang']
    elv_ang = sun_info['elv_ang']
    irrad = sun_info['irrad']
    assert(len(azi_ang) == len(elv_ang) == len(irrad))
    nt = len(irrad)

    # calculate poa from DNI and solar azi & zenith, assuming array tilt=40, azim=180
    sin_tilt = np.sin(40.0 * np.pi / 180.)
    cos_tilt = np.cos(40.0 * np.pi / 180.)
    zen_rad = (90.0 - elv_ang) * np.pi / 180.
    cos_aoi = np.cos(zen_rad) * cos_tilt + np.sin(zen_rad) * sin_tilt * np.cos((azi_ang - 180.0) * np.pi / 180.)
    poa_beam = irrad * cos_aoi

    weight_poa = np.array(irrad) / sum(poa_beam)

    dxy = (1, 1)
    x = np.arange(0, lxy[0] + 0.1, dxy[0]) - lxy[0] / 2.0
    y = np.arange(0, lxy[1] + 0.1, dxy[1])

    n_proc = int(mp.cpu_count() - 2)
    # n_proc = 1

    mesh_nx, mesh_ny = len(x) - 1, len(y) - 1
    avg_shadow = np.zeros((mesh_nx, mesh_ny))

    pool = mp.Pool(processes=n_proc)
    start_time = 0
    timestep_interval = int(nt / n_proc)
    for i in range(n_proc):
        if i == n_proc - 1:
            timestep_interval = nt - start_time
        result = pool.apply_async(compute_centered_mask, args=(start_time, timestep_interval, x, y, tower_height,
                                                                tower_width, rotor_rad, shadow_info, azi_ang, elv_ang,
                                                                weight_poa))
        start_time = start_time + timestep_interval
        avg_shadow += result.get()

    return avg_shadow


def turbines_shadow_cast_avg(turb_info, site_dim, shadow_info, sun_info):
    """
    Calculates the percent incoming radiation left on a site after considering turbine shadows
    :param turb_info:
        Dictionary containing:
            x: ordered array of turbine x coordinates
            y: ordered array of turbine y coordinates
            tower_height: in meters
            tower_width: in meters
            rotor_rad: in meters
    :param site_dim:
        Tuple containing max height and max width of site
    :param shadow_info:
        Dictionary containing:
            tower_shadow_weight: how much incoming radiation is blocked by tower
            rotor_shadow_weight: how much of the swept area do the blades cover
    :param sun_info:
        Dictionary containing:
            azi_ang: array of sun azimuth angles
            elv_ang: array of sun elevation angles
            irrad: array of irradiance by which to weight to shadows
    :param dxy:
        Tuple of x and y step in mesh grid

    :return:
        A mesh grid containing the percent incoming radiation on each cell
    """

    if 'azi_ang' not in sun_info:
        print("sun_info input requires azi_ang entry")
    nt = len(sun_info['azi_ang'])

    # generate a mesh of the entire site
    site_x, site_y = int(site_dim[0]), int(site_dim[1])
    lxy = (int(site_x * 2), site_y)
    dxy = (1, 1)

    x = np.arange(0, site_x, dxy[0])
    xc = 0.5 * (x[1:] + x[:-1])
    y = np.arange(0, site_y, dxy[1])
    yc = 0.5 * (y[1:] + y[:-1])
    xy, yx = np.meshgrid(xc, yc, indexing='ij')

    avg_shadow = weighted_avg_masks_by_poa(turb_info, shadow_info, sun_info, dxy, lxy)

    # l_meshx, l_meshy = np.meshgrid( np.arange(0, lxy[0]), np.arange(0, lxy[1]), indexing='ij')
    # plt.pcolormesh(l_meshx, l_meshy, avg_mask, norm=Normalize(0.0, 1.0), cmap=cm.gray)
    # plt.show()

    # superpose shadow mask from all turbines
    superposed_mask = np.zeros((site_x, site_y))
    t_x, t_y = turb_info['x'], turb_info['y']
    n_turb = len(t_x)

    for i in range(lxy[0]):
        for j in range(lxy[1]):
            if avg_shadow[i, j] == 1:
                continue
            for turb in range(n_turb):
                x = t_x[turb] - lxy[0]/2
                y = t_y[turb]
                pos_x, pos_y = int(x + i), int(y + j)
                if 0 <= pos_x < site_x and 0 <= pos_y < site_y:
                    superposed_mask[pos_x, pos_y] = 1 - (1 - avg_shadow[i, j]) * (1 - superposed_mask[pos_x, pos_y])

    superposed_mask = 1.0 - superposed_mask
    # l_meshx, l_meshy = np.meshgrid( np.arange(0, site_x), np.arange(0, site_y), indexing='ij')
    # plt.pcolormesh(l_meshx, l_meshy, superposed_mask, norm=Normalize(0.0, 1.0), cmap=cm.gray)
    # plt.show()

    plt.tick_params(labelsize=12)
    shdwplt = plt.pcolormesh(xy, yx, superposed_mask, cmap=cm.gray)
    # plt.colorbar()
    # plt.plot([0, 0], [0, tower_height / 2.0], c='k')
    # plt.scatter(0, tower_height / 2.0, color='k', marker='2', s=300)
    plt.ylabel('South-North [m]', size=14)
    plt.xlabel('West-East [m]', size=14)
    # plt.scatter(D*np.sin(theta), D*np.cos(theta),c='k')
    plt.xlim((int(-site_x/10), site_x + int(site_x/10)))
    plt.ylim((int(-site_y/10), site_y + int(site_y/10)))
    plt.title('Turbine shading weighted by irradiance for ' + str(nt) + ' hours')
    plt.savefig('weighted_turbine_shading_' + str(nt) + '.png', dpi=900)

    # plt.show()

    return superposed_mask


def get_unshaded_box(shadow_mask, threshold=0.0):
    """
    Calculates the size of the smallest box that encloses all the shading above a threshold ratio,
    where the box is given as length east, length west and length north, measured from the turbine position
    :param shadow_mask: shadow percent on mesh grid with dx, dy = (1, 1)
    :param threshold: float between 0 (no shading) and 1 (completely shaded)
    :return: len_east, len_west, len_north
    """
    mask_dim = np.shape(shadow_mask)
    mask = np.zeros(mask_dim)
    mask[shadow_mask > threshold] = 1.0

    mask_indices = mask.copy()
    for i in range(mask_dim[0]):
        for j in range(mask_dim[1]):
            mask_indices[i, j] *= j
    len_north = np.max(mask_indices)
    arg_max = np.argmax(mask_indices, axis=1)

    min_west = 0
    while min_west < mask_dim[0]/2 and arg_max[min_west] == 0:
        min_west += 1
    len_west = mask_dim[0]/2 - min_west

    min_east = mask_dim[0] - 1
    while min_east > 0 and arg_max[min_east] == 0:
        min_east -= 1
    len_east = min_east - mask_dim[0]/2

    return len_east, len_west, len_north, mask

data_intp_fx_east = None
data_intp_fx_west = None
data_intp_fx_north = None

def get_unshaded_box_from_data(radius, threshold):
    """
    Calculates the size of the smallest box that encloses all the shading above a threshold ratio by interpolating
    from prior calculations of shadow_masks as found in data/wind_turb_fit.py.
    Interpolation functions saved in global variables since they only need to be calculated once
    :param radius: meters
    :param threshold: float between 0 (no shading) and 1 (completely shaded)
    :return: len_east, len_west, len_north
    """

    x_radius = [30, 40, 50, 60, 70, 80]
    y_threshold = [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007]
    east_len_per_radius_per_threshold = [[801.0, 1069.0, 1336.0, 1593.0, 1771.0, 1793.0],
                                         [476.0, 645.0, 799.0, 955.0, 1128.0, 1282.0],
                                         [414.0, 553.0, 691.0, 830.0, 968.0, 1107.0],
                                         [365.0, 493.0, 616.0, 739.0, 863.0, 986.0],
                                         [215.0, 290.0, 362.0, 422.0, 509.0, 581.0],
                                         [190.0, 253.0, 317.0, 380.0, 444.0, 507.0],
                                         [170.0, 227.0, 284.0, 344.0, 404.0, 464.0],
                                         [163.0, 222.0, 275.0, 333.0, 389.0, 445.0]]

    west_len_per_radius_per_threshold = [[1794.0, 1794.0, 1794.0, 1794.0, 1794.0, 1794.0],
                                         [766.0, 1021.0, 1276.0, 1527.0, 1571.0, 1742.0],
                                         [618.0, 827.0, 1033.0, 1242.0, 1448.0, 1518.0],
                                         [459.0, 613.0, 776.0, 927.0, 1082.0, 1234.0],
                                         [421.0, 562.0, 703.0, 844.0, 988.0, 1129.0],
                                         [321.0, 427.0, 536.0, 642.0, 748.0, 854.0],
                                         [273.0, 364.0, 456.0, 548.0, 638.0, 731.0],
                                         [251.0, 337.0, 423.0, 515.0, 601.0, 687.0]]

    north_len_per_radius_per_threshold = [[926.0, 926.0, 1079.0, 1083.0, 1083.0, 1083.0],
                                          [550.0, 732.0, 917.0, 1083.0, 1083.0, 1083.0],
                                          [452.0, 605.0, 756.0, 909.0, 1060.0, 1083.0],
                                          [354.0, 473.0, 592.0, 715.0, 834.0, 953.0],
                                          [325.0, 434.0, 592.0, 652.0, 763.0, 872.0],
                                          [247.0, 329.0, 413.0, 495.0, 577.0, 659.0],
                                          [217.0, 291.0, 363.0, 436.0, 507.0, 582.0],
                                          [209.0, 278.0, 348.0, 418.0, 488.0, 558.0]]

    global data_intp_fx_west, data_intp_fx_east, data_intp_fx_north
    if not data_intp_fx_east:
        data_intp_fx_east = interpolate.interp2d(x_radius, y_threshold, east_len_per_radius_per_threshold, kind='linear')
    if not data_intp_fx_west:
        data_intp_fx_west = interpolate.interp2d(x_radius, y_threshold, west_len_per_radius_per_threshold, kind='linear')
    if not data_intp_fx_north:
        data_intp_fx_north = interpolate.interp2d(x_radius, y_threshold, north_len_per_radius_per_threshold, kind='linear')

    return data_intp_fx_east(radius, threshold), data_intp_fx_west(radius, threshold), data_intp_fx_north(radius, threshold)


def get_unshaded_areas_on_site(t_x: list, t_y: list, radius: float, threshold: float, plot_bool: bool = False):
    """
    Calculates which cells on a mesh grid are either outside the site boundaries or within a turbine's shadow box
    :param t_x: ordered array of turbine x coordinates
    :param t_y: ordered array of turbine y coordinates
    :param radius: meters
    :param threshold: float between 0 (no shading) and 1 (completely shaded)
    :param plot_bool: True/False
    :return: Polygon with shaded interior
    """
    len_east, len_west, len_north = get_unshaded_box_from_data(radius, threshold)

    # superpose shadow box from single turbine onto all the turbines
    nTurbs = len(t_x)

    shaded_regions = []
    for i in range(nTurbs):
        shadow = Polygon(((t_x[i] - len_west, t_y[i]),
                          (t_x[i] + len_east, t_y[i]),
                          (t_x[i] + len_east, t_y[i] + len_north),
                          (t_x[i] - len_west, t_y[i] + len_north)))
        shaded_regions.append(shadow)

    if plot_bool:
        for polygon in list(shaded_regions):
            xs, ys = polygon.exterior.xy
            plt.plot(xs, ys)
        # plt.savefig("unshaded_areas_on_site")
        plt.show()

    return shaded_regions
