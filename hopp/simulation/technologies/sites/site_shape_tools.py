from shapely.geometry import Polygon, MultiPolygon, box
import numpy as np
import pandas as pd
from hopp.tools.layout.plot_tools import plot_shape
import matplotlib.pyplot as plt

def calc_dist_between_two_points_cartesian(x1,y1,x2,y2):
    """Calculate the distance between two points.

    Args:
        x1 (np.ndarray | float): x coordinate of first point.
        y1 (np.ndarray | float): y coordinate of first point.
        x2 (np.ndarray | float): x coordinate of second point.
        y2 (np.ndarray | float): y coordinate of second point.

    Returns:
        np.ndarray | float: distance between two points
    """
    dx = np.abs(x2-x1)
    dy = np.abs(y2-y1)
    return np.sqrt((dx**2) + (dy**2))

def calc_angle_between_two_points_cartesian(x0, y0, x1, y1):
    """Calculate angle between two points.

    Args:
        x0 (np.ndarray | float): x coordinate of first point.
        y0 (np.ndarray | float): y coordinate of first point.
        x1 (np.ndarray | float): x coordinate of second point.
        y1 (np.ndarray | float): y coordinate of second point.

    Returns:
        np.ndarray | float: angle between two points (degrees)
    """
    dx = x1 - x0
    dy = y1 - y0
    angle_deg = np.rad2deg(np.arctan2(dx,dy))
    if isinstance(angle_deg, float): 
        return angle_deg if angle_deg >= 0 else angle_deg + 360 
    return np.where(angle_deg < 0, angle_deg + 360, angle_deg)


def check_site_verts(verts):
    """Check that vertices are valid and re-sort as needed.

    Args:
        verts (2D :obj:`numpy.ndarray`): vertices of site polygon. list of [x,y] coordinates in meters.

    Returns:
        numpy.ndarray: vertices ordered so that no linear rings may cross each other.
    """
    x_points, y_points = verts.T
    x0 = x_points.min()
    y0 = y_points.min()
    
    dx = 0 - x0 #dx is positive if x0 is negative
    dy = 0 - y0 #dy is positive if y0 is negative

    x_pos = x_points + dx
    y_pos = y_points + dy

    x_center = (max(x_pos) - min(x_pos))/2
    y_center = (max(y_pos) - min(y_pos))/2

    distances = calc_dist_between_two_points_cartesian(x_center,y_center,x_pos,y_pos)
    angles = calc_angle_between_two_points_cartesian(x_center,y_center,x_pos,y_pos)
    
    df = ( 
        pd.DataFrame({"x_pos": x_pos, "y_pos": y_pos, "distances": distances, "angles": angles}) 
        .sort_values(["angles", "distances"]) 
    ) 
    df.x_pos += dx 
    df.y_pos += dy 
    organized_verts = df[["x_pos", "y_pos"]].values
    return organized_verts

def make_square(area_m2, x0=0.0, y0=0.0):
    """Generate square polygon shape of specified area.

    Args:
        area_m2 (float): area of shape in square meters.
        x0 (float, Optional): left-most x coordinate of the shape. Defaults to 0.0.
        y0 (float, Optional): bottom-most x coordinate of the shape. Defaults to 0.0.

    Returns:
        2-element tuple containing

        - **poly** (:obj:`shapely.geometry.Polygon`): site boundary polygon
        - **vertices** (2D :obj:`numpy.ndarray`): vertices of site polygon. list of [x,y] coordinates in meters.
    """
    site_length = np.sqrt(area_m2)
    y1 = y0 + site_length
    x1 = x0 + site_length
    poly = box(x0, y0, x1, y1)
    vertices = np.array([[x0,x0], [x1,y0], [x1,y1], [x0,y1]])
    return poly, vertices

def make_rectangle(area_m2, aspect_ratio=1.5, x0=0.0, y0=0.0):
    """Generate rectangle polygon shape of specified area.

    Args:
        area_m2 (float): area of shape in square meters.
        aspect_ratio (float, Optional): ratio of width/height. Defaults to 1.5.
            (width corresponds to x coordinates, height corresponds to y coordinates)
        x0 (float, Optional): left-most x coordinate of the shape. Defaults to 0.0.
        y0 (float, Optional): bottom-most x coordinate of the shape. Defaults to 0.0.

    Returns:
        2-element tuple containing

        - **poly** (:obj:`shapely.geometry.Polygon`): site boundary polygon
        - **vertices** (2D :obj:`numpy.ndarray`): vertices of site polygon. list of [x,y] coordinates in meters.        
    """
    height = np.sqrt(area_m2/aspect_ratio)
    width = area_m2/height
    x1 = x0 + width
    y1 = y0 + height
    poly = box(x0, y0, x1, y1)
    vertices = np.array([[x0,x0], [x1,y0], [x1,y1], [x0,y1]])
    return poly,vertices

def make_circle(area_m2, deg_diff = 5.0, x0=0.0, y0=0.0):
    """Generate circle polygon shape of specified area.

    Args:
        area_m2 (float): area of shape in square meters.
        deg_diff (float | int): difference in degrees for generating boundary. default to 10.
            number of points generated is equal to ``360/deg_diff``
        x0 (float, Optional): left-most x coordinate of the shape. Defaults to 0.0.
        y0 (float, Optional): bottom-most x coordinate of the shape. Defaults to 0.0.

    Returns:
        2-element tuple containing

        - **poly** (:obj:`shapely.geometry.Polygon`): site boundary polygon
        - **vertices** (2D :obj:`numpy.ndarray`): vertices of site polygon. list of [x,y] coordinates in meters.
    """
    r = np.sqrt(area_m2/np.pi)
    dx = np.deg2rad(deg_diff)
    rads = np.arange(0, 2*np.pi, dx)
    x_coords = r*np.cos(rads)
    y_coords = r*np.sin(rads)

    x_points = x_coords 
    if any(x_coords < x0): 
        x_diff = x0 - x_coords 
        x_points += x_diff.max()

    y_points = y_coords 
    if any(y_coords < y0): 
        y_diff = y0 - y_coords 
        y_points += y_diff.max()
    
    vertices = np.vstack((x_points, y_points)).T
    poly = Polygon(vertices)

    return poly, vertices

def make_hexagon(area_m2, x0=0.0, y0=0.0):
    """Generate hexagon polygon shape of specified area.

    Args:
        area_m2 (float): area of shape in square meters.
        x0 (float, Optional): left-most x coordinate of the shape. Defaults to 0.0.
        y0 (float, Optional): bottom-most x coordinate of the shape. Defaults to 0.0.

    Returns:
        2-element tuple containing

        - **poly** (:obj:`shapely.geometry.Polygon`): site boundary polygon
        - **vertices** (2D :obj:`numpy.ndarray`): vertices of site polygon. list of [x,y] coordinates in meters.
    """
    s = np.sqrt(area_m2*(2/(3*np.sqrt(3))))
    rads = np.arange(0,2*np.pi,np.deg2rad(60))
    x_coords = s*np.cos(rads)
    y_coords = s*np.sin(rads)

    x_points = x_coords 
    if any(x_coords < x0): 
        x_diff = x0 - x_coords 
        x_points += x_diff.max()

    y_points = y_coords 
    if any(y_coords < y0): 
        y_diff = y0 - y_coords 
        y_points += y_diff.max()

    vertices = np.vstack((x_points, y_points)).T
    poly = Polygon(vertices)

    return poly, vertices

def rotate_shape(site_polygon, rotation_angle_deg):
    # in degrees where 0 is north, increasing clockwise
    # 90 degrees is east, 180 degrees is south, 270 degrees is west
    # get center points
    xc = site_polygon.centroid.x
    yc = site_polygon.centroid.y

    vertices = np.array(site_polygon.exterior.coords)

    # translate coordinates to have origin at polygon center
    xc_points, yc_points = (vertices - [xc, yc]).T

    theta = np.deg2rad(rotation_angle_deg)

    # rotate clockwise about the origin
    cos_theta = np.cos(theta) 
    sin_theta = np.sin(theta) 
    xr_points = (xc_points * cos_theta) + (yc_points * sin_theta)
    yr_points = (-1 * xc_points * sin_theta) + (yc_points * cos_theta)
    
    # translate points back to original coordinate reference system
    rotated_vertices = np.vstack((xr_points, yr_points)).T + [xc, yc]
    rotated_polygon = Polygon(rotated_vertices)
    
    return rotated_polygon, rotated_vertices


def plot_site_polygon(
    site_polygon,
    figure=None,
    axes=None,
    border_color=(0, 0, 0),
    alpha=0.95,
    linewidth=1.0
    ):
        bounds = site_polygon.bounds
        site_sw_bound = np.array([bounds[0], bounds[1]])
        site_ne_bound = np.array([bounds[2], bounds[3]])
        site_center = .5 * (site_sw_bound + site_ne_bound)
        max_delta = max(site_ne_bound - site_sw_bound)
        reach = (max_delta / 2) * 1.3
        min_plot_bound = site_center - reach
        max_plot_bound = site_center + reach

        if not figure and not axes:
            figure = plt.figure(1)
            axes = figure.add_subplot(111)

        axes.set_aspect('equal')
        axes.set(xlim=(min_plot_bound[0], max_plot_bound[0]), ylim=(min_plot_bound[1], max_plot_bound[1]))
        plot_shape(figure, axes, site_polygon, '--', color=border_color, alpha=alpha, linewidth=linewidth / 2)
        if isinstance(site_polygon, Polygon):
            shape = [site_polygon]
        elif isinstance(site_polygon, MultiPolygon):
            shape = site_polygon.geoms
        for geom in shape:    
            xs, ys = geom.exterior.xy    
            plt.fill(xs, ys, alpha=0.3, fc='g', ec='none')

        plt.tick_params(which='both', labelsize=15)
        plt.xlabel('x (m)', fontsize=15)
        plt.ylabel('y (m)', fontsize=15)

        return figure, axes