from pytest import approx
from shapely.geometry import Polygon
import hopp.simulation.technologies.sites.site_shape_tools as shape_tools
import numpy as np

def test_circle_area():
    area_m2 = 1e3
    polygon, vertices = shape_tools.make_circle(area_m2, deg_diff = 1.0)
    assert polygon.area == approx(area_m2,rel = 1e-3)

def test_square_area():
    area_m2 = 1e3
    polygon, vertices = shape_tools.make_square(area_m2)
    assert polygon.area == approx(area_m2,rel = 1e-3)

def test_rectangle_area():
    area_m2 = 1e3
    polygon, vertices = shape_tools.make_rectangle(area_m2)
    assert polygon.area == approx(area_m2,rel = 1e-3)

def test_hexagon_area():
    area_m2 = 1e3
    polygon, vertices = shape_tools.make_hexagon(area_m2)
    assert polygon.area == approx(area_m2,rel = 1e-3)

def test_circle_vertices_default():
    area_m2 = 1e3
    polygon, vertices = shape_tools.make_circle(area_m2)
    x_verts = [v[0] for v in vertices]
    y_verts = [v[1] for v in vertices]
    assert min(x_verts) == 0.0
    assert min(y_verts) == 0.0

def test_square_vertices_default():
    area_m2 = 1e3
    polygon, vertices = shape_tools.make_square(area_m2)
    x_verts = [v[0] for v in vertices]
    y_verts = [v[1] for v in vertices]
    assert len(vertices)==4
    assert min(x_verts) == 0.0
    assert min(y_verts) == 0.0

def test_rectangle_vertices_default():
    area_m2 = 1e3
    polygon, vertices = shape_tools.make_rectangle(area_m2)
    x_verts = [v[0] for v in vertices]
    y_verts = [v[1] for v in vertices]
    assert len(vertices)==4
    assert min(x_verts) == 0.0
    assert min(y_verts) == 0.0

def test_hexagon_vertices_default():
    area_m2 = 1e3
    polygon, vertices = shape_tools.make_hexagon(area_m2)
    x_verts = [v[0] for v in vertices]
    y_verts = [v[1] for v in vertices]
    assert len(vertices)==6
    assert min(x_verts) == 0.0
    assert min(y_verts) == 0.0


def test_circle_vertices_offset():
    area_m2 = 1e3
    x0 = 5.0
    y0 = -4.0
    polygon, vertices = shape_tools.make_circle(area_m2, x0 = x0, y0 = y0)
    x_verts = [v[0] for v in vertices]
    y_verts = [v[1] for v in vertices]
    assert min(x_verts) == x0
    assert min(y_verts) == y0

def test_square_vertices_offset():
    area_m2 = 1e3
    x0 = 5.0
    y0 = -4.0
    polygon, vertices = shape_tools.make_square(area_m2, x0 = x0, y0 = y0)
    x_verts = [v[0] for v in vertices]
    y_verts = [v[1] for v in vertices]
    assert len(vertices) == 4
    assert min(x_verts) == x0
    assert min(y_verts) == y0

def test_rectangle_vertices_offset():
    area_m2 = 1e3
    x0 = 5.0
    y0 = -4.0
    aspect_ratio = 1.5
    polygon, vertices = shape_tools.make_rectangle(area_m2, aspect_ratio = aspect_ratio, x0 = x0, y0 = y0)
    x_verts = [v[0] for v in vertices]
    y_verts = [v[1] for v in vertices]
    assert len(vertices) == 4
    assert min(x_verts) == x0
    assert min(y_verts) == y0

def test_hexagon_vertices_offset():
    area_m2 = 1e3
    x0 = 5.0
    y0 = -4.0
    polygon, vertices = shape_tools.make_hexagon(area_m2, x0 = x0, y0 = y0)
    x_verts = [v[0] for v in vertices]
    y_verts = [v[1] for v in vertices]
    assert len(vertices) == 6
    assert min(x_verts) == x0
    assert min(y_verts) == y0

def test_distance_between_points():
    x0 = 0.0
    y0 = 0.0
    dy = 5.0
    dx = 0.0
    x1 = x0 + dx
    y2 = y0 + dy
    distance = shape_tools.calc_dist_between_two_points_cartesian(x0,y0,x1,y2)
    assert distance == approx(dy,1e-3)

def test_angle_between_points():
    x0 = 0.0
    y0 = 0.0
    x1 = 1.0
    y1 = 1.0
    angle = shape_tools.calc_angle_between_two_points_cartesian(x0,y0,x1,y1)
    assert angle == approx(45.0,1e-3)

def test_sort_site_verts():
    invalid_x_points = [0.0,5.0,0.0,5.0]
    invalid_y_points = [0.0,0.0,5.0,5.0]
    invalid_verts = [[x,y] for x,y in zip(invalid_x_points,invalid_y_points)]
    invalid_verts = np.array(invalid_verts)
    valid_verts = shape_tools.check_site_verts(invalid_verts)
    valid_shape = Polygon(valid_verts)
    assert valid_shape.area == approx(25.0,1e-3)

def test_rotate_site_center_area():
    area_m2 = 1e3
    rotation_angle = 45
    polygon_original, vertices_original = shape_tools.make_hexagon(area_m2)
    rotated_polygon, rotated_vertices = shape_tools.rotate_shape(
        site_polygon = polygon_original, 
        rotation_angle_deg = rotation_angle
    )
    assert rotated_polygon.centroid.x == approx(polygon_original.centroid.x,abs = 1e-3)
    assert rotated_polygon.centroid.y == approx(polygon_original.centroid.y,abs = 1e-3)
    assert rotated_polygon.area == approx(polygon_original.area,abs = 1e-3)

def test_rotate_site_vertices():
    area_m2 = 1e3
    rotation_angle = 45
    polygon_original, vertices_original = shape_tools.make_square(area_m2)
    rotated_polygon, rotated_vertices = shape_tools.rotate_shape(
        site_polygon = polygon_original, 
        rotation_angle_deg = rotation_angle
    )
    assert rotated_polygon.exterior.xy[0][0] == approx(polygon_original.centroid.x,abs = 1e-3)
    assert rotated_polygon.exterior.xy[1][1] == approx(polygon_original.centroid.y,abs = 1e-3)
