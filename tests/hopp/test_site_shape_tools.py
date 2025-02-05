from pytest import approx
from shapely.geometry import Polygon
import hopp.simulation.technologies.sites.site_shape_tools as shape_tools

def test_circle_area():
    area_m2 = 1e3
    x0 = 0.0
    y0 = 0.0
    polygon, vertices = shape_tools.make_circle(area_m2, x0 = x0, y0 = y0)
    assert polygon.area == approx(area_m2,rel = 0.1)

def test_square_area():
    area_m2 = 1e3
    x0 = 0.0
    y0 = 0.0
    polygon, vertices = shape_tools.make_square(area_m2, x0 = x0, y0 = y0)
    assert polygon.area == approx(area_m2,rel = 0.1)

def test_rectangle_area():
    area_m2 = 1e3
    x0 = 0.0
    y0 = 0.0
    aspect_ratio = 1.5
    polygon, vertices = shape_tools.make_rectangle(area_m2, aspect_ratio = aspect_ratio, x0 = x0, y0 = y0)
    assert polygon.area == approx(area_m2,rel = 0.1)

def test_hexagon_area():
    area_m2 = 1e3
    x0 = 0.0
    y0 = 0.0
    polygon, vertices = shape_tools.make_hexagon(area_m2, x0 = x0, y0 = y0)
    assert polygon.area == approx(area_m2,rel = 0.1)

def test_circle_vertices_default():
    area_m2 = 1e3
    x0 = 0.0
    y0 = 0.0
    polygon, vertices = shape_tools.make_circle(area_m2, x0 = x0, y0 = y0)
    x_verts = [v[0] for v in vertices]
    y_verts = [v[1] for v in vertices]
    assert min(x_verts)==x0
    assert min(y_verts)==y0

def test_square_vertices_default():
    area_m2 = 1e3
    x0 = 0.0
    y0 = 0.0
    polygon, vertices = shape_tools.make_square(area_m2, x0 = x0, y0 = y0)
    x_verts = [v[0] for v in vertices]
    y_verts = [v[1] for v in vertices]
    assert len(vertices)==4
    assert min(x_verts)==x0
    assert min(y_verts)==y0

def test_rectangle_vertices_default():
    area_m2 = 1e3
    x0 = 0.0
    y0 = 0.0
    aspect_ratio = 1.5
    polygon, vertices = shape_tools.make_rectangle(area_m2, aspect_ratio = aspect_ratio, x0 = x0, y0 = y0)
    x_verts = [v[0] for v in vertices]
    y_verts = [v[1] for v in vertices]
    assert len(vertices)==4
    assert min(x_verts)==x0
    assert min(y_verts)==y0

def test_hexagon_vertices_default():
    area_m2 = 1e3
    x0 = 0.0
    y0 = 0.0
    polygon, vertices = shape_tools.make_hexagon(area_m2, x0 = x0, y0 = y0)
    x_verts = [v[0] for v in vertices]
    y_verts = [v[1] for v in vertices]
    assert len(vertices)==6
    assert min(x_verts)==x0
    assert min(y_verts)==y0


def test_circle_vertices_offset():
    area_m2 = 1e3
    x0 = 5.0
    y0 = -4.0
    polygon, vertices = shape_tools.make_circle(area_m2, x0 = x0, y0 = y0)
    x_verts = [v[0] for v in vertices]
    y_verts = [v[1] for v in vertices]
    assert min(x_verts)==x0
    assert min(y_verts)==y0

def test_square_vertices_offset():
    area_m2 = 1e3
    x0 = 5.0
    y0 = -4.0
    polygon, vertices = shape_tools.make_square(area_m2, x0 = x0, y0 = y0)
    x_verts = [v[0] for v in vertices]
    y_verts = [v[1] for v in vertices]
    assert len(vertices)==4
    assert min(x_verts)==x0
    assert min(y_verts)==y0

def test_rectangle_vertices_offset():
    area_m2 = 1e3
    x0 = 5.0
    y0 = -4.0
    aspect_ratio = 1.5
    polygon, vertices = shape_tools.make_rectangle(area_m2, aspect_ratio = aspect_ratio, x0 = x0, y0 = y0)
    x_verts = [v[0] for v in vertices]
    y_verts = [v[1] for v in vertices]
    assert len(vertices)==4
    assert min(x_verts)==x0
    assert min(y_verts)==y0

def test_hexagon_vertices_offset():
    area_m2 = 1e3
    x0 = 5.0
    y0 = -4.0
    polygon, vertices = shape_tools.make_hexagon(area_m2, x0 = x0, y0 = y0)
    x_verts = [v[0] for v in vertices]
    y_verts = [v[1] for v in vertices]
    assert len(vertices)==6
    assert min(x_verts)==x0
    assert min(y_verts)==y0

def test_distance_between_points():
    x0 = 0.0
    y0 = 0.0
    dy = 5.0
    dx = 0.0
    x1 = x0 + dx
    y2 = y0 + dy
    distance = shape_tools.calc_dist_between_two_points_cartesian(x0,y0,x1,y2)
    assert distance == dy

def test_angle_between_points():
    x0 = 0.0
    y0 = 0.0
    x1 = 1.0
    y1 = 1.0
    angle = shape_tools.calc_angle_between_two_points_cartesian(x0,y0,x1,y1)
    assert angle==45.0

def test_sort_site_verts():
    invalid_x_points = [0.0,5.0,0.0,5.0]
    invalid_y_points = [0.0,0.0,5.0,5.0]
    invalid_verts = [[x,y] for x,y in zip(invalid_x_points,invalid_y_points)]
    valid_verts = shape_tools.check_site_verts(invalid_verts)
    valid_shape = Polygon(valid_verts)
    assert valid_shape.area == 25.0