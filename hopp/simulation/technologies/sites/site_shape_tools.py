from shapely.geometry import Polygon, MultiPolygon, Point, shape, box
import numpy as np

def calc_dist_between_two_points_cartesian(x1,y1,x2,y2):
    dx = np.abs(x2-x1)
    dy = np.abs(y2-y1)
    return np.sqrt((dx**2) + (dy**2))

def calc_angle_between_two_points_cartesian(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    angle_deg = np.rad2deg(np.arctan2(dx,dy))
    if angle_deg<0:
        angle_deg += 360
    return angle_deg


def check_site_verts(verts):
    x_points = [v[0] for v in verts]
    y_points = [v[1] for v in verts]
    x0 = min(x_points)
    y0 = min(y_points)
    
    dx = 0 - x0 #dx is positive if x0 is negative
    dy = 0 - y0 #dy is positive if y0 is negative

    x_pos = [x+dx for x in x_points]
    y_pos = [y+dy for y in y_points]

    x_center = (max(x_pos) - min(x_pos))/2
    y_center = (max(y_pos) - min(y_pos))/2

    distances = [None]*len(x_points)
    angles =  [None]*len(x_points)
    i =0
    for x,y in zip(x_pos,y_pos):
        distances[i] = calc_dist_between_two_points_cartesian(x_center,y_center,x,y)
        angles[i] = calc_angle_between_two_points_cartesian(x_center,y_center,x,y)
        i +=1 
    
    #1) sort based on angles from smallest to largest
    angles, distances, x_pos, y_pos = (list(t) for t in zip(*sorted(zip(angles, distances, x_pos, y_pos))))
    #2) sort any same angles based on distance:
    unique_angle,angle_cnt = np.unique(angles,return_counts=True)
    if any(a>1 for a in angle_cnt):
        repeat_angles = [unique_angle[i] for i,cnt in enumerate(angle_cnt) if cnt>1]
        for rep_ang in repeat_angles:
            indx_rep = list(np.argwhere(angles==rep_ang).flatten())
            # sort based on distance
            distances[indx_rep], angles[indx_rep], x_pos[indx_rep], y_pos[indx_rep] = (list(t) for t in zip(*sorted(zip(distances[indx_rep], angles[indx_rep], x_pos[indx_rep], y_pos[indx_rep]))))
    organized_verts = [[x+dx,y+dy] for x,y in zip(x_pos,y_pos)]
    return organized_verts

def make_square(area_m2,x0=0.0,y0=0.0):
    site_length = np.sqrt(area_m2)
    center = site_length/2
    y1 = y0 + site_length
    x1 = x0 + site_length
    poly = box(x0,y0,x1,y1)
    verts = [[x0,x0],[x1,y0],[x1,y1],[x0,y1]]
    vertices = np.array([np.array(v) for v in verts])
    return poly, vertices

def make_rectangle(area_m2,aspect_ratio=1.5,x0=0.0,y0=0.0):
    #aspect ratio is width/height
    # width * height = area
    # width = aspect*height
    height = np.sqrt(area_m2/aspect_ratio)
    width = area_m2/height
    x1 = x0 + width
    y1 = y0 + height
    poly = box(x0,y0,x1,y1)
    verts = [[x0,x0],[x1,y0],[x1,y1],[x0,y1]]
    vertices = np.array([np.array(v) for v in verts])
    return poly,vertices

def make_circle(area_m2,deg_diff = 10,x0=0.0,y0=0.0):
    r = np.sqrt(area_m2/np.pi)
    dx = np.deg2rad(deg_diff)
    rads = np.arange(0,2*np.pi,dx)
    x_coords = r*np.cos(rads)
    y_coords = r*np.sin(rads)

    if any(x<x0 for x in x_coords):
        x_diff = [x0-x for x in x_coords if x<x0]
        x_adj = max(x_diff)
        x_points = [x+x_adj for x in x_coords]
    else:
        x_points = [x for x in x_coords]
    if any(y<y0 for y in y_coords):
        y_diff = [y0-y for y in y_coords if y<y0]
        y_adj = max(y_diff)
        y_points = [y+y_adj for y in y_coords]
    else:
        y_points = [y for y in y_coords]

    coords = []
    for x,y in zip(x_points,y_points):
        coords.append([x,y])

    poly = Polygon(coords)
    vertices = np.array([np.array(v) for v in coords])
    return poly, vertices

def make_hexagon(area_m2,x0=0.0,y0=0.0):
    s = np.sqrt(area_m2*(2/(3*np.sqrt(3))))
    rads = np.arange(0,2*np.pi,np.deg2rad(60))
    x_coords = s*np.cos(rads)
    y_coords = s*np.sin(rads)

    if any(x<x0 for x in x_coords):
        x_diff = [x0-x for x in x_coords if x<x0]
        x_adj = max(x_diff)
        x_points = [x+x_adj for x in x_coords]
    else:
        x_points = [x for x in x_coords]
    if any(y<y0 for y in y_coords):
        y_diff = [y0-y for y in y_coords if y<y0]
        y_adj = max(y_diff)
        y_points = [y+y_adj for y in y_coords]
    else:
        y_points = [y for y in y_coords]

    coords = []
    for x,y in zip(x_points,y_points):
        coords.append([x,y])

    vertices = np.array([np.array(v) for v in coords])
    poly = Polygon(coords)
    return poly, vertices

