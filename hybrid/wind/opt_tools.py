# function tools for the flatirons_layout
import numpy as np

from scipy.optimize import minimize

import hybrid.wind.func_tools as func_tools


def optimize_wind_AEP(x: [int], scenario):
    nTurbs = int(len(x) / 2)
    return -optimize_wind_AEP_from_coordinate_lists(x[0:nTurbs], x[nTurbs:], scenario)


def optimize_wind_AEP_from_coordinate_lists(x_coordinates: [int], y_coordinates: [int], scenario):
    scenario.systems['Wind']['Windpower'].Farm.wind_farm_xCoordinates = x_coordinates
    scenario.systems['Wind']['Windpower'].Farm.wind_farm_yCoordinates = y_coordinates
    
    # Run the default system and view outputs
    outputs = scenario.output_values(scenario.run_single())
    
    # print('Power output = ', outputs['Wind']['annual_energy']/1000)
    return outputs['Wind']['annual_energy'] / 1000


def layout_opt(x0, scenario, verts):
    if type(x0) is np.ndarray:
        x0 = x0.tolist()
    
    # find the initial starting power production
    nTurbs = int(len(x0) / 2)
    scenario.systems['Wind']['Windpower'].Farm.wind_farm_xCoordinates = x0[0:nTurbs]
    scenario.systems['Wind']['Windpower'].Farm.wind_farm_yCoordinates = x0[nTurbs:]
    
    # Run the default system and view outputs
    outputs = scenario.output_values(scenario.run_single())
    
    power0 = outputs['Wind']['annual_energy'] / 1000
    
    # constraints
    D = scenario.systems['Wind']['Windpower'].Turbine.wind_turbine_rotor_diameter
    min_spacing = 2 * D
    
    cons = []
    tmp1 = {'type': 'ineq', 'fun': lambda x: func_tools.spaceConstraint(x) - min_spacing}
    tmp2 = {'type': 'ineq', 'fun': lambda x, *args: func_tools.distance_from_polygon_con(x, verts), 'args': (verts,)}
    cons.append(tmp1)
    cons.append(tmp2)
    
    # start out by finding if it is inside the box
    x_box = [verts[i][0] for i in range(len(verts))]
    y_box = [verts[i][1] for i in range(len(verts))]
    
    # bounds
    bnds_low = [np.min(x_box)] * nTurbs + [np.min(y_box)] * nTurbs
    bnds_up = [np.max(x_box)] * nTurbs + [np.max(y_box)] * nTurbs
    
    # optimize the layout
    resPlant = minimize(optimize_wind_AEP, x0, args=(scenario,), method='COBYLA', constraints=cons,
                        options={'maxiter': 100000, 'rhobeg': 5})
    
    # find the initial starting power production
    scenario.systems['Wind']['Windpower'].Farm.wind_farm_xCoordinates = resPlant.x[0:nTurbs].tolist()
    scenario.systems['Wind']['Windpower'].Farm.wind_farm_yCoordinates = resPlant.x[nTurbs:].tolist()
    
    # Run the default system and view outputs
    outputs = scenario.output_values(scenario.run_single())
    
    power_opt = outputs['Wind']['annual_energy'] / 1000
    print('AEP gain = ', 100 * (power_opt - power0) / power0)
    
    return resPlant.x, power_opt


# amount of area off limits
def PolygonArea(corners):
    n = len(corners)  # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


def shadow_weighting(area, x, y, x_center, y_center):
    dist = np.zeros(len(x))
    for i in range(len(x)):
        dist = 1000 / np.sqrt((x[i] - x_center) ** 2 + (y[i] - y_center) ** 2)
    
    return np.sum(dist) * area


def optimize_solar(x, nArrays, x_turb, y_turb):
    # cost function - exponential function - promote the largest areas possible
    area = np.zeros(nArrays)
    total = 0
    for i in range(nArrays):
        x_center = x[4 * i + 2]
        y_center = x[4 * i + 3]
        length = x[4 * i]
        width = x[4 * i + 1]
        verts = [[x_center - length / 2, y_center - width / 2],
                 [x_center - length / 2, y_center + width / 2],
                 [x_center + length / 2, y_center + width / 2],
                 [x_center + length / 2, y_center - width / 2]
                 ]
        area[i] = PolygonArea(verts)
        weighted_area = shadow_weighting(area[i], x_turb, y_turb, x_center, y_center)
        total = area[i] + total
    
    # print('Length = ', x[0])
    # print('Width = ', x[1])
    # print('Coords = ', x[2],x[3])
    print('Area = ', area)
    
    return -total


def no_obstacles(x, verts, nArrays, turb_locs):
    # calculate the number of turbines or other verts that are in the solar arrays
    
    obstacles = 0
    count = 0
    x_turb = turb_locs['x']
    y_turb = turb_locs['y']
    verts_record = []
    x_verts = np.zeros(nArrays * 4)
    y_verts = np.zeros(nArrays * 4)
    dist = 0
    for i in range(nArrays):
        
        x_center = x[4 * i + 2]
        y_center = x[4 * i + 3]
        length = x[4 * i]
        width = x[4 * i + 1]
        verts = [[x_center - length / 2, y_center - width / 2],
                 [x_center - length / 2, y_center + width / 2],
                 [x_center + length / 2, y_center + width / 2],
                 [x_center + length / 2, y_center - width / 2]
                 ]
        # dist = np.zeros(len(x_turb))
        for j in range(len(x_turb)):
            # print(func_tools.point_inside_polygon(x_turb[j],y_turb[j],verts))
            if func_tools.point_inside_polygon(x_turb[j], y_turb[j], verts):
                dist = func_tools.inside_region(x_turb[j], y_turb[j], verts) + dist
                # print('Distance inside the solar array = ', dist)
                obstacles = obstacles + 1
        
        verts_record.append(verts)
        for i in range(len(verts)):
            x_verts[count] = verts[i][0]
            y_verts[count] = verts[i][1]
            count = count + 1
    
    # are any of the vertices in the other footprints
    # for i in range(nArrays):
    #     for j in range(nArrays):
    #         if i != j:
    #             for k in range(4):
    #                 if func_tools.point_inside_polygon(x_verts[4*j+k],y_verts[4*j+k],verts_record[i]):
    #                     dist = dist + func_tools.inside_region(x_verts[4*j+k],y_verts[4*j+k],verts_record[i])
    #                     #print('corners inside another array...')
    #                     obstacles = obstacles + 1
    
    print('number of obstacles = ', obstacles)
    print('Distance inside solar farm = ', dist)
    
    return dist


def in_polygon(x, verts, nArrays):
    # make sure all the vertices are in the polygon
    n_verts = 0
    dist = np.zeros((nArrays, 4))
    for i in range(nArrays):
        
        x_center = x[4 * i + 2]
        y_center = x[4 * i + 3]
        length = x[4 * i]
        width = x[4 * i + 1]
        
        # print(func_tools.point_inside_polygon(x_center - length/2, y_center - width/2, verts))
        if not func_tools.point_inside_polygon(x_center - length / 2, y_center - width / 2, verts):
            dist[i, 0] = func_tools.distance_from_polygon(x_center - length / 2, y_center - width / 2, verts)
            n_verts = n_verts + 1
        if not func_tools.point_inside_polygon(x_center - length / 2, y_center + width / 2, verts):
            dist[i, 1] = func_tools.distance_from_polygon(x_center - length / 2, y_center + width / 2, verts)
            n_verts = n_verts + 1
        if not func_tools.point_inside_polygon(x_center + length / 2, y_center - width / 2, verts):
            dist[i, 2] = func_tools.distance_from_polygon(x_center + length / 2, y_center - width / 2, verts)
            n_verts = n_verts + 1
        if not func_tools.point_inside_polygon(x_center + length / 2, y_center + width / 2, verts):
            dist[i, 3] = func_tools.distance_from_polygon(x_center + length / 2, y_center + width / 2, verts)
            n_verts = n_verts + 1
        
        print('Distance outside boundaries = ', np.sum(dist))
    
    return np.sum(dist)


def spaceConstraint(x, verts, nArrays):
    # make sure the polygons are sufficiently spaced
    dist = 0
    for i in range(nArrays):
        length_i = x[4 * i]
        width_i = x[4 * i + 1]
        x_i = x[4 * i + 2]
        y_i = x[4 * i + 3]
        for j in range(nArrays):
            length_j = x[4 * j]
            width_j = x[4 * j + 1]
            x_j = x[4 * j + 2]
            y_j = x[4 * j + 3]
            if i != j:
                dist_x = (x_j - length_j) - (x_i + length_i)
                dist_y = (y_j - width_j) - (y_i + width_i)
                if (x_j > x_i) and (dist_x < 0.0):
                    dist = dist_x + dist
                if (y_j > y_i) and (dist_y < 0.0):
                    dist = dist_y + dist
    
    return dist


def solar_opt(x0, nArrays, verts, x, y):
    # design variables
    # 1. length and width of the arary (eventually, this will be number in rows and number of rows and tilt angle)
    # 2. x, y locations of the center
    # 3. (mixed integer - later) - number of sections - the above number of rows and number in rows are integers ->
    # using length and width as a proxy
    
    # constraints:
    # 1. no turbins must be in the center
    # 2. all vertices are contained within polygon
    
    x_verts = np.zeros(len(verts))
    y_verts = np.zeros(len(verts))
    for i in range(len(verts)):
        x_verts[i] = verts[i][0]
        y_verts[i] = verts[i][1]
    
    turb_locs = dict()
    turb_locs['x'] = x
    turb_locs['y'] = y
    
    # constraints
    cons = []
    tmp1 = {
        'type': 'ineq', 'fun': lambda x, *args: no_obstacles(x, verts, nArrays, turb_locs),
        'args': (verts, nArrays, turb_locs)
        }
    tmp2 = {'type': 'ineq', 'fun': lambda x, *args: in_polygon(x, verts, nArrays), 'args': (verts, nArrays)}
    tmp3 = {'type': 'ineq', 'fun': lambda x, *args: spaceConstraint(x, verts, nArrays), 'args': (verts, nArrays)}
    cons.append(tmp1)
    cons.append(tmp2)
    cons.append(tmp3)
    
    bnds = [(0.1, np.max(x_verts) - np.min(x_verts)),
            (0.1, np.max(y_verts) - np.min(y_verts)),
            (np.min(x_verts), np.max(x_verts)),
            (np.min(y_verts), np.max(y_verts))]
    
    # optimize the solar layout
    resPlant = minimize(optimize_solar, x0, args=(nArrays, x, y), method='COBYLA',
                        constraints=cons,
                        bounds=bnds,
                        options={'eps': 1, 'maxiter': 10000})
    
    # check = optimize_solar(x0,nArrays,x,y)
    # cons1 = no_obstacles(x0,verts,nArrays,turb_locs)
    # con2 = in_polygon(x0,verts,nArrays)
    
    print(resPlant)
    
    opt_x = resPlant.x
    # opt_x = x0
    
    print('Initial = ', x0)
    print('Optimal = ', opt_x)
    
    return opt_x
