# function tools for the flatirons_layout

import numpy as np 
import matplotlib.pyplot as plt 



# determine if a point is inside a given polygon or not
# Polygon is a list of (x,y) pairs.

def find_edge(x_pt,y_pt,verts,x_verts,y_verts,max_min,dir):

    N = 100

    if dir == 'x':
        y = y_pt*np.ones(N)
        if max_min == 'min':
            x = np.linspace(np.min(x_verts),x_pt,N)
        else:
            x = np.linspace(x_pt,np.max(x_verts),N)
    else:
        x = x_pt*np.ones(N)
        if max_min == 'min':
            print('min y = ',np.min(y_verts))
            print(y_verts)
            y = np.linspace(np.min(y_verts),y_pt,N)
        else:
            y = np.linspace(y_pt,np.max(y_verts),N)

    # see if the points are inside the polygon
    idx = []
    for i in range(N):
        if point_inside_polygon(x[i],y[i],verts):
            idx.append(i)
    # print(idx)
    print(len(idx))
    if max_min == 'max':
        if dir == 'y':
            return y[idx[len(idx)-1]]
        if dir == 'x':
            return x[idx[len(idx)-1]]
    elif max_min == 'min':
        if dir == 'y':
            return y[idx[0]]
        if dir == 'x':
            return x[idx[0]]



# def find_edge_x(verts,x_verts,y_verts,istart,iend):

#     x = np.linspace(np.min(y_verts),np.max(y_verts),100)
#     np.where()


def find_closest(north,east,dist,x_pts,y_pts,x,y,x_verts,y_verts,verts):

    x_idx = []
    y_idx = []
    idx_exclude = []

    L = 0
    L2 = 250

    # north
    n_idx = np.where((north >= L) & (np.abs(east) <= L2))[0]
    # print('North: ', n_idx)
    if len(n_idx) == 0:
        x_idx.append(x_pts)
        #y_idx.append(find_edge(x_pts))
        y_idx.append(find_edge(x_pts,y_pts,verts,x_verts,y_verts,'max','y'))
    else:
        x_tmp = [x[i] for i in n_idx]
        y_tmp = [y[i] for i in n_idx]
        dist_tmp = [dist[i] for i in n_idx]
        # print(np.where(dist_tmp==np.min(dist_tmp))[0][0])
        x_idx.append(x_tmp[np.where(dist_tmp==np.min(dist_tmp))[0][0]])
        y_idx.append(y_tmp[np.where(dist_tmp==np.min(dist_tmp))[0][0]])
        idx_exclude.append(np.where((x_idx[0] == x) & (y_idx[0] == y))[0][0])

    
    # south
    s_idx = np.where((north <= -L) & (np.abs(east) <= L2))[0]
    for i in range(len(idx_exclude)):
        s_idx = s_idx[np.where(s_idx != idx_exclude[0])]
        # print('South: ', s_idx,idx_exclude[0])
    if len(s_idx) == 0:
        x_idx.append(x_pts)
        # y_idx.append(np.min(y_verts))
        y_idx.append(find_edge(x_pts,y_pts,verts,x_verts,y_verts,'min','y'))
    else:
        x_tmp = [x[i] for i in s_idx]
        y_tmp = [y[i] for i in s_idx]
        dist_tmp = [dist[i] for i in s_idx]
        # print(np.where(dist_tmp==np.min(dist_tmp))[0][0])
        x_idx.append(x_tmp[np.where(dist_tmp==np.min(dist_tmp))[0][0]])
        y_idx.append(y_tmp[np.where(dist_tmp==np.min(dist_tmp))[0][0]])

        idx_exclude.append(np.where((x_idx[1] == x) & (y_idx[1] == y))[0][0])

    # east
    e_idx = np.where((east >= L) & (np.abs(north) <= L2) )[0]
    for i in range(len(idx_exclude)):
        e_idx = e_idx[np.where(e_idx != idx_exclude[i])]
        # print('East: ', e_idx,idx_exclude)
    # print('Length of east index = ', len(e_idx))
    if len(e_idx) == 0:
        # x_idx.append(np.max(x_verts))
        x_idx.append(find_edge(x_pts,y_pts,verts,x_verts,y_verts,'max','x'))
        y_idx.append(y_pts)
    else:
        x_tmp = [x[i] for i in e_idx]
        y_tmp = [y[i] for i in e_idx]
        dist_tmp = [dist[i] for i in e_idx]
        # print(np.where(dist_tmp==np.min(dist_tmp))[0][0])
        x_idx.append(x_tmp[np.where(dist_tmp==np.min(dist_tmp))[0][0]])
        y_idx.append(y_tmp[np.where(dist_tmp==np.min(dist_tmp))[0][0]])

        idx_exclude.append(np.where((x_idx[2] == x) & (y_idx[2] == y))[0][0])
    # west
    w_idx = np.where((east <= -L) & (np.abs(north) <= L2))[0]
    for i in range(len(idx_exclude)):
        w_idx = w_idx[np.where(w_idx != idx_exclude[i])]
        # print('West:', w_idx,idx_exclude)
    if len(w_idx) == 0:
        # x_idx.append(np.min(x_verts))
        x_idx.append(find_edge(x_pts,y_pts,verts,x_verts,y_verts,'min','y'))
        y_idx.append(y_pts)
    else:
        x_tmp = [x[i] for i in w_idx]
        y_tmp = [y[i] for i in w_idx]
        dist_tmp = [dist[i] for i in w_idx]
        # print(np.where(dist_tmp==np.min(dist_tmp))[0][0])
        x_idx.append(x_tmp[np.where(dist_tmp==np.min(dist_tmp))[0][0]])
        y_idx.append(y_tmp[np.where(dist_tmp==np.min(dist_tmp))[0][0]])

    return x_idx, y_idx

def plot_site(verts,plt_style,labels):
    for i in range(len(verts)):
        if i == 0:
            #print('here')
            plt.plot( [verts[0][0],verts[len(verts)-1][0]],[verts[0][1],verts[len(verts)-1][1]],plt_style,label=labels)
        else:
            plt.plot( [verts[i][0],verts[i-1][0]],[verts[i][1],verts[i-1][1]],plt_style)

    plt.grid()
    

def point_inside_polygon(x,y,poly):

    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n+1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def distance_from_polygon(x,y,poly):
    
    dist = np.zeros(len(poly))
    
    in_poly = point_inside_polygon(x,y,poly)
    
    for i in range(len(poly)):
        if i == len(poly)-1:
            m = (poly[i][1]-poly[0][1]) / (poly[i][0]-poly[0][0])
        else:
            m = (poly[i][1]-poly[i+1][1]) / (poly[i][0]-poly[i+1][0])

        b = poly[i][1] - m*poly[i][0]

        a1 = -m
        b1 = 1
        c1 = -b

        dist[i] = np.abs(a1*x + b1*y + c1) / np.sqrt( (a1)**2 + (b1)**2 )
        
    if in_poly:
        dist_out = np.min(dist)
    else:
        dist_out = -np.min(dist)
        
    return dist_out


def distance_from_polygon_con(x_in,poly):
    
    nTurbs = int(len(x_in)/2)
    
    x = x_in[0:nTurbs]
    y = x_in[nTurbs:]
    
    # start out by finding if it is inside the box
    len_poly = len(poly)
    x_box = [poly[i][0] for i in range(len_poly)]
    y_box = [poly[i][1] for i in range(len_poly)]
    x_min = np.min(x_box)
    x_max = np.max(x_box)
    y_min = np.min(y_box)
    y_max = np.max(y_box)

    dist_out = np.zeros(nTurbs)

    # check the box first
    for k in range(nTurbs):
        if x[k] < x_min:
            # print('outside box')
            dist_out[k] = x[k] - x_min
        elif x[k] > x_max:
            # print('outside box')
            dist_out[k] = x_max - x[k]
        elif y[k] < y_min:
            # print('outside box')
            dist_out[k] = y[k] - y_min
        elif y[k] > y_max:
            # print('outside box')
            dist_out[k] = y_max - y[k]
        else:
            # =========================================
            # then check how close it is to the box
            # =========================================
            dist = np.zeros(len_poly)
            in_poly = point_inside_polygon(x[k], y[k], poly)

            for i in range(len_poly):

                if i == len_poly-1:
                    m = (poly[i][1]-poly[0][1]) / (poly[i][0]-poly[0][0])
                else:
                    m = (poly[i][1]-poly[i+1][1]) / (poly[i][0]-poly[i+1][0])

                b = poly[i][1] - m*poly[i][0]

                a1 = -m
                b1 = 1
                c1 = -b

                dist[i] = np.abs(a1*x[k] + b1*y[k] + c1) / np.sqrt( (a1)**2 + (b1)**2 )

            if in_poly:
                #print('inside polygon')
                dist_out[k] = np.min(dist)
            else:
                #print('outside polygon')
                dist_out[k] = -np.min(dist)
            
    print('Minimum distance from poly edge: ', np.min(dist_out))
        
    return np.min(dist_out)*(10**6)


def spaceConstraint(x_in):
    nTurbs = int(len(x_in)/2)
    
    x = x_in[0:nTurbs]
    y = x_in[nTurbs:]

    min_dist = 999999999
    for i in range(nTurbs):
        dist = [np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) for j in range(i)]
        dist.append(min_dist)
        min_dist = np.min(dist)
                
    print('Minimum distance from next turbine: ', min_dist/(501), 'D')
                
    return min_dist


def determineSpacing(x, y, D, nx, ny, minSpace):
    trackSpace = 100*D
    for i in range(len(nx)):
        trackSpace = min(np.sqrt( (x - nx[i])**2 + (y - ny[i])**2 ), trackSpace)

    if trackSpace > minSpace:
        return True
    else:
        return False


def findTurbines(verts,num_turbs,min_spacing,D):
	# compute min and max of the vertices of the polygon
    xMean = 0
    xMin = 0
    yMean = 0
    yMin = 0
    for i in range(len(verts)):

        xMean = np.max([verts[i][0],xMean])
        yMean = np.max([verts[i][1],yMean])
        xMin = np.min([verts[i][0],xMin])
        yMin = np.min([verts[i][1],yMin])


    # 	turbines    = [turbine for _, turbine in floris.farm.flow_field.turbine_map.items()]
    # 	D 			= np.mean([turbine.rotor_diameter for turbine in turbines]) 

    # plot the polygon
    for i in range(len(verts)):
        plt.plot(verts[i][0],verts[i][1],'ko')
        if i == len(verts)-1:
            plt.plot([verts[i][0],verts[0][0]],[verts[i][1],verts[0][1]],'b')        
        else:
            plt.plot([verts[i][0],verts[i+1][0]],[verts[i][1],verts[i+1][1]],'b')

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Enclosed space with randomly placed turbines')
    plt.axis('equal')

    # continue until you find the specified number of turbines
    count = 0
    check = 0
    turbine_x = []
    turbine_y = []
    while count < num_turbs:
        Nx = (xMean - xMin)*np.random.rand(1) + xMin 
        Ny = (yMean - yMin)*np.random.rand(1) + yMin 
        m = point_inside_polygon(Nx, Ny, verts)
        if not m:
            continue
        spacing = True
        if count > 1:
            spacing = determineSpacing(Nx, Ny, D, turbine_x[0:count], turbine_y[0:count], min_spacing)
        check = check + 1

        if check >= 1000:
            print('Cannot find more turbines sufficiently spaced apart.')
            print('Number of turbines specified = ', count)
            return turbine_x[0:count], turbine_y[0:count]
        
        if m and spacing:
            plt.plot(Nx, Ny, 'go')
            #print('here')
            turbine_x.append(Nx[0])
            turbine_y.append(Ny[0])
            count = count + 1
            check = 0
            #print('Identified turbine ', count)

	    #else:
	    #    plt.plot(Nx[i],Ny[i],'ro')
    #print(turbine_x,turbine_y)

    return turbine_x, turbine_y


def gridded_turbines(verts,num_turbs,min_x_spacing,min_y_spacing):

    xMean = 0
    xMin = 0
    yMean = 0
    yMin = 0
    for i in range(len(verts)):
        xMean = np.max([verts[i][0],xMean])
        yMean = np.max([verts[i][1],yMean])
        xMin = np.min([verts[i][0],xMin])
        yMin = np.min([verts[i][1],yMin])

    # plot the polygon
    # for i in range(len(verts)):
    #     plt.plot(verts[i][0],verts[i][1],'ko')
    #     if i == len(verts)-1:
    #         plt.plot([verts[i][0],verts[0][0]],[verts[i][1],verts[0][1]],'b')        
    #     else:
    #         plt.plot([verts[i][0],verts[i+1][0]],[verts[i][1],verts[i+1][1]],'b')


    # determine gridded coordinates (rectangle)
    xtmp = np.arange(xMin,xMean,min_x_spacing)
    ytmp = np.arange(yMin,yMean,min_y_spacing)

    x_locs = np.zeros(len(xtmp)*len(ytmp))
    y_locs = np.zeros(len(xtmp)*len(ytmp))
    count = 0
    for i in range(len(xtmp)):
        for j in range(len(ytmp)):
            x_locs[count] = xtmp[i]
            y_locs[count] = ytmp[j]
            count = count + 1

    # determine if gridded coordinates are within the polygon
    turbine_x = []
    turbine_y = []
    for i in range(len(x_locs)):
        m = point_inside_polygon(x_locs[i],y_locs[i],verts)
        if m:
            turbine_x.append(x_locs[i])
            turbine_y.append(y_locs[i])
            # plt.plot(x_locs[i],y_locs[i],'go')

        # else:
        # 	plt.plot(x_locs[i],y_locs[i],'ro')

    # plt.axis('equal')
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')

    return turbine_x,turbine_y

def distance_from_polygon(x,y,poly):

    # TODO: change this to a function of nearest line segment not line
    
    
    # start out by finding if it is inside the box
    x_box = np.zeros(len(poly))
    y_box = np.zeros(len(poly))

    for i in range(len(poly)):
        x_box[i] = poly[i][0]
        y_box[i] = poly[i][1]
        
    # check the box first
    if x < np.min(x_box):
        # print('outside box')
        dist_out = x - np.min(x_box)
    elif x > np.max(x_box):
        # print('outside box')
        dist_out = np.max(x_box) - x
    elif y < np.min(y_box):
        # print('outside box')
        dist_out = y - np.max(y_box)
    elif y > np.max(y_box):
        # print('outside box')
        dist_out = np.max(y_box) - y
    else:
    # =========================================
    # then check how close it is to the box
    # =========================================
        dist = np.zeros(len(poly))
        in_poly = point_inside_polygon(x,y,poly)
        # print(in_poly)

        for i in range(len(poly)):

            if i == len(poly)-1:
                m = (poly[i][1]-poly[0][1]) / (poly[i][0]-poly[0][0])
            else:
                m = (poly[i][1]-poly[i+1][1]) / (poly[i][0]-poly[i+1][0])

            b = poly[i][1] - m*poly[i][0]

            a1 = -m
            b1 = 1
            c1 = -b

            dist[i] = np.abs(a1*x + b1*y + c1) / np.sqrt( (a1)**2 + (b1)**2 )

        if in_poly:
            # print('inside polygon')
            dist_out = np.min(dist)
        else:
            # print('outside polygon')
            dist_out = -np.min(dist)
            
    # print('Distance from poly edge: ', dist_out)
        
    return dist_out

def inside_region(x,y,verts):

    # check each side of the square 
    x_verts = np.zeros(len(verts))
    y_verts = np.zeros(len(verts))
    for i in range(len(verts)):
        x_verts[i] = verts[i][0]
        y_verts[i] = verts[i][1]

    # check each side 
    if (x > np.min(x_verts)) and (x < np.max(x_verts)) and (y > np.min(y_verts)) and (y < np.max(y_verts)):
        # print('point is inside box')
        dist = np.max([np.min(x_verts) - x,
                       x - np.max(x_verts),
                       np.min(y_verts) - y,
                       y - np.max(y_verts)])
    else:
        # print('point is outside box')
        dist = 0

    return dist
        





        