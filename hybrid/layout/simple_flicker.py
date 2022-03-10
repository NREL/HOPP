# implement a simple shading model

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from shapely.geometry import Point, Polygon

class SimpleFlicker():

    def __init__(self, solar_verts, T, turbine_locs):

        self.turbine_locs = [[0, 0]]
        self.solar_verts = solar_verts
        self.turbine_locs = turbine_locs


    def rotate(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy

    def find_angle(self, T_in):

        # find the omega
        from scipy import interpolate

        T = np.array([6, 12, 18])
        omega = np.array([-90, 0, 90])
        f = interpolate.interp1d(T, omega)

        if T_in < 6:
            omega_out = 0
            print('Sun is not high enough for a shadow...')
        elif T_in > 18:
            omega_out = 0
            print('Sun is not high enough for a shadow...')
        else:
            omega_out = f(T_in)

        return -np.radians(omega_out)




    def calculate_shadow(self, time_idx, show=True):
        # user inputs
        T = time_idx  # time (in military time)
        d = 10 # number of days since the new year


        # turbine parameters
        HH = 90 # hub height
        D = 126 # rotor diameter
        wd = 5  # tower width is 5 m?

        # turbine location
        x_loc = self.turbine_locs[0]
        y_loc = self.turbine_locs[1]

        # position
        lat = 39.7555
        lon = -105.2211

        # calculate the shadow
        delta = np.radians(-23.45 * np.cos( np.radians(360/365 * (d + 10)) ))
        omega = self.find_angle(T)

        # tower shadow
        Fx = -( np.cos(delta) * np.sin(omega) / (np.sin(lat) * np.sin(delta) + np.cos(lat) * np.cos(delta) * np.cos(omega)))
        numY = ( np.sin(np.radians(lat)) * np.cos(delta) * np.cos(omega) - np.cos(np.radians(lat)) * np.cos(delta) )
        denY = ( np.sin(np.radians(lat)) * np.sin(delta) + np.cos(np.radians(lat)) * np.cos(delta) * np.cos(omega) )
        Fy = -numY / denY

        # plot turbine shadow and rotor shadow
        fig, ax = plt.subplots()
        plt.plot(x_loc,y_loc,'bo')
        plt.plot([x_loc + wd/2, (x_loc+wd/2) + (HH) * Fx], [y_loc, y_loc + (HH) * Fy],'k')
        plt.plot([x_loc - wd/2, (x_loc-wd/2) + (HH) * Fx], [y_loc, y_loc + (HH) * Fy], 'k')

        length = (HH + D/2) * Fx - (HH - D/2) * Fx
        angle = np.degrees(-90 - np.tan(Fx/Fy))
        a = length/2
        b = D/2
        x = np.linspace(-a,a,100)
        y = b * np.sqrt( 1 - (x/a)**2 )
        rx = np.zeros(len(x))
        ry = np.zeros(len(y))
        rx2 = np.zeros(len(x))
        ry2 = np.zeros(len(y))
        poly_rotor = []
        for i in range(len(x)):
            rx[i], ry[i] = self.rotate([0,0], [x[i],y[i]], np.radians(angle))
            poly_rotor.append((rx[i]+(HH*Fx)+x_loc,ry[i]+(HH*Fy)+y_loc))

        for i in range(len(x)):
            rx2[i], ry2[i] = self.rotate([0, 0], [x[i], -y[i]], np.radians(angle))
            poly_rotor.append((rx2[i]+(HH*Fx)+x_loc,ry2[i]+(HH*Fy)+y_loc))
        plt.plot(rx+(HH*Fx)+x_loc,ry+(HH*Fy)+y_loc,'k')
        plt.plot(rx2+(HH*Fx)+x_loc,ry2+(HH*Fy)+y_loc,'k')

        for i in range(len(self.solar_verts)-1):
            plt.plot([self.solar_verts[i][0], self.solar_verts[i+1][0]], [self.solar_verts[i][1], self.solar_verts[i+1][1]],'r')
        plt.plot([self.solar_verts[0][0], self.solar_verts[i + 1][0]], [self.solar_verts[0][1], self.solar_verts[i + 1][1]], 'r')

        plt.xlim([-500,500])
        plt.ylim([-500, 500])
        plt.grid()

        if show:
            plt.show()

        poly_tower = [(x_loc + wd/2, y_loc), (x_loc - wd/2, y_loc),
                      (x_loc - wd/2 + (HH) * Fx, y_loc + HH * Fy), (x_loc + wd/2 + HH * Fx, y_loc + HH * Fy)]

        return poly_rotor, poly_tower


    def point_inside(self, point, coords):

        # Create Point objects
        p1 = Point(point[0], point[1])

        # Create a Polygon
        poly = Polygon(coords)

        # check if point is within polygon
        return p1.within(poly)

    def determine_boundaries(self):

        x_min = 0
        x_max = 0
        y_min = 0
        y_max = 0
        for point in self.solar_verts:

            # check x points
            if point[0] < x_min:
                x_min = point[0]
            elif point[0] > x_max:
                x_max = point[0]

            # check y points
            if point[1] < y_min:
                y_min = point[1]
            elif point[1] > y_max:
                y_max = point[1]

        return x_min, x_max, y_min, y_max

    def calculate_overlap(self, T, show=False):

        # determine xmin, xmax, ymin, ymax
        xmin, xmax, ymin, ymax = self.determine_boundaries(self.solar_verts)

        # solar boundaries - assume rectangle
        # generation points inside the solar_verts
        N = 10
        x = np.linspace(xmin,xmax,N)
        y = np.linspace(ymin,ymax,N)

        # turbine parameters
        D = 126
        Area = np.pi * (D / 2)**2

        # cycle through turbine shadows
        # determine if those points are within the turbine shadow
        inside_shadow = np.zeros((N,N))
        for i in range(len(self.turbine_locs)):

            poly_rotor, poly_tower = self.calculate_shadow(T, show=False)
            for j in range(N):
                for k in range(N):
                    point = [x[j],y[k]]
                    if inside_shadow[j,k] == 0:
                        if self.point_inside(point, poly_rotor):
                            inside_shadow[j,k] = 1/3  # not sure what the ratio should be here
                        elif self.point_inside(point, poly_tower):
                            inside_shadow[j, k] = 1

        for i in range(N):
            for j in range(N):
                if inside_shadow[i,j] == 1:
                    plt.plot(x[i],y[j],'go')
                else:
                    plt.plot(x[i], y[j], 'bo')

        if show:
            plt.show()

        return np.sum(inside_shadow) / (N*N)

    def calculate_losses(self, T, show=False):

        losses = self.calculate_overlap(T, show=show)

        print('Percent losses: ', 100 * losses, '%')

        return 100 * losses






