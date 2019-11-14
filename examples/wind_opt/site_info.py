import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import (
    Polygon,
    LinearRing,
    Point,
    )
from shapely.geometry.base import BaseGeometry

from hybrid.wind import func_tools
from hybrid.resource import SolarResource, WindResource


class SiteInfo:
    
    def __init__(self, data):
        self.data = data
        self.vertices = np.array([np.array(v) for v in data['site_boundaries']['verts']])
        self.polygon: Polygon = Polygon(self.vertices)
        if 'lat' not in data or 'lon' not in data:
            raise ValueError("SiteInfo requires lat and lon")
        if 'year' not in data:
            data['year'] = 2012
        self.solar_resource = SolarResource(data['lat'], data['lon'], data['year'])
        # TODO: allow hub height to be used as an optimization variable
        self.wind_resource = WindResource(data['lat'], data['lon'], data['year'], wind_turbine_hub_ht=80)

    @property
    def boundary(self) -> BaseGeometry:
        # TODO: remove boundaries of interior holes
        # return self.polygon.boundary.difference(self.polygon.interiors)
        return self.polygon.boundary
    
    @property
    def bounding_box(self) -> np.ndarray:
        return np.array([np.min(self.site_info.vertices, 0), np.max(self.site_info.vertices, 0)])
    
    def get_evenly_spaced_points_along_border(self, shape: BaseGeometry, spacing: float) -> [Point]:
        length = shape.length
        result = []
        d = 0.0
        while d <= length:
            result.append(shape.interpolate(d))
            d += spacing
        # print('shape: ', [c for c in shape.coords])
        # print('boundary: ', [(p.x, p.y) for p in result])
        return result
    
    def plot(self):
        # func_tools.plot_site(verts,'ko-','Simplified')
        func_tools.plot_site(self.vertices, 'k', 'True')
        # plt.legend()
        plt.tick_params(which='both', labelsize=15)
        plt.xlabel('x (m)', fontsize=15)
        plt.ylabel('y (m)', fontsize=15)
        # plt.close()
