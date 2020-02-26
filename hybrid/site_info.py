import logging
from shapely.geometry import *
from shapely.geometry.base import *

import matplotlib.pyplot as plt
import numpy as np

from hybrid.resource import (
    SolarResource,
    WindResource,
    )
from hybrid.wind import func_tools
logger = logging.getLogger('hybrid_system')


class SiteInfo:
    
    def __init__(self, data):
        self.data = data
        self.lat = data['lat']
        self.lon = data['lon']
        self.vertices = None
        self.polygon = None
        if 'site_boundaries' in data.keys():
            self.vertices = np.array([np.array(v) for v in data['site_boundaries']['verts']])
            self.polygon: Polygon = Polygon(self.vertices)
        if 'lat' not in data or 'lon' not in data:
            raise ValueError("SiteInfo requires lat and lon")
        if 'year' not in data:
            data['year'] = 2012
        self.solar_resource = SolarResource(data['lat'], data['lon'], data['year'])
        self.wind_resource = WindResource(data['lat'], data['lon'], data['year'], wind_turbine_hub_ht=80)
        self.n_timesteps = self.solar_resource.n_timesteps
        self.urdb_label = None
        if 'urdb_label' in data.keys():
            self.urdb_label = data['urdb_label']

        logger.info("SiteInfo created with lat {}, lon {}, urdb {}, site {}".format(self.lat, self.lon,
                                                                                     self.urdb_label, self.polygon))
        #
        # def ei(obj):
        #     try:
        #         return 'size: ' + str(len(dill.dumps(obj)))
        #     except:
        #         return 'error'
        # objgraph.show_refs(self, filename='site_info.png',
        #                    max_depth=6, too_many=20,
        #                    extra_info=ei,
        #                    refcounts=True, shortnames=True
        #                    )
    
    @property
    def boundary(self) -> BaseGeometry:
        # TODO: remove boundaries of interior holes
        # return self.polygon.boundary.difference(self.polygon.interiors)
        return self.polygon.exterior
    
    @property
    def bounding_box(self) -> np.ndarray:
        return np.array([np.min(self.vertices, 0), np.max(self.vertices, 0)])
    
    @property
    def max_distance(self) -> float:
        bounding_box = self.bounding_box
        return np.linalg.norm((bounding_box[1] - bounding_box[0]), 2)
    
    @property
    def center(self) -> Point:
        bounding_box = self.bounding_box
        return (bounding_box[1] - bounding_box[0]) * .5
    
    def plot(self):
        # func_tools.plot_site(verts,'ko-','Simplified')
        func_tools.plot_site(self.vertices, 'k', 'True')
        # plt.legend()
        plt.tick_params(which='both', labelsize=15)
        plt.xlabel('x (m)', fontsize=15)
        plt.ylabel('y (m)', fontsize=15)
        # plt.close()
