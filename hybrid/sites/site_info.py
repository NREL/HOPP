import matplotlib.pyplot as plt
from shapely.geometry import *
from shapely.geometry.base import *

from hybrid.resource import (
    SolarResource,
    WindResource,
    ElectricityPrices
    )
from hybrid.layout.plot_tools import plot_shape
from hybrid.log import hybrid_logger as logger
from hybrid.keys import set_nrel_key_dot_env


def plot_site(verts, plt_style, labels):
    for i in range(len(verts)):
        if i == 0:
            plt.plot([verts[0][0], verts[len(verts) - 1][0]], [verts[0][1], verts[len(verts) - 1][1]],
                     plt_style, label=labels)
        else:
            plt.plot([verts[i][0], verts[i - 1][0]], [verts[i][1], verts[i - 1][1]], plt_style)

    plt.grid()


class SiteInfo:
    
    def __init__(self, data, solar_resource_file="", wind_resource_file="", grid_resource_file=""):
        set_nrel_key_dot_env()
        self.data = data
        self.vertices = np.array([np.array(v) for v in data['site_boundaries']['verts']])
        self.polygon: Polygon = Polygon(self.vertices)
        self.valid_region = self.polygon.buffer(1e-8)
        if 'lat' not in data or 'lon' not in data:
            raise ValueError("SiteInfo requires lat and lon")
        self.lat = data['lat']
        self.lon = data['lon']
        if 'year' not in data:
            data['year'] = 2012
        self.solar_resource = SolarResource(data['lat'], data['lon'], data['year'], filepath=solar_resource_file)
        # TODO: allow hub height to be used as an optimization variable
        self.wind_resource = WindResource(data['lat'], data['lon'], data['year'], wind_turbine_hub_ht=80,
                                          filepath=wind_resource_file)
        self.elec_prices = ElectricityPrices(data['lat'], data['lon'], data['year'], filepath=grid_resource_file)
        self.n_timesteps = len(self.solar_resource.data['gh']) // 8760 * 8760
        self.n_periods_per_day = self.n_timesteps // 365  # TODO: Does not handle leap years well
        self.interval = (60*24)/self.n_periods_per_day
        self.urdb_label = data['urdb_label'] if 'urdb_label' in data.keys() else None
        logger.info("Set up SiteInfo with solar and wind resource files: {}, {}".format(self.solar_resource.filename,
                                                                                         self.wind_resource.filename))

    @property
    def boundary(self) -> BaseGeometry:
        # TODO: remove boundaries of interior holes
        # return self.polygon.boundary.difference(self.polygon.interiors)
        return self.polygon.exterior
    
    @property
    def bounding_box(self) -> np.ndarray:
        return np.array([np.min(self.vertices, 0), np.max(self.vertices, 0)])
    
    @property
    def center(self) -> Point:
        bounding_box = self.bounding_box
        return (bounding_box[1] - bounding_box[0]) * .5
    
    def plot(self,
             figure=None,
             axes=None,
             border_color=(0, 0, 0),
             alpha=0.95,
             linewidth=4.0
             ):
        bounds = self.polygon.bounds
        site_sw_bound = np.array([bounds[0], bounds[1]])
        site_ne_bound = np.array([bounds[2], bounds[3]])
        site_center = .5 * (site_sw_bound + site_ne_bound)
        max_delta = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        reach = (max_delta / 2) * 1.3
        min_plot_bound = site_center - reach
        max_plot_bound = site_center + reach

        if not figure and not axes:
            figure = plt.figure(1)
            axes = figure.add_subplot(111)

        axes.set_aspect('equal')
        axes.set(xlim=(min_plot_bound[0], max_plot_bound[0]), ylim=(min_plot_bound[1], max_plot_bound[1]))
        plot_shape(figure, axes, self.polygon, '--', color=border_color, alpha=alpha, linewidth=linewidth / 2)

        plt.tick_params(which='both', labelsize=15)
        plt.xlabel('x (m)', fontsize=15)
        plt.ylabel('y (m)', fontsize=15)

        return figure, axes
