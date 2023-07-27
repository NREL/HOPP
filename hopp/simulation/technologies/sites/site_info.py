import matplotlib.pyplot as plt
from shapely.geometry import *
from shapely.geometry.base import *
from attrs import define, field

from hopp.simulation.base import BaseClass
from hopp.simulation.technologies.resource.solar_resource import SolarResource
from hopp.simulation.technologies.resource.wind_resource import WindResource
from hopp.simulation.technologies.resource.elec_prices import ElectricityPrices
from hopp.simulation.technologies.layout.plot_tools import plot_shape
from hopp.utilities.log import hybrid_logger as logger
from hopp.utilities.keys import set_nrel_key_dot_env
from hopp.type_dec import NDArrayFloat, resource_file_converter


def plot_site(verts, plt_style, labels):
    for i in range(len(verts)):
        if i == 0:
            plt.plot([verts[0][0], verts[len(verts) - 1][0]], [verts[0][1], verts[len(verts) - 1][1]],
                     plt_style, label=labels)
        else:
            plt.plot([verts[i][0], verts[i - 1][0]], [verts[i][1], verts[i - 1][1]], plt_style)

    plt.grid()

@define
class SiteInfo(BaseClass):
    """
    Site specific information

    Attributes
    ----------
    data : dict 
        dictionary of initialization data
    lat : float
        site latitude [decimal degrees]
    long : float
        site longitude [decimal degrees]
    vertices : np.array
        site boundary vertices [m]
    polygon : shapely.geometry.polygon
        site polygon
    valid_region : shapely.geometry.polygon
        `tidy` site polygon
    solar_resource : :class:`hybrid.resource.SolarResource`
        class containing solar resource data
    wind_resource : :class:`hybrid.resource.WindResource`
        class containing wind resource data
    elec_prices : :class:`hybrid.resource.ElectricityPrices`
        Class containing electricity prices
    n_timesteps : int
        Number of timesteps in resource data
    n_periods_per_day : int
        Number of time periods per day
    interval : int
        Number of minutes per time interval 
    urdb_label : string
        `Link Utility Rate DataBase <https://openei.org/wiki/Utility_Rate_Database>`_ label for REopt runs
    capacity_hours : list
        Boolean list where ``True`` if the hour counts for capacity payments, ``False`` otherwise
    desired_schedule : list
        Absolute desired load profile [MWe]
    follow_desired_schedule : boolean
        ``True`` if a desired schedule was provided, ``False`` otherwise
    """
    data: dict = field(converter=dict)
    solar_resource_file: str = field(default="", converter=resource_file_converter)
    wind_resource_file: str = field(default="", converter=resource_file_converter)
    grid_resource_file: str = field(default="", converter=resource_file_converter)
    hub_height: float = field(default=97.)
    capacity_hours: list = field(default=[])
    desired_schedule: list = field(default=[])

    vertices: NDArrayFloat = field(init=False)
    polygon: Polygon = field(init=False)
    valid_region = field(init=False)
    lat: NDArrayFloat = field(init=False)
    lon: NDArrayFloat = field(init=False)
    solar_resource: SolarResource = field(init=False)
    n_timesteps: int = field(init=False)
    wind_resource: WindResource = field(init=False)
    elec_prices: ElectricityPrices = field(init=False)
    n_periods_per_day: int = field(init=False)
    interval: int = field(init=False)
    urdb_label: str | None = field(init=False)
    follow_desired_schedule: bool = field(init=False)

    def __attrs_post_init__(self) -> None:
        """
        Site specific information required by the hybrid simulation class and layout optimization.

        :param data: dict, containing the following keys:

            #. ``lat``: float, latitude [decimal degrees]
            #. ``lon``: float, longitude [decimal degrees]
            #. ``year``: int, year used to pull solar and/or wind resource data. If not provided, default is 2012 [-]
            #. ``elev``: float (optional), elevation (metadata purposes only) [m] 
            #. ``tz``: int (optional), timezone code (metadata purposes only) [-]
            #. ``no_solar``: bool (optional), if ``True`` solar data download for site is skipped, otherwise solar resource is downloaded from NSRDB
            #. ``no_wind``: bool (optional), if ``True`` wind data download for site is skipped, otherwise wind resource is downloaded from wind-toolkit
            #. ``site_boundaries``: dict (optional), with the following keys:

                * ``verts``: list of list [x,y], site boundary vertices [m]
                * ``verts_simple``: list of list [x,y], simple site boundary vertices [m]

            #. ``urdb_label``: string (optional), `Link Utility Rate DataBase <https://openei.org/wiki/Utility_Rate_Database>`_ label for REopt runs

            .. TODO: Can we get rid of verts_simple and simplify site_boundaries

        :param solar_resource_file: string, location (path) and filename of solar resource file (if not downloading from NSRDB)
        :param wind_resource_file: string, location (path) and filename of wind resource file (if not downloading from wind-toolkit)
        :param grid_resource_file: string, location (path) and filename of grid pricing data 
        :param hub_height: int (default = 97), turbine hub height for resource download [m]
        :param capacity_hours: list of booleans, (8760 length) ``True`` if the hour counts for capacity payments, ``False`` otherwise
        :param desired_schedule: list of floats, (8760 length) absolute desired load profile [MWe]
        """
        set_nrel_key_dot_env()
        if 'site_boundaries' in self.data:
            self.vertices = np.array([np.array(v) for v in self.data['site_boundaries']['verts']])
            self.polygon: Polygon = Polygon(self.vertices)
            self.valid_region = self.polygon.buffer(1e-8)
        if 'kml_file' in self.data:
            self.kml_data, self.polygon, self.data['lat'], self.data['lon'] = self.kml_read(self.data['kml_file'])
            self.polygon = self.polygon.buffer(1e-8)
        if 'lat' not in self.data or 'lon' not in self.data:
            raise ValueError("SiteInfo requires lat and lon")
        self.lat = self.data['lat']
        self.lon = self.data['lon']
        if 'year' not in self.data:
            self.data['year'] = 2012
        
        if 'no_solar' not in self.data:
            self.data['no_solar'] = False

        if not self.data['no_solar']:
            self.solar_resource = SolarResource(self.data['lat'], self.data['lon'], self.data['year'], filepath=self.solar_resource_file)
            self.n_timesteps = len(self.solar_resource.data['gh']) // 8760 * 8760

        if 'no_wind' not in self.data:
            self.data['no_wind'] = False

        if not self.data['no_wind']:
            # TODO: allow hub height to be used as an optimization variable
            self.wind_resource = WindResource(self.data['lat'], self.data['lon'], self.data['year'], wind_turbine_hub_ht=self.hub_height,
                                            filepath=self.wind_resource_file)
            self.n_timesteps = 8760

        self.elec_prices = ElectricityPrices(self.data['lat'], self.data['lon'], self.data['year'], filepath=self.grid_resource_file)
        # self.n_timesteps = len(self.solar_resource.data['gh']) // 8760 * 8760
        self.n_periods_per_day = self.n_timesteps // 365  # TODO: Does not handle leap years well
        self.interval = int((60*24)/self.n_periods_per_day)
        self.urdb_label = self.data['urdb_label'] if 'urdb_label' in self.data.keys() else None

        if len(self.capacity_hours) == self.n_timesteps:
            self.capacity_hours = self.capacity_hours
        else:
            self.capacity_hours = [False] * self.n_timesteps

        # Desired load schedule for the system to dispatch against
        self.desired_schedule = self.desired_schedule
        self.follow_desired_schedule = len(self.desired_schedule) == self.n_timesteps

            # FIXME: this a hack
        # if 'no_wind' in data:
        #     logger.info("Set up SiteInfo with solar resource files: {}".format(self.solar_resource.filename))
        # else:
        #     logger.info(
        #         "Set up SiteInfo with solar and wind resource files: {}, {}".format(self.solar_resource.filename,
        #                                                                             self.wind_resource.filename))

    # TODO: determine if the below functions are obsolete

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
