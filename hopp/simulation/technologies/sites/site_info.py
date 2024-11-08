from typing import Optional, Union
from pathlib import Path

from attrs import define, field
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon, MultiPolygon, Point, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from shapely import make_valid
from fastkml import kml, KML
import pyproj
import utm

from hopp.simulation.technologies.resource import (
    SolarResource,
    WindResource,
    WaveResource,
    ElectricityPrices
)
from hopp.tools.layout.plot_tools import plot_shape
from hopp.utilities.log import hybrid_logger as logger
from hopp.utilities.keys import set_nrel_key_dot_env
from hopp.type_dec import (
    hopp_array_converter as converter, NDArrayFloat, resource_file_converter,
    hopp_float_type
)
from hopp.simulation.base import BaseClass
from hopp.utilities.validators import contains

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
    Represents site-specific information needed by the hybrid simulation class and layout optimization.

    Args:
        data: Dictionary containing site-specific information.
        solar_resource_file: Path to solar resource file. Defaults to "".
        wind_resource_file: Path to wind resource file. Defaults to "".
        grid_resource_file: Path to grid pricing data file. Defaults to "".
        hub_height: Turbine hub height for resource download in meters. Defaults to 97.0.
        capacity_hours: Boolean list indicating hours for capacity payments. Defaults to [].
        desired_schedule: Absolute desired load profile in MWe. Defaults to [].
        curtailment_value_type: whether to curtail power above grid interconnection limit or desired schedule. 
            Options "interconnect_kw" or "desired_schedule". Defaults to "interconnect_kw".
        solar: Whether to set solar data for this site. Defaults to True.
        wind: Whether to set wind data for this site. Defaults to True.
        wave: Whether to set wave data for this site. Defaults to False.
        wind_resource_origin: Which wind resource API to use, defaults to WIND Toolkit
    """
    # User provided
    data: dict
    solar_resource_file: Union[Path, str] = field(default="", converter=resource_file_converter)
    wind_resource_file: Union[Path, str] = field(default="", converter=resource_file_converter)
    wave_resource_file: Union[Path, str] = field(default="", converter=resource_file_converter)
    grid_resource_file: Union[Path, str] = field(default="", converter=resource_file_converter)
    hub_height: hopp_float_type = field(default=97., converter=hopp_float_type)
    capacity_hours: NDArray = field(default=[], converter=converter(bool))
    desired_schedule: NDArrayFloat = field(default=[], converter=converter())
    curtailment_value_type: str = field(default="interconnect_kw", validator=contains(["interconnect_kw", "desired_schedule"]))

    solar: bool = field(default=True)
    wind: bool = field(default=True)
    wave: bool = field(default=False)
    wind_resource_origin: str = field(default="WTK", validator=contains(["WTK", "TAP"]))

    # Set in post init hook
    n_timesteps: int = field(init=False, default=None)
    lat: hopp_float_type = field(init=False)
    lon: hopp_float_type = field(init=False)
    year: int = field(init=False, default=2012)
    tz: Optional[int] = field(init=False, default=None)
    solar_resource: Optional[SolarResource] = field(init=False, default=None)
    wind_resource: Optional[WindResource] = field(init=False, default=None)
    wave_resoure: Optional[WaveResource] = field(init=False, default=None)
    elec_prices: Optional[ElectricityPrices] = field(init=False, default=None)
    n_periods_per_day: int = field(init=False)
    interval: int = field(init=False)
    follow_desired_schedule: bool = field(init=False)
    polygon: Union[Polygon, BaseGeometry] = field(init=False)
    vertices: NDArrayFloat = field(init=False)
    kml_data: Optional[KML] = field(init=False, default=None)

    # .. TODO: Can we get rid of verts_simple and simplify site_boundaries

    def __attrs_post_init__(self):
        """
        The following are set in this post init hook:
            lat (numpy.float64): Site latitude in decimal degrees.
            lon (numpy.float64): Site longitude in decimal degrees.
            tz (int, optional): Timezone code for metadata purposes only. Defaults to None.
            vertices (:obj:`NDArray`): Site boundary vertices in meters.
            polygon (:obj:`shapely.geometry.polygon.Polygon`): Site polygon.
            valid_region (:obj:`shapely.geometry.polygon.Polygon`): Tidy site polygon.
            solar_resource (:obj:`hopp.simulation.technologies.resource.SolarResource`): Class containing solar resource data.
            wind_resource (:obj:`hopp.simulation.technologies.resource.WindResource`): Class containing wind resource data.
            wave_resoure (:obj:`hopp.simulation.technologies.resource.WaveResource`): Class containing wave resource data.
            elec_prices (:obj:`hopp.simulation.technologies.resource.ElectricityPrices`): Class containing electricity prices.
            n_timesteps (int): Number of timesteps in resource data.
            n_periods_per_day (int): Number of time periods per day.
            interval (int): Number of minutes per time interval.
            urdb_label (str): Link to `Utility Rate DataBase <https://openei.org/wiki/Utility_Rate_Database>`_ label for REopt runs.
            follow_desired_schedule (bool): Indicates if a desired schedule was provided. Defaults to False.
        """
        set_nrel_key_dot_env()

        data = self.data
        if 'site_boundaries' in data:
            self.vertices = np.array([np.array(v) for v in data['site_boundaries']['verts']])
            self.polygon = Polygon(self.vertices)
            self.polygon = self.polygon.buffer(1e-8)
        if 'kml_file' in data:
            self.kml_data, self.polygon, data['lat'], data['lon'] = self.kml_read(data['kml_file'])
            self.polygon = self.polygon.buffer(1e-8)

        if 'lat' not in data or 'lon' not in data:
            raise ValueError("SiteInfo requires lat and lon")
        self.lat = data['lat']
        self.lon = data['lon']

        if 'year' not in data:
            data['year'] = 2012
        if 'tz' in data:
            self.tz = data['tz']
        
        if self.solar:
            self.solar_resource = SolarResource(data['lat'], data['lon'], data['year'], filepath=self.solar_resource_file)
            self.n_timesteps = len(self.solar_resource.data['gh']) // 8760 * 8760
        if self.wave:
            self.wave_resource = WaveResource(data['lat'], data['lon'], data['year'], filepath = self.wave_resource_file)
            self.n_timesteps = 8760

        if self.wind:
            # TODO: allow hub height to be used as an optimization variable
            self.wind_resource = WindResource(data['lat'], data['lon'], data['year'], wind_turbine_hub_ht=self.hub_height,
                                                filepath=self.wind_resource_file, source=self.wind_resource_origin)
            n_timesteps = len(self.wind_resource.data['data']) // 8760 * 8760
            if self.n_timesteps is None:
                self.n_timesteps = n_timesteps
            elif self.n_timesteps != n_timesteps:
                raise ValueError(f"Wind resource timesteps of {n_timesteps} different than other resource timesteps of {self.n_timesteps}")

        self.elec_prices = ElectricityPrices(data['lat'], data['lon'], data['year'], filepath=self.grid_resource_file)
        self.n_periods_per_day = self.n_timesteps // 365  # TODO: Does not handle leap years well
        self.interval = int((60*24)/self.n_periods_per_day)
        self.urdb_label = data['urdb_label'] if 'urdb_label' in data.keys() else None

        if len(self.capacity_hours) != self.n_timesteps:
            self.capacity_hours = np.array([False] * self.n_timesteps)

        # Desired load schedule for the system to dispatch against
        self.follow_desired_schedule = len(self.desired_schedule) == self.n_timesteps
        if len(self.desired_schedule) > 0 and len(self.desired_schedule) != self.n_timesteps:
            raise ValueError('The provided desired schedule does not match length of the simulation horizon.')
            # FIXME: this a hack

        if self.wind:
            logger.info("Set up SiteInfo with wind resource files: {}".format(self.wind_resource.filename))
        if self.solar:
            logger.info("Set up SiteInfo with solar resource files: {}".format(self.solar_resource.filename))
        if self.wave:
            logger.info("Set up SiteInfo with wave resource files: {}".format(self.wave_resource.filename))

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
             linewidth=1.0
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
        if isinstance(self.polygon, Polygon):
            shape = [self.polygon]
        elif isinstance(self.polygon, MultiPolygon):
            shape = self.polygon.geoms
        for geom in shape:    
            xs, ys = geom.exterior.xy    
            plt.fill(xs, ys, alpha=0.3, fc='g', ec='none')

        plt.tick_params(which='both', labelsize=15)
        plt.xlabel('x (m)', fontsize=15)
        plt.ylabel('y (m)', fontsize=15)

        return figure, axes

    def kml_write(self, filepath, turb_coords=None, solar_region=None, wind_radius=200):
        if self.kml_data is None:
            raise AttributeError("No KML data to write.")

        if turb_coords is not None:
            turb_coords = np.atleast_2d(turb_coords)
            for n, (x, y) in enumerate(turb_coords):
                self.append_kml_data(self.kml_data, Point(x, y).buffer(wind_radius), f"Wind Turbine {n + 1}")
        if solar_region is not None:
            if isinstance(solar_region, Polygon):
                solar_region = [solar_region]
            elif isinstance(solar_region, MultiPolygon):
                solar_region = solar_region.geoms
            for n, poly in enumerate(solar_region):
                self.append_kml_data(self.kml_data, poly, f"Solar Region {n + 1}")
        with open(filepath, 'w') as kml_file:
            kml_str = self.kml_data.to_string(prettyprint=True)
            kml_file.write(kml_str)

    @staticmethod
    def kml_read(filepath):
        k = kml.KML()
        with open(filepath) as kml_file:
            k.from_string(kml_file.read().encode("utf-8"))
        features = list(k.features())[0]
        placemarks = list(list(features.features())[0].features())
        
        gmaps_epsg = pyproj.CRS("EPSG:4326")
        project = None

        valid_region = None
        for pm in placemarks:
            if "boundary" in pm.name.lower():
                shapely_object = shape(pm.geometry)
                valid_region = make_valid(shapely_object)
                lon, lat = valid_region.centroid.x, valid_region.centroid.y
                if project is None:
                    zone_num = utm.from_latlon(lat, lon)[2]
                    utm_proj= pyproj.CRS(f'EPSG:326{zone_num}')
                    project = pyproj.Transformer.from_crs(gmaps_epsg, utm_proj, always_xy=True).transform
                valid_region = transform(project, valid_region)
                break
        if valid_region is None:
            raise ValueError("KML file needs to have a placemark with a name containing 'Boundary'")
        for pm in placemarks:
            if 'exclusion' in pm.name.lower():
                try:
                    valid_region = valid_region.difference(transform(project, shape(pm.geometry.buffer(0))))
                except:
                    valid_region = valid_region.difference(transform(project, make_valid(shape(pm.geometry))))
        return k, valid_region, lat, lon

    @staticmethod
    def append_kml_data(kml_data, polygon, name):
        folder = kml_data._features[0]._features[0]
        new_pm = kml.Placemark(name=name)
        new_pm.geometry = polygon
        folder.append(new_pm)
