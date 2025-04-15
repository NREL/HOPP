from typing import Optional, Union
from pathlib import Path

from attrs import define, field
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon, MultiPolygon, Point, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from shapely.validation import make_valid
from fastkml import kml, KML
import pyproj
import utm

from hopp.simulation.technologies.resource import (
    SolarResource,
    WindResource,
    WaveResource,
    TidalResource,
    ElectricityPrices,
    HPCWindData,
    HPCSolarData,
    AlaskaWindData,
    BCHRRRWindData,
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
import hopp.simulation.technologies.sites.site_shape_tools as shape_tools
from hopp import ROOT_DIR
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
        wave_resource_file: Path to wave resource file. Defaults to "".
        tidal_resource_file: Path to tidal resource file. Defaults to "".
        grid_resource_file: Path to grid pricing data file. Defaults to "".
        path_resource: Path to folder to save resource files. 
            Defaults to ROOT/simulation/resource_files.
        wtk_source_path (Optional): directory of Wind Toolkit h5 files hosted on HPC.
            Only used if renewable_resource_origin != "API".
        nsrdb_source_path (Optional): directory of NSRDB h5 files hosted on HPC.
            Only used if renewable_resource_origin != "API".
        hub_height: Turbine hub height for resource download in meters. Defaults to 97.0.
        capacity_hours: Boolean list indicating hours for capacity payments. Defaults to [].
        desired_schedule: Absolute desired load profile in MWe. Defaults to [].
        curtailment_value_type: whether to curtail power above grid interconnection limit or desired schedule. 
            Options "interconnect_kw" or "desired_schedule". Defaults to "interconnect_kw".
        solar: Whether to set solar data for this site. Defaults to True.
        wind: Whether to set wind data for this site. Defaults to True.
        wave: Whether to set wave data for this site. Defaults to False.
        tidal: Whether to set tidal data for this site. Defaults to False.
        renewable_resource_origin (str): whether to download resource data from API or load directly from datasets files.
            Options are "API" or "HPC". Defaults to "API".
        wind_resource_origin: Which wind resource API to use, defaults to "WTK" for WIND Toolkit.
            Options are "WTK", "TAP" or "BC-HRRR".
        site_buffer (Optional): value to buffer site polygon. Defaults to 1e-8.
        solar_resource (Optional): dictionary or object containing solar resource data.
        wind_resource (Optional): dictionary or object containing wind resource data.
        wind_resource_region (Optional): which region to use for wind resource data. Defaults to "conus". Options are: 
            
            - "conus": continental United States
            - "ak": Alaska
    """
    # User provided
    data: dict
    """dictionary of site info data with key as:

        - **lat** (*float*): site latitude.
        - **lon** (*float*): site longitude.
        - **elev** (*int, Optional*): elevation of site (m).
        - **year** (*int*): year to get resource data for. Defaults to 2012.
        - **tz** (*int, Optional*): timezone of site.
        - **site_boundaries** (*dict,Optional*):
            - **verts** (*list[list[float]]*): vertices of site polygon. list of [x,y] coordinates in meters.
            - **verts_simple** (*list[list[float]]*): TODO
        - **site_details** (*dict, Optional*):
            - **site_area_m2** (*float*): area of site in square meters.
            - **site_area_km2** (*float*): area of site in square kilometers. required if ``site_area_m2`` is not provided.
            - **site_shape** (*str, Optional*): shape of site area. Options are "circle", "rectangle", "square" or "hexagon". Defaults to "square".
            - **x0** (*float, Optional*): left-most x coordinate of the site in meters. Defaults to 0.0.
            - **y0** (*float, Optional*): bottom-most x coordinate of the site in meters. Defaults to 0.0.
            - **aspect_ratio** (*float, Optional*): aspect ratio (width/height) 
                Only used if ``site_shape`` is set as "rectangle". Defaults to 1.5.
            - **degrees_between_points** (*float | int, Optional*): difference in degrees for generating circular boundary. 
                Only used if ``site_shape`` is set as "circle". Defaults to 10.
        - **solar_lat** (*float, Optional*): latitude to get solar resource data if solar plant is in a different location than lat/lon. Defaults to **lat** value above.
        - **solar_lon** (*float, Optional*): longitude to get solar resource data if solar plant is in a different location than lat/lon. Defaults to **lon** value above.
        - **solar_year** (*int, Optional*): resource year for solar data if wanting a different resource year than ``data["year"]``. Defaults to **year** value above.
        - **wind_lat** (*float, Optional*): latitude to get wind resource data if wind plant is in a different location than lat/lon. Defaults to **lat** value above.
        - **wind_lon** (*float, Optional*): longitude to get wind resource data if wind plant is in a different location than lat/lon. Defaults to **lon** value above.
        - **wind_year** (*int, Optional*): resource year for wind data if wanting different resource than ``data["year"]``. Defaults to **year** value above.
        - **urdb_label** (*str, Optional*): string corresponding to data from utility rate database. Defaults to None.
    """

    
    solar_resource_file: Union[Path, str] = field(default="", converter=resource_file_converter)
    wind_resource_file: Union[Path, str] = field(default="", converter=resource_file_converter)
    wave_resource_file: Union[Path, str] = field(default="", converter=resource_file_converter)
    tidal_resource_file: Union[Path, str] = field(default="", converter=resource_file_converter)
    grid_resource_file: Union[Path, str] = field(default="", converter=resource_file_converter)

    path_resource: Optional[Union[Path, str]] = field(default=ROOT_DIR / "simulation" / "resource_files")
    wtk_source_path: Optional[Union[Path,str]] = field(default = "")
    nsrdb_source_path: Optional[Union[Path,str]] = field(default = "")

    hub_height: hopp_float_type = field(default=97., converter=hopp_float_type)
    capacity_hours: NDArray = field(default=[], converter=converter(bool))
    desired_schedule: NDArrayFloat = field(default=[], converter=converter())
    curtailment_value_type: str = field(default="interconnect_kw", validator=contains(["interconnect_kw", "desired_schedule"]))

    solar: bool = field(default=True)
    wind: bool = field(default=True)
    wave: bool = field(default=False)
    tidal: bool = field(default=False)
    renewable_resource_origin: str = field(default="API", validator=contains(["API", "HPC"]))
    wind_resource_origin: str = field(default="WTK", validator=contains(["WTK", "TAP", "BC-HRRR"]))
    wind_resource_region: str = field(default="conus", validator=contains(["conus", "ak"]), converter=(str.strip, str.lower))

    site_buffer: Optional[float] = field(default = 1e-8)

    # Set in post init hook
    lat: hopp_float_type = field(init=False)
    lon: hopp_float_type = field(init=False)
    elev: Optional[float] = field(init=False, default=None)
    year: int = field(init=False, default=2012)
    tz: Optional[int] = field(init=False, default=None)
    vertices: NDArrayFloat = field(init=False)
    polygon: Union[Polygon, BaseGeometry] = field(init=False)
    solar_resource: Optional[Union[SolarResource,HPCSolarData]] = field(default=None)
    wind_resource: Optional[Union[WindResource,HPCWindData,AlaskaWindData,BCHRRRWindData]] = field(default=None)
    wave_resource: Optional[WaveResource] = field(init=False, default=None)
    tidal_resource: Optional[TidalResource] = field(init=False, default=None)
    elec_prices: Optional[ElectricityPrices] = field(init=False, default=None)
    n_timesteps: int = field(init=False, default=None)
    n_periods_per_day: int = field(init=False)
    interval: int = field(init=False)
    urdb_label: str = field(init=False)
    follow_desired_schedule: bool = field(init=False)
    kml_data: Optional[KML] = field(init=False, default=None)

    # .. TODO: Can we get rid of verts_simple and simplify site_boundaries

    def __attrs_post_init__(self):
        """
        The following are set in this post init hook:
            lat (numpy.float64): Site latitude in decimal degrees.
            lon (numpy.float64): Site longitude in decimal degrees.
            elev (float, Optional): Elevation of the site in meters. Defaults to None.
            year(int): Resource data year.
            tz (int, Optional): Timezone code for metadata purposes only. Defaults to None.
            vertices (:obj:`NDArray`): Site boundary vertices in meters.
            polygon (:obj:`shapely.geometry.polygon.Polygon`): Site polygon.
            solar_resource (:obj:`hopp.simulation.technologies.resource.SolarResource`): Class containing solar resource data.
            wind_resource (:obj:`hopp.simulation.technologies.resource.WindResource`): Class containing wind resource data.
            wave_resource (:obj:`hopp.simulation.technologies.resource.WaveResource`): Class containing wave resource data.
            tidal_resource (:obj:`hopp.simulation.technologies.resource.TidalResource`): Class containing tidal resource data.
            elec_prices (:obj:`hopp.simulation.technologies.resource.ElectricityPrices`): Class containing electricity prices.
            n_timesteps (int): Number of timesteps in resource data.
            n_periods_per_day (int): Number of time periods per day.
            interval (int): Number of minutes per time interval.
            urdb_label (str): Link to `Utility Rate DataBase <https://openei.org/wiki/Utility_Rate_Database>`_ label for REopt runs.
            follow_desired_schedule (bool): Indicates if a desired schedule was provided. Defaults to False.
            kml_data (KML, Optional): KML data to be used when definining site boundaries.
        """
        if self.renewable_resource_origin=="API":
            set_nrel_key_dot_env()

        data = self.data
        self.polygon,self.vertices = self.create_site_polygon(data)
        
        if 'kml_file' in data:
            self.kml_data, self.polygon, data['lat'], data['lon'] = self.kml_read(data['kml_file'])
            self.polygon = self.polygon.buffer(self.site_buffer)

        if 'lat' not in data or 'lon' not in data:
            raise ValueError("SiteInfo requires lat and lon")
        self.lat = data['lat']
        self.lon = data['lon']

        if 'year' not in data:
            data['year'] = self.year
        
        self.year = data["year"]

        if 'tz' in data:
            self.tz = data['tz']
        
        if 'elev' in data:
            self.elev = data['elev']
        
        if self.solar:
            self.solar_resource = self.initialize_solar_resource(data)
            self.n_timesteps = len(self.solar_resource.data['gh']) // 8760 * 8760
            data.setdefault("elev", self.solar_resource.data["elev"])
            data.setdefault("tz", self.solar_resource.data["tz"])
            if self.tz is None:
                self.tz = data['tz']
            if self.elev is None:
                self.elev = data['elev']
        if self.wave:
            self.wave_resource = WaveResource(data['lat'], data['lon'], data['year'], filepath = self.wave_resource_file)
            self.n_timesteps = 8760
        if self.tidal:
            self.tidal_resource = TidalResource(data['lat'], data['lon'], data['year'], filepath = self.tidal_resource_file)
            self.n_timesteps = 8760
        if self.wind:
            # TODO: allow hub height to be used as an optimization variable
            self.wind_resource = self.initialize_wind_resource(data)
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
            logger.info("Set up SiteInfo with wind resource file: {}".format(self.wind_resource.filename))
        if self.solar:
            logger.info("Set up SiteInfo with solar resource file: {}".format(self.solar_resource.filename))
        if self.wave:
            logger.info("Set up SiteInfo with wave resource file: {}".format(self.wave_resource.filename))
        if self.tidal:
            logger.info("Set up SiteInfo with tidal resource file: {}".format(self.tidal_resource.filename))
    def create_site_polygon(self,data:dict):
        """function to create site polygon.

        Args:
            data (dict): dictionary of site info data

        Returns:
            2-element tuple containing

            - **poly** (:obj:`shapely.geometry.Polygon`): site boundary polygon
            - **vertices** (2D :obj:`numpy.ndarray`): vertices of site polygon. list of [x,y] coordinates in meters.
        """
        polygon = None
        vertices = None
        if 'site_boundaries' in data: 
            if 'verts' in data['site_boundaries']: 
                vertices = np.array(data["site_boundaries"]["verts"])
                polygon = Polygon(vertices)
                polygon = polygon.buffer(self.site_buffer) #why is this needed?
        elif 'site_details' in data:
            if 'site_area_m2' in data["site_details"] or 'site_area_km2' in data["site_details"]:
                if 'site_area_km2' in data["site_details"]:
                    data["site_details"].update({"site_area_m2": data["site_details"]["site_area_km2"]*1e6})
                data["site_details"].setdefault("site_shape", "square")
                data["site_details"].setdefault("x0", 0.0)
                data["site_details"].setdefault("y0", 0.0)
                polygon, vertices = self.make_site_polygon_from_shape(data["site_details"])
                polygon = polygon.buffer(self.site_buffer)
        return polygon, vertices

    def make_site_polygon_from_shape(self,site_details:dict):
        """create site polygon and vertices if "site_details" provided in ``data``.

        Args:
            site_details (dict): sub-dictionary of ``data``, equivalent to ``data["site_details"]``

        Raises:
            ValueError: if ``site_details["site_shape"]`` is not one of the following: "circle", "square", "rectangle", or "hexagon"

        Returns:
            2-element tuple containing

            - **poly** (:obj:`shapely.geometry.Polygon`): site boundary polygon
            - **vertices** (2D :obj:`numpy.ndarray`): vertices of site polygon. list of [x,y] coordinates in meters.
        """
        if (shape := site_details.get("site_shape", None)) is None:
            return None, None

        shape = shape.lower()
        if shape == "circle":
            site_details.setdefault("degrees_between_points", 10.0)
            polygon, vertices = shape_tools.make_circle(
                area_m2 = site_details['site_area_m2'], 
                deg_diff = site_details["degrees_between_points"],
                x0 = site_details["x0"], 
                y0 = site_details["y0"]
            )
            return polygon, vertices
        if shape == "square":
            polygon, vertices = shape_tools.make_square(
                area_m2 = site_details['site_area_m2'], 
                x0 = site_details["x0"], 
                y0 = site_details["y0"]
            )
            return polygon, vertices
        if shape == "rectangle":
            site_details.setdefault("aspect_ratio", 1.5)
            polygon, vertices = shape_tools.make_rectangle(
                area_m2 = site_details['site_area_m2'],
                aspect_ratio = site_details["aspect_ratio"], 
                x0 = site_details["x0"], 
                y0 = site_details["y0"]
            )
            return polygon, vertices
        if shape == "hexagon":
            polygon, vertices = shape_tools.make_hexagon(
                area_m2 = site_details['site_area_m2'], 
                x0 = site_details["x0"], 
                y0 = site_details["y0"]
            )
            return polygon, vertices
        
        raise ValueError("invalid entry for `site_shape`, site_shape must be either 'circle', 'rectangle', 'square' or 'hexagon'")
        
        
    
    def initialize_solar_resource(self,data:dict):
        """Download/load solar resource data

        Args:
            data (dict): Dictionary containing site-specific information.

        Returns:
            :obj:`hopp.simulation.technologies.resource.SolarResource` or :obj:`hopp.simulation.technologies.resource.HPCSolarData`: solar resource data class
        """
        
        solar_lat = data.setdefault("solar_lat", data["lat"])
        solar_lon = data.setdefault("solar_lon", data["lon"])
        solar_year = data.setdefault("solar_year", data["year"])

        if self.solar_resource is None:
            if self.renewable_resource_origin == "API":
                solar_resource = SolarResource(solar_lat, solar_lon, solar_year, path_resource=self.path_resource, filepath=self.solar_resource_file)
            else:
                solar_resource = HPCSolarData(solar_lat, solar_lon, solar_year,nsrdb_source_path = self.nsrdb_source_path, filepath=self.solar_resource_file)
            return solar_resource
        if isinstance(self.solar_resource,dict):
            solar_resource = SolarResource(solar_lat, solar_lon, solar_year,resource_data = self.solar_resource)
            return solar_resource
        
        return self.solar_resource

    def initialize_wind_resource(self, data: dict):
        """Download/load wind resource data

        Args:
            data (dict): Dictionary containing site-specific information.

        Returns:
            :obj:`hopp.simulation.technologies.resource.WindResource` or :obj:`hopp.simulation.technologies.resource.HPCWindData`: wind resource data class
        """
        # Extract parameters with defaults from data dictionary
        wind_lat = data.setdefault("wind_lat", data["lat"])
        wind_lon = data.setdefault("wind_lon", data["lon"])
        wind_year = data.setdefault("wind_year", data["year"])
        
        # If wind resource is already provided as an object, return it directly
        if self.wind_resource is not None and not isinstance(self.wind_resource, dict):
            return self.wind_resource
        
        # If wind resource is provided as a dictionary, convert to appropriate object
        if isinstance(self.wind_resource, dict):
            if self.wind_resource_region == "conus":
                return WindResource(wind_lat, wind_lon, wind_year, 
                                   wind_turbine_hub_ht=self.hub_height, 
                                   resource_data=self.wind_resource)
            elif self.wind_resource_region == "ak":
                return AlaskaWindData(lat=wind_lat, lon=wind_lon, year=wind_year, 
                                     hub_height_meters=self.hub_height, 
                                     resource_data=self.wind_resource)
        
        # Create new wind resource based on region and resource origin
        if self.wind_resource_region == "ak":
            return AlaskaWindData(lat=wind_lat, lon=wind_lon, year=wind_year, 
                                 hub_height_meters=self.hub_height,
                                 path_resource=self.path_resource, 
                                 filename=self.wind_resource_file)
        
        # Handle Continental US (conus) region
        if self.renewable_resource_origin == "API":
            if self.wind_resource_origin in ["WTK", "TAP"]:
                return WindResource(wind_lat, wind_lon, wind_year, 
                                   wind_turbine_hub_ht=self.hub_height,
                                   path_resource=self.path_resource, 
                                   filepath=self.wind_resource_file, 
                                   source=self.wind_resource_origin)
            elif self.wind_resource_origin == "BC-HRRR":
                return BCHRRRWindData(wind_lat, wind_lon, wind_year, 
                                     hub_height_meters=self.hub_height,
                                     path_resource=self.path_resource, 
                                     filename=self.wind_resource_file)
            else:
                raise ValueError("Invalid entry for `wind_resource_origin`, must be either 'WTK', 'TAP' or 'BC-HRRR'")
        elif self.renewable_resource_origin == "HPC":
            return HPCWindData(wind_lat, wind_lon, wind_year, 
                              wind_turbine_hub_ht=self.hub_height,
                              wtk_source_path=self.wtk_source_path, 
                              filepath=self.wind_resource_file)

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
