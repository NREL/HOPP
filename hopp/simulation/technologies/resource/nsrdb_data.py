from rex import NSRDBX
from rex.sam_resource import SAMResource
import numpy as np 
from hopp.simulation.technologies.resource.resource import Resource
from typing import Optional, Union
from pathlib import Path
import os
NSRDB_DEP = "/datasets/NSRDB/deprecated_v3/nsrdb_"

# NOTE: Current version of PSM v3.2.2 which corresponds to /api/nsrdb/v2/solar/psm3-2-2-download 
NSRDB_NEW = "/datasets/NSRDB/current/nsrdb_"

# Pull Solar Resource Data directly from NSRDB on HPC
# To be called instead of SolarResource from hopp.simulation.technologies.resource
class HPCSolarData(Resource):
    """
    Class to manage Solar Resource data from NSRDB Datasets

    Args:
        lat (float): site latitude
        lon (float): site longitude
        year: year to get resource data for
        nsrdb_source_path (str): directory where NSRDB data is hosted on HPC. Defaults to ""
        filepath (str): filepath to NSRDB h5 file on HPC. Defaults to "".
            - should be formatted as: /path/to/file/name_of_file.h5
    """


    def __init__(
        self,
        lat: float,
        lon: float,
        year: int,
        nsrdb_source_path: Union[str,Path] = "",
        filepath: str = "",
        ):
        """
        Output: self.data
            dictionary:
                tz: float
                elev: float
                lat: float
                lon: float
                year: list of floats
                month: list of floats
                day: list of floats
                hour: list of floats
                minute: list of floats
                dn: list of floats
                df: list of floats
                gh: list of floats
                wspd: list of floats
                tdry: list of floats
                pres: list of floats
                tdew: list of floats
        """
        # NOTE: self.data must be compatible with PVWatts.SolarResource.solar_resource_data to https://nrel-pysam.readthedocs.io/en/main/modules/Pvwattsv8.html#PySAM.Pvwattsv8.Pvwattsv8.SolarResource
        self.latitude = lat
        self.longitude = lon
        self.year = year
        super().__init__(lat, lon, year)

        if filepath == "" and nsrdb_source_path=="":
            # use default filepath
            self.nsrdb_file = NSRDB_NEW + "{}.h5".format(self.year)
        elif filepath != "" and nsrdb_source_path == "":
            # filepath (full h5 filepath) is provided by user
            self.nsrdb_file = filepath
        elif filepath=="" and nsrdb_source_path !="":
            # directory of h5 files (nsrdb_source_path) is provided by user
            self.nsrdb_file = os.path.join(str(nsrdb_source_path),"nsrdb_{}.h5".format(self.year))
        else:
            # use default filepaths
            self.nsrdb_file = NSRDB_NEW + "{}.h5".format(self.year)
        
        # Pull data from HPC NSRDB dataset
        self.download_resource()

        # Set solar resource data into SAM/PySAM digestible format
        self.format_data()
        

    def download_resource(self):
        """load NSRDB h5 file using rex and get solar resource data for location
        specified by (self.lat, self.lon)
        """
        
        # Open file with rex NSRDBX object
        with NSRDBX(self.nsrdb_file, hsds=False) as f:
            # get gid of location closest to given lat/lon coordinates
            site_gid = f.lat_lon_gid((self.latitude,self.longitude))

            # extract timezone, elevation, latitude and longitude from meta dataset with gid
            self.time_zone = f.meta['timezone'].iloc[site_gid]
            self.elevation = f.meta['elevation'].iloc[site_gid]
            self.nsrdb_latitude = f.meta['latitude'].iloc[site_gid]
            self.nsrdb_longitude = f.meta['longitude'].iloc[site_gid]
            
            # extract remaining datapoints: year, month, day, hour, minute, dn, df, gh, wspd,tdry, pres, tdew
            # NOTE: datasets have readings at 0 and 30 minutes each hour, HOPP/SAM workflow requires only 30 minute reading values -> filter 0 minute readings with [1::2]
            # NOTE: datasets are not auto shifted by timezone offset -> wrap extraction in SAMResource.roll_timeseries(input_array, timezone, #steps in an hour=1) to roll timezones
            # NOTE: solar_resource.py code references solar_zenith_angle and RH = relative_humidity but I couldn't find them actually being utilized. Captured them below just in case.
            self.year_arr = f.time_index.year.values[1::2]
            self.month_arr = f.time_index.month.values[1::2]
            self.day_arr = f.time_index.day.values[1::2]
            self.hour_arr = f.time_index.hour.values[1::2]
            self.minute_arr = f.time_index.minute.values[1::2]
            self.dni_arr = SAMResource.roll_timeseries((f['dni', :, site_gid][1::2]), self.time_zone, 1)
            self.dhi_arr = SAMResource.roll_timeseries((f['dhi', :, site_gid][1::2]), self.time_zone, 1)
            self.ghi_arr = SAMResource.roll_timeseries((f['ghi', :, site_gid][1::2]), self.time_zone, 1)
            self.wspd_arr = SAMResource.roll_timeseries((f['wind_speed', :, site_gid][1::2]), self.time_zone, 1)
            self.tdry_arr = SAMResource.roll_timeseries((f['air_temperature', :, site_gid][1::2]), self.time_zone, 1)
            # self.relative_humidity_arr = SAMResource.roll_timeseries((f['relative_humidity', :, site_gid][1::2]), self.time_zone, 1)
            # self.solar_zenith_arr = SAMResource.roll_timeseries((f['solar_zenith_angle', :, site_gid][1::2]), self.time_zone, 1)
            self.pres_arr = SAMResource.roll_timeseries((f['surface_pressure', :, site_gid][1::2]), self.time_zone, 1)
            self.tdew_arr = SAMResource.roll_timeseries((f['dew_point', :, site_gid][1::2]), self.time_zone, 1)

            self.site_gid = site_gid
            

    def format_data(self):
        # Remove data from feb29 on leap years
        if (self.year % 4) == 0:
            feb29 = np.arange(1416,1440)
            self.year_arr = np.delete(self.year_arr, feb29)
            self.month_arr = np.delete(self.month_arr, feb29)
            self.day_arr = np.delete(self.day_arr, feb29)
            self.hour_arr = np.delete(self.hour_arr, feb29)
            self.minute_arr = np.delete(self.minute_arr, feb29)
            self.dni_arr = np.delete(self.dni_arr, feb29)
            self.dhi_arr = np.delete(self.dhi_arr, feb29)
            self.ghi_arr = np.delete(self.ghi_arr, feb29)
            self.wspd_arr = np.delete(self.wspd_arr, feb29)
            self.tdry_arr = np.delete(self.tdry_arr, feb29)
            # self.relative_humidity_arr = np.delete(self.relative_humidity_arr, feb29)
            # self.solar_zenith_arr = np.delete(self.solar_zenith_arr, feb29)
            self.pres_arr = np.delete(self.pres_arr, feb29)
            self.tdew_arr = np.delete(self.tdew_arr, feb29)

        # round to desired precision and convert to desired data type
        # NOTE: unsure if SAM/PySAM is sensitive to data types and decimal precision. 
        # If not sensitive, can remove .astype() and round() to increase computational efficiency
        self.time_zone = float(self.time_zone)
        self.elevation = round(float(self.elevation), 0)
        self.nsrdb_latitude = round(float(self.nsrdb_latitude), 2)
        self.nsrdb_longitude = round(float(self.nsrdb_longitude),2)
        self.year_arr = list(self.year_arr.astype(float, copy=False))
        self.month_arr = list(self.month_arr.astype(float, copy=False))
        self.day_arr = list(self.day_arr.astype(float, copy=False))
        self.hour_arr = list(self.hour_arr.astype(float, copy=False))
        self.minute_arr = list(self.minute_arr.astype(float, copy=False))
        self.dni_arr = list(self.dni_arr.astype(float, copy=False))
        self.dhi_arr = list(self.dhi_arr.astype(float, copy=False))
        self.ghi_arr = list(self.ghi_arr.astype(float, copy=False))
        self.wspd_arr = list(self.wspd_arr.astype(float, copy=False))
        self.tdry_arr = list(self.tdry_arr.astype(float, copy=False))
        # self.relative_humidity_arr = list(np.round(self.relative_humidity_arr, decimals=1))
        # self.solar_zenith_angle_arr = list(np.round(self.solar_zenith_angle_arr, decimals=1))
        self.pres_arr = list(self.pres_arr.astype(float, copy=False))
        self.tdew_arr = list(self.tdew_arr.astype(float, copy=False))
        self.data = {} #unsure if this will cause problem
    
    @Resource.data.setter
    def data(self,data_dict):
        dic = {
            # 'site_gid': self.site_gid,
            # 'nsrdb_lat':self.nsrdb_latitude,
            # 'nsrdb_lon':self.nsrdb_longitude,
            'tz' :     self.time_zone,
            'elev' :   self.elevation,
            'lat' :    self.nsrdb_latitude,
            'lon' :    self.nsrdb_longitude,
            'year' :   self.year_arr,
            'month' :  self.month_arr,
            'day' :    self.day_arr,
            'hour' :   self.hour_arr,
            'minute' : self.minute_arr,
            'dn' :     self.dni_arr,
            'df' :     self.dhi_arr,
            'gh' :     self.ghi_arr,
            'wspd' :   self.wspd_arr,
            'tdry' :   self.tdry_arr,
            'pres' :   self.pres_arr,
            'tdew' :   self.tdew_arr
            }
        self._data = dic