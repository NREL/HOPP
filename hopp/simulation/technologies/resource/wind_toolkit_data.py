from rex import WindX
from rex.sam_resource import SAMResource
import numpy as np 
from typing import Optional, Union
from pathlib import Path
import os
from hopp.simulation.technologies.resource.resource import Resource

WTK_V10_BASE = "/datasets/WIND/conus/v1.0.0/wtk_conus_"
WTK_V11_BASE = "/datasets/WIND/conus/v1.1.0/wtk_conus_"


class HPCWindData(Resource):
    """
    Args:
        lat (float): latitude
        lon (float): longitude
        year (int): year to get resource data for (valid years are 2007-2014)
        hub_height_meters (float): turbine hub height
        wtk_source_path (str): directory of wind resource data on HPC. Defaults to ""
        filepath (str): filepath for wind toolkit h5 file on HPC. Defaults to ""
            - should be formatted as: /path/to/file/name_of_file.h5
    """


    def __init__(
        self,
        lat: float,
        lon: float,
        year: int,
        hub_height_meters: float,
        wtk_source_path: Union[str,Path] = "",
        filepath: str = "",
        ):
        
        self.latitude = lat
        self.longitude = lon
        self.year = year
        super().__init__(lat, lon, year)

        self.hub_height_meters = hub_height_meters
        self.allowed_hub_heights_meters = [10, 40, 60, 80, 100, 120, 140, 160, 200]
        self.data_hub_heights = self.calculate_heights_to_download()
        

        if filepath == "" and wtk_source_path=="":
            # use default filepaths based on resource year
            if self.year < 2014:
                self.wtk_file = WTK_V10_BASE + "{}.h5".format(self.year)
            elif self.year == 2014:
                self.wtk_file = WTK_V11_BASE + "{}.h5".format(self.year)
        elif filepath != "" and wtk_source_path == "":
            # filepath (full h5 filepath) is provided by user
            self.wtk_file = filepath
        elif filepath == "" and wtk_source_path != "":
            # directory of h5 files (wtk_source_path) is provided by user
            self.wtk_file = os.path.join(str(wtk_source_path),"wtk_conus_{}.h5".format(self.year))
        else:
            # use default filepaths
            if self.year < 2014:
                self.wtk_file = WTK_V10_BASE + "{}.h5".format(self.year)
            elif self.year == 2014:
                self.wtk_file = WTK_V11_BASE + "{}.h5".format(self.year)
        
        # Pull data from HPC NSRDB dataset
        self.download_resource()

        # Set solar resource data into SAM/PySAM digestible format
        self.format_data() 

    
    def calculate_heights_to_download(self):
        """
        Given the system hub height, and the available hub heights from WindToolkit,
        determine which heights to download to bracket the hub height
        """
        hub_height_meters = self.hub_height_meters

        # evaluate hub height, determine what heights to download
        heights = [hub_height_meters]
        if hub_height_meters not in self.allowed_hub_heights_meters:
            height_low = self.allowed_hub_heights_meters[0]
            height_high = self.allowed_hub_heights_meters[-1]
            for h in self.allowed_hub_heights_meters:
                if h < hub_height_meters:
                    height_low = h
                elif h > hub_height_meters:
                    height_high = h
                    break
            heights[0] = height_low
            heights.append(height_high)

        return heights
    
    def download_resource(self):
        """load WTK h5 file using rex and get wind resource data for location
        specified by (self.lat, self.lon)
        """
        # NOTE: Current setup of files on HPC WINDToolkit v1.0.0 = 2007-2013, v1.1.0 = 2014
    
        # Open file with rex WindX object
        with WindX(self.wtk_file, hsds=False) as f:
            # get gid of location closest to given lat/lon coordinates and timezone offset
            site_gid = f.lat_lon_gid((self.latitude, self.longitude))
            time_zone = f.meta['timezone'].iloc[site_gid]

            # instantiate temp dictionary to hold each attributes dataset
            self.wind_dict = {}
            # loop through hub heights to download, capture datasets
            # NOTE: datasets are not auto shifted by timezone offset -> wrap extraction in SAMResource.roll_timeseries(input_array, timezone, #steps in an hour=1) to roll timezones
            # NOTE: pressure datasets unit = Pa, convert to atm via division by 101325
            for h in self.data_hub_heights:
                self.wind_dict['temperature_{height}m_arr'.format(height=h)] = SAMResource.roll_timeseries((f['temperature_{height}m'.format(height=h), :, site_gid]), time_zone, 1)
                self.wind_dict['pressure_{height}m_arr'.format(height=h)] = SAMResource.roll_timeseries((f['pressure_{height}m'.format(height=h), :, site_gid]/101325), time_zone, 1)
                self.wind_dict['windspeed_{height}m_arr'.format(height=h)] = SAMResource.roll_timeseries((f['windspeed_{height}m'.format(height=h), :, site_gid]), time_zone, 1)
                self.wind_dict['winddirection_{height}m_arr'.format(height=h)] = SAMResource.roll_timeseries((f['winddirection_{height}m'.format(height=h), :, site_gid]), time_zone, 1)    

            self.site_gid = site_gid
    def format_data(self):
        # Remove data from feb29 on leap years
        if (self.year % 4) == 0:
            feb29 = np.arange(1416,1440)
            for key, value in self.wind_dict.items():
                self.wind_dict[key] = np.delete(value, feb29)

        # round to desired precision and concatenate data into format needed for data dictionary
        if len(self.data_hub_heights) == 2:
            # NOTE: Unsure if SAM/PySAM is sensitive to data types ie: floats with long precision vs to 2 or 3 decimals. 
            # If not sensitive, can remove following 8 lines of code to increase computational efficiency
            self.wind_dict['temperature_{h}m_arr'.format(h=self.data_hub_heights[0])] = np.round((self.wind_dict['temperature_{h}m_arr'.format(h=self.data_hub_heights[0])]), decimals=1)
            self.wind_dict['pressure_{h}m_arr'.format(h=self.data_hub_heights[0])] = np.round((self.wind_dict['pressure_{h}m_arr'.format(h=self.data_hub_heights[0])]), decimals=2)
            self.wind_dict['windspeed_{h}m_arr'.format(h=self.data_hub_heights[0])] = np.round((self.wind_dict['windspeed_{h}m_arr'.format(h=self.data_hub_heights[0])]), decimals=3)
            self.wind_dict['winddirection_{h}m_arr'.format(h=self.data_hub_heights[0])] = np.round((self.wind_dict['winddirection_{h}m_arr'.format(h=self.data_hub_heights[0])]), decimals=1)
            self.wind_dict['temperature_{h}m_arr'.format(h=self.data_hub_heights[1])] = np.round((self.wind_dict['temperature_{h}m_arr'.format(h=self.data_hub_heights[1])]), decimals=1)
            self.wind_dict['pressure_{h}m_arr'.format(h=self.data_hub_heights[1])] = np.round((self.wind_dict['pressure_{h}m_arr'.format(h=self.data_hub_heights[1])]), decimals=2)
            self.wind_dict['windspeed_{h}m_arr'.format(h=self.data_hub_heights[1])] = np.round((self.wind_dict['windspeed_{h}m_arr'.format(h=self.data_hub_heights[1])]), decimals=3)
            self.wind_dict['winddirection_{h}m_arr'.format(h=self.data_hub_heights[1])] = np.round((self.wind_dict['winddirection_{h}m_arr'.format(h=self.data_hub_heights[1])]), decimals=1)
            # combine all data into one 2D list
            self.combined_data = [list(a) for a in zip(self.wind_dict['temperature_{h}m_arr'.format(h=self.data_hub_heights[0])],
                                                       self.wind_dict['pressure_{h}m_arr'.format(h=self.data_hub_heights[0])],
                                                       self.wind_dict['windspeed_{h}m_arr'.format(h=self.data_hub_heights[0])],
                                                       self.wind_dict['winddirection_{h}m_arr'.format(h=self.data_hub_heights[0])],
                                                       self.wind_dict['temperature_{h}m_arr'.format(h=self.data_hub_heights[1])],
                                                       self.wind_dict['pressure_{h}m_arr'.format(h=self.data_hub_heights[1])],
                                                       self.wind_dict['windspeed_{h}m_arr'.format(h=self.data_hub_heights[1])],
                                                       self.wind_dict['winddirection_{h}m_arr'.format(h=self.data_hub_heights[1])])]

        elif len(self.data_hub_heights) == 1:
            # NOTE: Unsure if SAM/PySAM is sensitive to data types ie: floats with long precision vs to 2 or 3 decimals. If not sensitive, can remove following 4 lines of code to increase computational efficiency
            self.wind_dict['temperature_{h}m_arr'.format(h=self.data_hub_heights[0])] = np.round((self.wind_dict['temperature_{h}m_arr'.format(h=self.data_hub_heights[0])]), decimals=1)
            self.wind_dict['pressure_{h}m_arr'.format(h=self.data_hub_heights[0])] = np.round((self.wind_dict['pressure_{h}m_arr'.format(h=self.data_hub_heights[0])]), decimals=2)
            self.wind_dict['windspeed_{h}m_arr'.format(h=self.data_hub_heights[0])] = np.round((self.wind_dict['windspeed_{h}m_arr'.format(h=self.data_hub_heights[0])]), decimals=3)
            self.wind_dict['winddirection_{h}m_arr'.format(h=self.data_hub_heights[0])] = np.round((self.wind_dict['winddirection_{h}m_arr'.format(h=self.data_hub_heights[0])]), decimals=1)
            # combine all data into one 2D list
            self.combined_data = [list(a) for a in zip(self.wind_dict['temperature_{h}m_arr'.format(h=self.data_hub_heights[0])],
                                                       self.wind_dict['pressure_{h}m_arr'.format(h=self.data_hub_heights[0])],
                                                       self.wind_dict['windspeed_{h}m_arr'.format(h=self.data_hub_heights[0])],
                                                       self.wind_dict['winddirection_{h}m_arr'.format(h=self.data_hub_heights[0])])]
        self.data = self.combined_data
    
    @Resource.data.setter
    def data(self, data_file):
        """Wind resource data formatted for SAM

        Output: self.data
            dictionary:
                heights: list(int): floats corresponding to hub-height for 'data' entry
                fields list(int): integers corresponding to data type for 'data' entry
                data:
                    temperature: Ambient temperature in degrees Celsius
                    pressure: Atmospheric pressure in in atmospheres.
                    windspeed: Wind speed in meters per second (m/s)
                    winddirection: Wind direction in degrees east of north (degrees).
                        Zero degrees indicates wind from the north, and 90 degrees indicates wind from the east.

        """
        dic = {
            'heights': [float(h) for h in self.data_hub_heights for i in range(4)],
            'fields':  [1, 2, 3, 4] * len(self.data_hub_heights),
            'data':    data_file #self.combined_data
            }
        self._data = dic