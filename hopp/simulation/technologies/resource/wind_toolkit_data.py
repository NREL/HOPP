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
    Class to manage Wind Resource data from Wind Toolkit Datasets
    
    Attributes:
        wtk_file: (str) path of file that resource data is pulled from
        site_gid: (int) id for Wind Toolkit location that resource data was pulled from
        wtk_latitude: (float) latitude of Wind Toolkit location corresponding to site_gid
        wtk_longitude: (float) longitude of Wind Toolkit location corresponding to site_gid
    """


    def __init__(
        self,
        lat: float,
        lon: float,
        year: int,
        wind_turbine_hub_ht: float,
        wtk_source_path: Union[str,Path] = "",
        filepath: str = "",
        ):
        """Class to pull wind resource data from WIND Toolkit datasets hosted on the HPC

        Args:
            lat (float): latitude corresponding to location for wind resource data
            lon (float): longitude corresponding to location for wind resource data
            year (int): year for resource data. must be between 2007 and 2014
            wind_turbine_hub_ht (float): turbine hub height (m)
            wtk_source_path (Union[str,Path], optional): directory where Wind Toolkit data is hosted on HPC. Defaults to "".
            filepath (str, optional): filepath to Wind Toolkit h5 file on HPC. Defaults to "".
                - should be formatted as: /path/to/file/name_of_file.h5
        Raises:
            ValueError: if year is not between 2007 and 2014 (inclusive)
            FileNotFoundError: if wtk_file is not valid filepath
        """
        super().__init__(lat, lon, year)

        self.hub_height_meters = wind_turbine_hub_ht
        self.allowed_hub_heights_meters = [10, 40, 60, 80, 100, 120, 140, 160, 200]
        self.data_hub_heights = self.calculate_heights_to_download()
        
        # Check for valid year
        if self.year < 2007 or self.year > 2014:
            raise ValueError(f"Resource year for WIND Toolkit Data must be between 2007 and 2014 but {self.year} was provided")
        
        if filepath == "" and wtk_source_path=="":
            # use default filepaths based on resource year
            if self.year < 2014 and self.year>=2007:
                self.wtk_file = WTK_V10_BASE + f"{self.year}.h5"
            elif self.year == 2014:
                self.wtk_file = WTK_V11_BASE + f"{self.year}.h5"
        elif filepath != "" and wtk_source_path == "":
            # filepath (full h5 filepath) is provided by user
            if ".h5" not in filepath:
                filepath = filepath + ".h5"
            self.wtk_file = str(filepath)
        elif filepath == "" and wtk_source_path != "":
            # directory of h5 files (wtk_source_path) is provided by user
            self.wtk_file = os.path.join(str(wtk_source_path),f"wtk_conus_{self.year}.h5")
        else:
            # use default filepaths
            if self.year < 2014 and self.year>=2007:
                self.wtk_file = WTK_V10_BASE + f"{self.year}.h5"
            elif self.year == 2014:
                self.wtk_file = WTK_V11_BASE + f"{self.year}.h5"
        
        # Check for valid filepath for Wind Toolkit file
        if not os.path.isfile(self.wtk_file):
            raise FileNotFoundError(f"Cannot find Wind Toolkit .h5 file, filepath {self.wtk_file} does not exist")
        
        # Pull data from HPC Wind Toolkit dataset
        self.download_resource()

        # Set wind resource data into SAM/PySAM digestible format
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
            # NOTE: datasets are not auto shifted by timezone offset 
            # -> wrap extraction in SAMResource.roll_timeseries(input_array, timezone, #steps in an hour=1) to roll timezones
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
            # NOTE: Unsure if SAM/PySAM is sensitive to data types ie: floats with long precision vs to 2 or 3 decimals. 
            # If not sensitive, can remove following 4 lines of code to increase computational efficiency
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
    def data(self, combined_data):
        """Sets data property with wind resource data formatted for SAM

            data (dict):
                :key heights (list(float)): floats corresponding to hub-height for 'data' entry.
                    ex: [100, 100, 100, 100, 120, 120, 120, 120]
                :key fields (list(int)): integers corresponding to data type for 'data' entry
                    ex: [1, 2, 3, 4, 1, 2, 3, 4]
                    for each field (int) the corresponding data is:
                    - 1: Ambient temperature in degrees Celsius
                    - 2: Atmospheric pressure in in atmospheres.
                    - 3: Wind speed in meters per second (m/s)
                    - 4: Wind direction in degrees east of north (degrees).
                :key data (list(list(floats)): 8760 list with data of corresponding field and hub-height
                    ex. data[timestep] is [-23.5, 0.65, 7.6, 261.2, -23.7, 0.65, 7.58, 261.1]
                        - -23.5 is temperature at 100m at timestep
                        - 7.6 is wind speed at 100m at timestep
                        - 7.58 is wind speed at 120m at timestep
        """
        dic = {
            'heights': [float(h) for h in self.data_hub_heights for i in range(4)],
            'fields':  [1, 2, 3, 4] * len(self.data_hub_heights),
            'data':    combined_data
            }
        self._data = dic