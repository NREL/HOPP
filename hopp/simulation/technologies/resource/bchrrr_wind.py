import os
from pathlib import Path
from typing import Union, Optional, List
import urllib.parse
from rex import WindX
from rex.sam_resource import SAMResource
import numpy as np 

from attrs import define, field

from hopp.utilities.keys import get_developer_nrel_gov_key, get_developer_nrel_gov_email
from hopp.utilities.validators import range_val
from hopp.simulation.technologies.resource.resource import Resource
from hopp import ROOT_DIR
from hopp.tools.resource.pysam_wind_tools import combine_wind_files

BCHRRR_BASE_URL = "https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-bchrrr-v1-0-0-download.csv?"
BCHRRR_HPC_BASE = "/datasets/WIND/HRRR/bchrrr_conus_"

@define
class BCHRRRWindData(Resource):    
    """ Class to manage Wind Resource data from BC-HRRR dataset using API calls or preloaded data.

        Args:
            lat (float): latitude corresponding to location for wind resource data
            lon (float): longitude corresponding to location for wind resource data
            year (int): year for resource data. must be between 2007 and 2014
            hub_height_meters (float): turbine hub height (m)
            path_resource (Union[str, Path], optional): filepath to resource_files directory. Defaults to ROOT_DIR/"simulation"/"resource_files".
            filepath (Union[str, Path], optional): file path of resource file to load
            use_api (bool, optional): Make an API call even if there's an existing file. Defaults to False.
            resource_data (Optional[dict], optional): dictionary of preloaded and formatted wind resource data. Defaults to None.
            use_hpc (bool, Optional): use hpc bchrrr data or pull API data. Defaults to API
           kwargs: extra kwargs
        """
    
    lat: float = field()
    lon: float = field()
    #: year for resource data. Must be between 2015 and 2023
    year: int = field(validator=range_val(2015, 2023))

    #: the hub-height for wind resource data (meters)
    hub_height_meters: float = field(validator=range_val(10.0, 200.0))

    
    # OPTIONAL INPUTS
    path_resource: Optional[Union[str, Path]] = field(default = ROOT_DIR / "simulation" / "resource_files")
    filename: Optional[Union[str, Path]] = field(default = None)
    use_api: Optional[bool] = field(default = False)
    resource_data: Optional[dict] = field(default = None)
    use_hpc: Optional[bool] = field(default = False)

    #: dictionary of heights and filenames to download from Wind Toolkit
    file_resource_heights: dict = field(default = None)

    # NOT INPUTS
    allowed_hub_height_meters: List[int] = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    

    def __attrs_post_init__(self):
        super().__init__(self.lat, self.lon, self.year)   

        # if resource_data is input as a dictionary then set_data   
        if isinstance(self.resource_data,dict):
            self.data = self.resource_data
            return 
        # if resource_data is not provided, download or load resource data
        if isinstance(self.path_resource,str):
            self.path_resource = Path(self.path_resource).resolve()
        if self.path_resource.parts[-1]!="wind":
            self.path_resource = self.path_resource / 'wind'
        
        if self.use_hpc:
            # Remove kestrel prefix for path
            self.path_resource = '/' / Path(*self.path_resource.parts[2:])
            self.data_hub_heights = self.calculate_heights_to_download()

            self.hpc_resource()
            # Pull data from HPC Wind Toolkit dataset
            self.download_resource_hpc()

            # Set wind resource data into SAM/PySAM digestible format
            self.format_data_hpc()    
        else:

            if self.filename is None:
                self.calculate_heights_to_download()

            self.check_download_dir()

            if not os.path.isfile(self.filename) or self.use_api:
                self.download_resource()
            
            self.format_data()
        
    def calculate_heights_to_download(self):
        """
        Given the system hub height, and the available hubheights from BC-HRRR Data,
        determine which heights to download to bracket the hub height
        """
        hub_height_meters = self.hub_height_meters

        # evaluate hub height, determine what heights to download
        heights = [hub_height_meters]
        if hub_height_meters not in self.allowed_hub_height_meters:
            height_low = self.allowed_hub_height_meters[0]
            height_high = self.allowed_hub_height_meters[-1]
            for h in self.allowed_hub_height_meters:
                if h < hub_height_meters:
                    height_low = h
                elif h > hub_height_meters:
                    height_high = h
                    break
            heights[0] = height_low
            heights.append(height_high)
        
        filename_base = f"{self.latitude}_{self.longitude}_BC_HRRR_{self.year}_{self.interval}min"
        file_resource_full = filename_base
        file_resource_heights = dict()

        for h in heights:
            h_int = int(h)
            file_resource_heights[h_int] = self.path_resource/(filename_base + f'_{h_int}m.csv')
            file_resource_full += f'_{h_int}m'
        file_resource_full += ".csv"

        self.file_resource_heights = file_resource_heights
        self.filename = self.path_resource / file_resource_full

        if self.use_hpc:
            return heights

    def update_height(self, hub_height_meters):
        self.hub_height_meters = hub_height_meters
        self.calculate_heights_to_download()

    def download_resource(self):
        """
        Downloads the wind data from the BC-HRRR dataset using an API call
        """
        success = False

        base_attributes = ["temperature","windspeed","winddirection"]
        attributes = ["pressure_0m", "precipitationrate_0m"]
        for height, f in self.file_resource_heights.items():
            attributes += [f"{a}_{height}m" for a in base_attributes]
        
        attributes_str = ",".join(k for k in attributes)
        input_data = {
            'attributes': attributes_str,
            'interval': self.interval,
            'api_key': get_developer_nrel_gov_key(),
            'email': get_developer_nrel_gov_email(),
            'names': [str(self.year)],
            'wkt': f"POINT({self.longitude} {self.latitude})"
        }
        url = BCHRRR_BASE_URL + urllib.parse.urlencode(input_data, True)
        success = self.call_api(url, filename=self.filename)
       
        if not success:
            raise ValueError('Unable to download wind data')

        return success
    
    def hpc_resource(self):
        """
        Downloads the wind data from the BC-HRRR dataset hosted on the HPC
        """
        self.bchrrr_file = BCHRRR_HPC_BASE + f"{self.year}.h5"

        # Check for valid filepath for Wind Toolkit file
        if not os.path.isfile(self.bchrrr_file):
            raise FileNotFoundError(f"Cannot find Wind Toolkit .h5 file, filepath {self.bchrrr_file} does not exist")
        
     

    def download_resource_hpc(self):
        """load BCHRRR h5 file using rex and get wind resource data for location
        specified by (self.lat, self.lon)
        """
        # NOTE: Current setup of files on HPC WINDToolkit v1.0.0 = 2007-2013, v1.1.0 = 2014
    
        # Open file with rex WindX object
        with WindX(self.bchrrr_file, hsds=False) as f:
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

    def format_data_hpc(self):
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
    

    def format_data(self):
        """
        Format as 'wind_resource_data' dictionary for use in PySAM.
        """
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f"{self.filename} does not exist. Try `download_resource` first.")

        self.data = self.filename

    @Resource.data.setter
    def data(self, data_info):

        if self.use_hpc:
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
                'data':    data_info
                }
            self._data = dic  
        else:       
            """
            Sets the wind resource data to a dictionary in SAM Wind format (see Pysam.ResourceTools.SRW_to_wind_data)
            """   
            if isinstance(data_info,dict):
                self._data = data_info
            if isinstance(data_info,(str, Path)):
                resource_heights = [k for k in self.file_resource_heights.keys()]
                self._data = combine_wind_files(str(data_info),resource_heights)
