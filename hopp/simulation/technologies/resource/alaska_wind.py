import csv, os
from pathlib import Path
from typing import Union, Optional, List
import pandas as pd
import urllib.parse

from attrs import define, field

from hopp.utilities.keys import get_developer_nrel_gov_key, get_developer_nrel_gov_email
from hopp.utilities.validators import range_val
from hopp.simulation.technologies.resource.resource import Resource
from hopp import ROOT_DIR
from hopp.tools.resource.pysam_wind_tools import combine_wind_files

AK_BASE_URL = "https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-alaska-v1-0-0-download.csv?"

@define
class AlaskaWindData(Resource):
    #: latitude corresponding to location for wind resource data
    lat: float = field()
    #: longitude corresponding to location for wind resource data
    lon: float = field()
    #: year for resource data. must be between 2018 and 2020
    year: int = field(validator=range_val(2018, 2020))

    #: the hub-height for wind resource data (meters)
    hub_height_meters: float = field(validator=range_val(10.0, 1000.0))
    
    #: filepath to resource_files directory. Defaults to ROOT_DIR/"simulation"/"resource_files".
    path_resource: Optional[Union[str, Path]] = field(default = ROOT_DIR / "simulation" / "resource_files")
    #: file path of resource file to load or download
    filename: Optional[Union[str, Path]] = field(default = None)
    #: Make an API call even if there's an existing file. Defaults to False.
    use_api: Optional[bool] = field(default = False)
    #: dictionary of preloaded and formatted wind resource data. Defaults to None.
    resource_data: Optional[dict] = field(default = None)

    #: dictionary of heights and filenames to download from Wind Toolkit
    file_resource_heights: dict = field(default = None)

    #: list of heights that wind resource data is available for downloading (meters)
    allowed_hub_height_meters: list[int] = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 500, 1000]
    

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

        if self.filename is None:
            self.calculate_heights_to_download()

        self.check_download_dir()

        if not os.path.isfile(self.filename) or self.use_api:
            self.download_resource()
        
        self.format_data()
        
    def calculate_heights_to_download(self):
        """
        Given the system hub height, and the available hubheights from WindToolkit,
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

        filename_base = f"{self.latitude}_{self.longitude}_WTK_Alaksa_{self.year}_{self.interval}min"
        file_resource_full = filename_base
        file_resource_heights = dict()

        for h in heights:
            h_int = int(h)
            file_resource_heights[h_int] = self.path_resource/(filename_base + f'_{h_int}m.csv')
            file_resource_full += f'_{h_int}m'
        file_resource_full += ".csv"

        self.file_resource_heights = file_resource_heights
        self.filename = self.path_resource / file_resource_full

    def update_height(self, hub_height_meters):
        """Update hub-height and corresponding attributes. 
        Also updates ``file_resource_heights`` and ``filename``.

        Args:
            hub_height_meters (float): hub-height for wind resource data (meters)
        """
        self.hub_height_meters = hub_height_meters
        self.calculate_heights_to_download()

    def download_resource(self):
        success = False

        base_attributs = ["temperature","windspeed","winddirection"]
        attributes = ["pressure_100m"]
        for height, f in self.file_resource_heights.items():
            attributes += [f"{a}_{height}m" for a in base_attributs]
        
        attributes_str = ",".join(k for k in attributes)
        input_data = {
            'attributes': attributes_str,
            'interval': self.interval,
            'api_key': get_developer_nrel_gov_key(),
            'email': get_developer_nrel_gov_email(),
            'names': [str(self.year)],
            'wkt': f"POINT({self.longitude} {self.latitude})"
        }
        url = AK_BASE_URL + urllib.parse.urlencode(input_data, True)
        success = self.call_api(url, filename=self.filename)
       
        if not success:
            raise ValueError('Unable to download wind data')

        return success

    def format_data(self):
        """
        Format as 'wind_resource_data' dictionary for use in PySAM.
        """
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f"{self.filename} does not exist. Try `download_resource` first.")

        self.data = self.filename

    @Resource.data.setter
    def data(self, data_info):
        """
        Sets the wind resource data to a dictionary in SAM Wind format.
        """
        if isinstance(data_info,dict):
            self._data = data_info
        if isinstance(data_info,(str, Path)):
            resource_heights = [k for k in self.file_resource_heights.keys()]
            self._data = combine_wind_files(str(data_info),resource_heights)
