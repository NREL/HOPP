import csv, os
from pathlib import Path
from typing import Union, Optional, List
import pandas as pd
import urllib.parse

# from PySAM.ResourceTools import CSV_to_wind_data

from hopp.utilities.keys import get_developer_nrel_gov_key, get_developer_nrel_gov_email
from hopp.utilities.validators import gt_zero, contains, range_val
from hopp.simulation.technologies.resource.resource import Resource
from hopp import ROOT_DIR
from hopp.tools.resource.pysam_wind_tools import combine_wind_files
#CSV_to_wind_data, combine_CSV_to_wind_data

AK_BASE_URL = "https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-alaska-v1-0-0-download.csv?"
# AK_LED_BASE_URL = "https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-led-alaska-download.csv?"


class AlaskaWindData(Resource):
    allowed_hub_height_meters: List[int] = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 500, 1000]
    all_attributes: str = ('boundary_layer_height,friction_velocity_2m,inversemoninobukhovlength_2m,latent_heat_flux,precipitation_0m,'
    'pressure_0m,pressure_100m,pressure_200m,pressure_500m,relativehumidity_2m,sensible_heat_flux,skin_temperature,'
    'temperature_1000m,temperature_100m,temperature_200m,temperature_20m,temperature_2m,temperature_300m,temperature_40m,'
    'temperature_500m,temperature_60m,temperature_80m,vertical_windspeed_120m,vertical_windspeed_200m,vertical_windspeed_20m,'
    'vertical_windspeed_40m,vertical_windspeed_500m,vertical_windspeed_80m,virtual_potential_temperature_1000m,'
    'virtual_potential_temperature_100m,virtual_potential_temperature_200m,virtual_potential_temperature_20m,'
    'virtual_potential_temperature_2m,virtual_potential_temperature_300m,virtual_potential_temperature_40m,'
    'virtual_potential_temperature_500m,virtual_potential_temperature_60m,virtual_potential_temperature_80m,'
    'winddirection_1000m,winddirection_100m,winddirection_10m,winddirection_120m,winddirection_140m,winddirection_160m,'
    'winddirection_180m,winddirection_200m,winddirection_20m,winddirection_250m,winddirection_300m,winddirection_40m,winddirection_500m,'
    'winddirection_60m,winddirection_80m,windspeed_1000m,windspeed_100m,windspeed_10m,windspeed_120m,windspeed_140m,windspeed_160m,windspeed_180m,'
    'windspeed_200m,windspeed_20m,windspeed_250m,windspeed_300m,windspeed_40m,windspeed_500m,windspeed_60m,windspeed_80m')
    
    #: the hub-height for wind resource data (meters)
    hub_height_meters: float
    # TODO: if optimizer will modify hub height, need to download a range rather than a single
    
    #: dictionary of heights and filenames to download from Wind Toolkit
    file_resource_heights: dict
    
    def __init__(
        self, 
        lat: float, 
        lon: float, 
        year: int, #must be between 2018 and 2020
        wind_turbine_hub_ht: float, 
        path_resource: Union[str, Path] = ROOT_DIR / "simulation" / "resource_files", 
        filepath: Union[str, Path] ="", 
        use_api: bool = False,
        resource_data: Optional[dict] = None,
    ):
        super().__init__(lat, lon, year)   

        # if resource_data is input as a dictionary then set_data   
        if isinstance(resource_data,dict):
            self.data = resource_data
            self.hub_height_meters = wind_turbine_hub_ht
            return 
        
        # if resource_data is not provided, download or load resource data
        if isinstance(path_resource,str):
            path_resource = Path(path_resource).resolve()
        if os.path.isdir(path_resource):
            self.path_resource = path_resource
        if path_resource.parts[-1]!="wind":
            self.path_resource = self.path_resource / 'wind'

        self.file_resource_heights = None
        self.update_height(wind_turbine_hub_ht)

        if filepath == "":
            self.filename = ""
            self.calculate_heights_to_download()
        else:
            self.filename = filepath

        self.check_download_dir()

        if not os.path.isfile(self.filename) or use_api:
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
            file_resource_heights[h_int] = self.path_resource/(filename_base + f'_{h_int}m.srw')
            file_resource_full += f'_{h_int}m'
        file_resource_full += ".csv"

        self.file_resource_heights = file_resource_heights
        self.filename = self.path_resource / file_resource_full

    def update_height(self, hub_height_meters):
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
        Sets the wind resource data to a dictionary in SAM Wind format (see Pysam.ResourceTools.SRW_to_wind_data)
        """
        if isinstance(data_info,dict):
            self._data = data_info
        if isinstance(data_info,(str, Path)):
            resource_heights = [k for k in self.file_resource_heights.keys()]
            self._data = combine_wind_files(str(data_info),resource_heights)
