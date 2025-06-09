import csv, os
from pathlib import Path
from typing import Union, Optional, List
from PySAM.ResourceTools import SRW_to_wind_data

from hopp.utilities.keys import get_developer_nrel_gov_key, get_developer_nrel_gov_email
from hopp.simulation.technologies.resource.resource import Resource
from hopp.tools.resource.pysam_wind_tools import combine_and_write_srw_files
from hopp import ROOT_DIR


WTK_BASE_URL = "https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-srw-download"
TAP_BASE_URL = "https://dw-tap.nrel.gov/v2/srw"


class WindResource(Resource):
    """ Class to manage Wind Resource data from API calls or preloaded data.
    """

    allowed_hub_height_meters: List[int] = [10, 40, 60, 80, 100, 120, 140, 160, 200]
    
    #: the hub-height for wind resource data (meters)
    hub_height_meters: float
    # TODO: if optimizer will modify hub height, need to download a range rather than a single
    
    #: dictionary of heights and filenames to download from Wind Toolkit
    file_resource_heights: dict


    def __init__(
        self, 
        lat: float, 
        lon: float, 
        year: int, 
        wind_turbine_hub_ht: float, 
        path_resource: Union[str, Path] = ROOT_DIR / "simulation" / "resource_files", 
        filepath: Union[str, Path] ="", 
        source: str ="WTK", 
        use_api: bool = False,
        resource_data: Optional[dict] = None,
        **kwargs
    ):
        """Resource class to download wind resource data using API call or set with preloaded dictionary

        Args:
            lat (float): latitude corresponding to location for wind resource data
            lon (float): longitude corresponding to location for wind resource data
            year (int): year for resource data. must be between 2007 and 2014
            wind_turbine_hub_ht (float): turbine hub height (m)
            path_resource (Union[str, Path], optional): filepath to resource_files directory. Defaults to ROOT_DIR/"simulation"/"resource_files".
            filepath (Union[str, Path], optional): file path of resource file to load
            source (str): Which API to use. Options are TAP and WIND Toolkit (WTK).
            use_api (bool, optional): Make an API call even if there's an existing file. Defaults to False.
            resource_data (Optional[dict], optional): dictionary of preloaded and formatted wind resource data. Defaults to None.
            kwargs: extra kwargs
        """
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

        self.source = source

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

        filename_base = f"{self.latitude}_{self.longitude}_windtoolkit_{self.year}_{self.interval}min"
        file_resource_full = filename_base
        file_resource_heights = dict()

        for h in heights:
            h_int = int(h)
            file_resource_heights[h_int] = self.path_resource/(filename_base + f'_{h_int}m.srw')
            file_resource_full += f'_{h_int}m'
        file_resource_full += ".srw"

        self.file_resource_heights = file_resource_heights
        self.filename = self.path_resource / file_resource_full

    def update_height(self, hub_height_meters):
        self.hub_height_meters = hub_height_meters
        self.calculate_heights_to_download()

    def download_resource(self):
        success = False

        for height, f in self.file_resource_heights.items():
            url = ""

            if self.source == "WTK":
                url = '{base}?year={year}&lat={lat}&lon={lon}&hubheight={hubheight}&api_key={api_key}&email={email}'.format(
                    base=WTK_BASE_URL, year=self.year, lat=self.latitude, lon=self.longitude, hubheight=height, api_key=get_developer_nrel_gov_key(), email=get_developer_nrel_gov_email()
                )
            elif self.source == "TAP":
                url = '{base}?height={hubheight}m&lat={lat}&lon={lon}&year={year}'.format(
                    base=TAP_BASE_URL, year=self.year, lat=self.latitude, lon=self.longitude, hubheight=height
                )

            success = self.call_api(url, filename=f)

        if not success:
            raise ValueError('Unable to download wind data')

        # combine into one file to pass to SAM
        if len(list(self.file_resource_heights.keys())) > 1:
            success = self.combine_wind_files()

            if not success:
                raise ValueError('Could not combine wind resource files successfully')
        return success

    def combine_wind_files(self):
        """
        Parameters
        ---------
        file_resource_heights: dict
            Keys are height in meters, values are corresponding files
            example {40: path_to_file, 60: path_to_file2}
        file_out: string
            File path to write combined srw file
        """
        
        success = combine_and_write_srw_files(self.file_resource_heights,self.filename)
        
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
        else:
            self._data = SRW_to_wind_data(data_info)
