import csv
from collections import defaultdict
import numpy as np
from PySAM.ResourceTools import SRW_to_wind_data

from hybrid.keys import get_developer_nrel_gov_key
from hybrid.log import hybrid_logger as logger
from hybrid.resource.resource import *


class WindResource(Resource):
    """ Class to manage Wind Resource data

    Attributes:
        hub_height_meters - the system height
            TODO: if optimizer will modify hub height, need to download a range rather than a single
        file_resource_heights - dictionary of heights and filenames to download from Wind Toolkit
        filename - the combined resource filename
    """

    allowed_hub_height_meters = [10, 40, 60, 80, 100, 120, 140, 160, 200]

    def __init__(self, lat, lon, year, wind_turbine_hub_ht, api='nrel', nasa_vegtype='vegtype_8', path_resource="", filepath="", **kwargs):
        """

        :param lat: float
        :param lon: float
        :param year: int
        :param api: string ('nrel' or 'nasa')
        :param nasa_vegtype: string (see wind-surface options at https://power.larc.nasa.gov/docs/methodology/meteorology/wind/#corrected-wind-speed)
        :param wind_turbine_hub_ht: int
        :param path_resource: directory where to save downloaded files
        :param filepath: file path of resource file to load
        :param kwargs:
        """
        super().__init__(lat, lon, year, api)

        if os.path.isdir(path_resource):
            self.path_resource = path_resource

        self.path_resource = os.path.join(self.path_resource, 'wind')

        self.__dict__.update(kwargs)

        self.hub_height_meters = wind_turbine_hub_ht

        self.file_resource_heights = None

        self.vegtype = nasa_vegtype

        if filepath == "":
            self.filename = ""
            self.calculate_heights_to_download()
        else:
            self.filename = filepath

        self.check_download_dir()

        if not os.path.isfile(self.filename):
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

        if self.api == 'nrel':
            file_resource_base = os.path.join(self.path_resource, str(self.latitude) + "_" + str(self.longitude) + "_windtoolkit_" + str(
                self.year) + "_" + str(self.interval) + "min")
        elif self.api == 'nasa':
            file_resource_base = os.path.join(self.path_resource, str(self.latitude) + "_" + str(self.longitude) + "_nasa_" + str(
                self.year) + "_" + str(self.interval) + "min")
        else:
            raise NameError(self.api + " does not exist. Try 'nrel' for the NREL developer network WindToolkit API or 'nasa' for NASA POWER API")
        file_resource_full = file_resource_base
        file_resource_heights = dict()

        for h in heights:
            file_resource_heights[h] = file_resource_base + '_' + str(h) + 'm.srw'
            file_resource_full += "_" + str(h) + 'm'
        file_resource_full += ".srw"

        self.file_resource_heights = file_resource_heights
        self.filename = file_resource_full

    def update_height(self, hub_height_meters):
        self.hub_height_meters = hub_height_meters
        self.calculate_heights_to_download()

    def download_resource(self):
        success = os.path.isfile(self.filename)
        if not success:
            if self.api.lower() == 'nrel':
                for height, f in self.file_resource_heights.items():
                    url = 'https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-srw-download?year={year}&lat={lat}&lon={lon}&hubheight={hubheight}&api_key={api_key}&email={email}'.format(
                        year=self.year, lat=self.latitude, lon=self.longitude, hubheight=height, api_key=get_developer_nrel_gov_key(), email=self.email)

                    success = self.call_api(url, filename=f)

            elif self.api.lower() == 'nasa':
                for height, f in self.file_resource_heights.items():
                    url = 'https://power.larc.nasa.gov/api/temporal/hourly/point?start={start}&end={end}&latitude={lat}&longitude={lon}&community=RE&parameters=T2M&format=srw&wind-surface={surface}&wind-elevation={hubheight}&site-elevation={hubheight}'.format(
                        start=self.start_date, end=self.end_date, lat=self.latitude, lon=self.longitude, surface=self.vegtype, hubheight=height)

                    success = self.call_api(url, filename=f)
            else:
                raise NameError(self.api + " does not exist. Try 'nrel' for the NREL developer network WindToolkit API or 'nasa' for NASA POWER API")

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
        data = [None] * 2
        for height, f in self.file_resource_heights.items():
            if os.path.isfile(f):
                with open(f) as file_in:
                    csv_reader = csv.reader(file_in, delimiter=',')
                    line = 0
                    for row in csv_reader:
                        if line < 2:
                            data[line] = row
                        else:
                            if line >= len(data):
                                data.append(row)
                            else:
                                data[line] += row
                        line += 1

        with open(self.filename, 'w', newline='') as fo:
            writer = csv.writer(fo)
            writer.writerows(data)

        return os.path.isfile(self.filename)

    def format_data(self):
        """
        Format as 'wind_resource_data' dictionary for use in PySAM.
        """
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(self.filename + " does not exist. Try `download_resource` first.")

        self.data = self.filename

    @staticmethod
    def adjust_nasa_data_to_hub_height(data_dict):
        # removes 2/29 data for leap years. PySAM Wind model can handle leap years, other modules like solar cannot
        # If in the future all component modules can handle leap years, then this logic can be removed
        if len(data_dict['data']) == 8784:
            del data_dict['data'][1416:1440]

        # Simplifications to allow use of NASA POWER Wind data when data points are > 10 m away from hub height of interest
        # This methodology assumes the NASA POWER API is called with all arguments (wind-surface, wind-elevation, site-elevation) as detailed in example below
        # https://power.larc.nasa.gov/api/temporal/hourly/point?start=20120101&end=20121231&latitude=35.2018&longitude=-101.9450&community=RE&parameters=T2M&format=srw&wind-surface=vegtype_4&wind-elevation=80&site-elevation=80
        # Can be removed in future if NASA POWER can provide all data points at specified hub height
        nasa_data_dict = {}
        full_data = np.array(data_dict['data'])
        num_datapoints = full_data.shape[0]
        nasa_data_dict['data'] = np.zeros((num_datapoints, 4))
        nasa_data_dict['data'][:, 0] = full_data[:, 0] # Grab the temp at 2 m
        nasa_data_dict['data'][:, 1] = full_data[:, 9] # Grab the corrected pressure at user specified hub height
        nasa_data_dict['data'][:, 2] = full_data[:, 8] # Grab the corrected speed at user specified hub height
        nasa_data_dict['data'][:, 3] = full_data[:, 7] # Grab the direction at 50 m
        nasa_data_dict['heights'] = [data_dict['heights'][-1]] * 4
        nasa_data_dict['fields'] = [1, 2, 3, 4]
        if data_dict['heights'][-1] == 10: 
            nasa_data_dict['data'][:, 3] = full_data[:, 5] # overwrites 50 m direction data with 10 m direction data
        nasa_data_dict['data'] = nasa_data_dict['data'].tolist()
        return nasa_data_dict

    @Resource.data.setter
    def data(self, data_file):
        """
        Sets the wind resource data to a dictionary in SAM Wind format (see Pysam.ResourceTools.SRW_to_wind_data)
        """
        
        self._data = SRW_to_wind_data(data_file)
        if self.api == "nasa":
            self._data = self.adjust_nasa_data_to_hub_height(self._data)