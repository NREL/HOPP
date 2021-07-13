import csv
from collections import defaultdict
import numpy as np

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

    def __init__(self, lat, lon, year, wind_turbine_hub_ht, path_resource="", filepath="", **kwargs):
        """

        :param lat: float
        :param lon: float
        :param year: int
        :param wind_turbine_hub_ht: int
        :param path_resource: directory where to save downloaded files
        :param filepath: file path of resource file to load
        :param kwargs:
        """
        super().__init__(lat, lon, year)

        if os.path.isdir(path_resource):
            self.path_resource = path_resource

        self.path_resource = os.path.join(self.path_resource, 'wind')

        self.__dict__.update(kwargs)

        self.hub_height_meters = wind_turbine_hub_ht

        self.file_resource_heights = None

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

        file_resource_base = os.path.join(self.path_resource, str(self.latitude) + "_" + str(self.longitude) + "_windtoolkit_" + str(
            self.year) + "_" + str(self.interval) + "min")
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

            for height, f in self.file_resource_heights.items():
                url = 'https://developer.nrel.gov/api/wind-toolkit/wind/wtk_srw_download?year={year}&lat={lat}&lon={lon}&hubheight={hubheight}&api_key={api_key}'.format(
                    year=self.year, lat=self.latitude, lon=self.longitude, hubheight=height, api_key=get_developer_nrel_gov_key())

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

        with open(self.filename, 'w') as fo:
            writer = csv.writer(fo)
            writer.writerows(data)

        return os.path.isfile(self.filename)

    def format_data(self):
        """
        Format as 'wind_resource_data' dictionary for use in PySAM.
        """
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(self.filename + " does not exist. Try `download_resource` first.")
        wfd = defaultdict(defaultdict)
        with open(self.filename) as file_in:
            file_in.readline()
            file_in.readline()
            reader = csv.DictReader(file_in)
            line = 0
            for row in reader:
                if line == 1:
                    heights = list(row.values())
                    heights_keys = [str(i) for i in heights]
                    for key in set(heights_keys):
                        wfd[key] = defaultdict(list)
                if line > 1:
                    n = 0
                    for col, dat in row.items():
                        height_dict = wfd[heights_keys[n]]
                        height_dict[col].append(float(dat))
                        n += 1
                line += 1

            self.data = wfd

    @Resource.data.setter
    def data(self, data_dict):
        """
        Sets the wind resource data.

        All arrays must be same length, corresponding to number of data records.

        Dictionary contains measurements at a variable number of hub heights.
        Data for each hub height is provided in a dictionary with the height in meters as key.
        ('Temperature', 'Pressure', 'Speed', 'Direction') required for each hub height.

        i.e. data_dict = {'80' : { 'Temperature' : [...],
                                    'Pressure' ...
                                 }
                         }
        """

        n_records = []
        heights = data_dict.keys()
        n_heights = len(heights)
        field_names = ('Temperature', 'Pressure', 'Speed', 'Direction')
        for height, meas in data_dict.items():
            for key in field_names:
                if key not in meas.keys():
                    raise ValueError(key + " required for wind data at hub height " + height)
                n_records.append(len(meas[key]))

        n_records = set(n_records)
        if len(n_records) > 1:
            raise ValueError("All arrays must be same length, corresponding to number of data records.")
        n_records = n_records.pop()

        wind_data_matrix = np.zeros((n_records, 4 * n_heights))
        heights_id = []
        fields_id = []
        for height in heights:
            heights_id += [int(height)] * 4
            for col in range(4):
                wind_data_matrix[:, col] = data_dict[height][field_names[col]]
                fields_id.append(col + 1)

        # check units on pressure
        if np.max(wind_data_matrix[:, field_names.index("Pressure")]) > 1.1:
            wind_data_matrix[:, field_names.index("Pressure")] /= 101.325

        self._data = dict({'heights': heights_id, 'fields': fields_id, 'data': wind_data_matrix.tolist()})
