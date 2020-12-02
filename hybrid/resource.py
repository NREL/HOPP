from abc import ABCMeta, abstractmethod
import csv
import os
import json
import requests
import time
from collections import defaultdict
import numpy as np

from hybrid.keys import get_developer_nrel_gov_key
from hybrid.log import hybrid_logger as logger


class Resource(metaclass=ABCMeta):
    """
    Class to manage resource data for a given lat & lon. If a resource file doesn't exist,
    it is downloaded and saved to 'resource_files' folder. The resource file is then read
    to the appropriate SAM resource data format.
    """
    def __init__(self, lat, lon, year, **kwargs):
        """
        Parameters
        ---------
        lat: float
            The latitude
        lon: float
            The longitude
        year: int
            The year of resource_files data
        """

        self.latitude = lat
        self.longitude = lon
        self.year = year

        self.n_timesteps = 8760

        # generic api settings
        self.interval = str(int(8760/365/24 * 60))
        self.leap_year = 'false'
        self.utc = 'false'
        self.name = 'hybrid-systems'
        self.affiliation = 'NREL'
        self.reason = 'hybrid-analysis'
        self.email = 'nicholas.diorio@nrel.gov'
        self.mailing_list = 'true'

        # paths
        self.path_current = os.path.dirname(os.path.abspath(__file__))
        self.path_resource = os.path.join(self.path_current, '..', 'resource_files')

        # update any passed in
        self.__dict__.update(kwargs)

        self.filename = None
        self._data = dict()

    def check_download_dir(self):
        if not os.path.isdir(os.path.dirname(self.filename)):
            os.makedirs(os.path.dirname(self.filename))

    @staticmethod
    def call_api(url, filename):
        """
        Parameters
        ---------
        url: string
            The API endpoint to return data from
        filename: string
            The filename where data should be written
        """

        n_tries = 0
        success = False
        while n_tries < 5:

            try:
                r = requests.get(url)
                if r:
                    localfile = open(filename, mode='w+')
                    localfile.write(r.text)
                    localfile.close()
                    if os.path.isfile(filename):
                        success = True
                        break
                elif r.status_code == 400 or r.status_code == 403:
                    print(r.url)
                    err = r.text
                    text_json = json.loads(r.text)
                    if 'errors' in text_json.keys():
                        err = text_json['errors']
                    raise requests.exceptions.HTTPError(err)
                elif r.status_code == 404:
                    raise requests.exceptions.HTTPError
            except requests.exceptions.Timeout:
                time.sleep(0.2)
                n_tries += 1

        return success

    @abstractmethod
    def download_resource(self):
        """Download resource for given lat/lon"""

    @abstractmethod
    def format_data(self):
        """Reads data from file and formats it for use in SAM"""

    @property
    def data(self):
        """Get data as dictionary formatted for SAM"""
        return self._data

    @data.setter
    @abstractmethod
    def data(self, data_dict):
        """Sets data from dictionary"""


class SolarResource(Resource):
    """
        Class to manage Solar Resource data
        """

    def __init__(self, lat, lon, year, path_resource="", filepath="", **kwargs):
        """

        :param lat: float
        :param lon: float
        :param year: int
        :param path_resource: directory where to save downloaded files
        :param filepath: file path of resource file to load
        :param kwargs:
        """
        super().__init__(lat, lon, year)

        if os.path.isdir(path_resource):
            self.path_resource = path_resource

        self.solar_attributes = 'ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle'

        self.path_resource = os.path.join(self.path_resource, 'solar')

        # Force override any internal definitions if passed in
        self.__dict__.update(kwargs)

        # resource_files files
        if filepath == "":
            filepath = os.path.join(self.path_resource,
                                    str(lat) + "_" + str(lon) + "_psmv3_" + str(self.interval) + "_" + str(
                                        year) + ".csv")
        self.filename = filepath

        self.check_download_dir()

        if not os.path.isfile(self.filename):
            self.download_resource()

        self.format_data()

        logger.info("SolarResource: {}".format(self.filename))

    def download_resource(self):
        url = 'https://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?wkt=POINT({lon}+{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(
            year=self.year, lat=self.latitude, lon=self.longitude, leap=self.leap_year, interval=self.interval,
            utc=self.utc, name=self.name, email=self.email,
            mailing_list=self.mailing_list, affiliation=self.affiliation, reason=self.reason, api=get_developer_nrel_gov_key(),
            attr=self.solar_attributes)

        success = self.call_api(url, filename=self.filename)

        return success

    def format_data(self):
        """
        Format as 'solar_resource_data' dictionary for use in PySAM.
        """
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(self.filename + " does not exist. Try `download_resource` first.")
        wfd = defaultdict(list)
        with open(self.filename) as file_in:
            info = []
            for i in range(2):
                info.append(file_in.readline())
                info[i] = info[i].split(",")
            if "Time Zone" not in info[0]:
                raise ValueError("`Time Zone` field not found in solar resource file.")
            tz = info[1][info[0].index("Time Zone")]
            elev = info[1][info[0].index("Elevation")]
            reader = csv.DictReader(file_in)
            for row in reader:
                for col, dat in row.items():
                    if len(col) < 1:
                        continue
                    wfd[col].append(float(dat))

            weather = dict()
            weather['tz'] = float(tz)
            weather['elev'] = float(elev)
            weather['lat'] = self.latitude
            weather['lon'] = self.longitude
            weather['year'] = wfd.pop('Year')
            weather['month'] = wfd.pop('Month')
            weather['day'] = wfd.pop('Day')
            weather['hour'] = wfd.pop('Hour')
            weather['minute'] = wfd.pop('Minute')
            weather['dn'] = wfd.pop('DNI')
            weather['df'] = wfd.pop('DHI')
            weather['gh'] = wfd.pop('GHI')
            weather['wspd'] = wfd.pop('Wind Speed')
            weather['tdry'] = wfd.pop('Temperature')

            self.data = weather

    @Resource.data.setter
    def data(self, data_dict):
        """
        Sets the solar resource data.

        All arrays must be same length, corresponding to number of data records.

        For hourly resource, year, month, day, hour, and minute will be auto-filled if not provided.

        :key tz: time zone, not UTC
        :key elev: elevation in meters
        :key year: array
        :key month: array
        :key day: array
        :key hour: array
        :key minute: array
        :key dn: array, direct normal irradiance
        :key df: array, direct horizontal irradiance
        :key wspd: array, wind speed [m/s]
        :key tdry: array, dry bulb temp [C]
        """
        if "tz" not in data_dict:
            raise ValueError("Time zone required as `tz`")
        if "elev" not in data_dict:
            raise ValueError("Elevation required as `elev`")
        n_records = len(data_dict['dn'])
        check_vals = ('df', 'wspd', 'tdry')
        if n_records != 8760:
            check_vals += ('year', 'month', 'day', 'hour', 'minute')
        for val in check_vals:
            if len(data_dict[val]) != n_records:
                raise ValueError("All arrays must be same length, corresponding to number of data records.")
        self._data = data_dict

    def roll_timezone(self, roll_hours, timezone):
        """

        :param roll_hours:
        :param timezone:
        :return:
        """
        rollable_keys = ['dn', 'df', 'gh', 'wspd', 'tdry']
        for key in rollable_keys:
            if any(k == key for k in rollable_keys):
                roll_range = range(0, -roll_hours + 1)

                weather_array = np.array(self._data[key])

                weather_array_rolled = np.delete(weather_array, roll_range)
                weather_array_rolled = np.pad(weather_array_rolled, (0, -roll_hours + 1), 'constant')

                self._data[key] = weather_array_rolled.tolist()

        self._data['tz'] = timezone
        logger.info('Rolled solar data by {} hours for timezone {}'.format(roll_hours, timezone))


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

