import os
from collections import defaultdict
from pathlib import Path
from typing import Union
import numpy as np
import csv
from PySAM.ResourceTools import SAM_CSV_to_solar_data

from hopp.utilities.keys import get_developer_nrel_gov_key, get_developer_nrel_gov_email
from hopp.utilities.log import hybrid_logger as logger
from hopp.simulation.technologies.resource.resource import Resource
from hopp import ROOT_DIR


BASE_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-2-2-download.csv"


class SolarResource(Resource):
    """
    Class to manage Solar Resource data.

    Args:
        lat: latitude
        lon: longitude
        year: year
        path_resource: directory where to save downloaded files
        filepath: file path of resource file to load
        use_api: Make an API call even if there's an existing file. Defaults to False
        kwargs: extra kwargs

    """

    def __init__(
        self, 
        lat: float, 
        lon: float, 
        year: int, 
        path_resource: Union[str, Path] = ROOT_DIR / "simulation" / "resource_files", 
        filepath: Union[str, Path] ="", 
        use_api: bool = False,
        **kwargs
    ):
        super().__init__(lat, lon, year)

        if os.path.isdir(path_resource):
            self.path_resource = path_resource

        self.solar_attributes = 'ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle,surface_pressure,dew_point'

        self.path_resource = os.path.join(self.path_resource, 'solar')

        # Force override any internal definitions if passed in
        self.__dict__.update(kwargs)

        # resource_files files
        if filepath == "":
            filepath = os.path.join(self.path_resource,
                                    str(lat) + "_" + str(lon) + "_psmv3_" + str(self.interval) + "_" + str(
                                        year) + ".csv")
        self.filename = filepath

        self.check_download_dir()   # FIXME: This breaks if weather file is in the same directory as caller

        if not os.path.isfile(self.filename) or use_api:
            self.download_resource()

        self.format_data()

        logger.info("SolarResource: {}".format(self.filename))

    def download_resource(self):
        url = '{base}?wkt=POINT({lon}+{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(
            base=BASE_URL, year=self.year, lat=self.latitude, lon=self.longitude, leap=self.leap_year, interval=self.interval,
            utc=self.utc, name=self.name, email=get_developer_nrel_gov_email(),
            mailing_list=self.mailing_list, affiliation=self.affiliation, reason=self.reason, api=get_developer_nrel_gov_key(),
            attr=self.solar_attributes)

        success = self.call_api(url, filename=self.filename)

        return success

    def format_data(self):
        """
        Format as 'solar_resource_data' dictionary for use in PySAM.
        """
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f"{self.filename} does not exist. Try `download_resource` first.")

        self.data = self.filename

    @Resource.data.setter
    def data(self, data_dict):
        """
        Sets the solar resource data

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
        :key tdew: array, dew point temp [C]
        :key press: array, atmospheric pressure [mbar]
        """
        self._data = SAM_CSV_to_solar_data(data_dict)
        # TODO: Update ResourceTools.py in pySAM to include pressure and dew point or relative humidity
        with open(data_dict) as file_in:
            wfd = defaultdict(list)
            for i in range(2):
                file_in.readline()
            reader = csv.DictReader(file_in)
            for row in reader:
                for col, dat in row.items():
                    if len(col) > 0:
                        wfd[col].append(float(dat))

        if 'Dew Point' in wfd:
            self._data['tdew'] = wfd.pop('Dew Point')
        elif 'RH' in wfd:
            self._data['rh'] = wfd.pop('RH')
        elif 'Pressure' in wfd:
            self._data['pres'] = wfd.pop('Pressure')


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
