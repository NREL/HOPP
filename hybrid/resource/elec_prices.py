import csv
from collections import defaultdict
import numpy as np

from hybrid.keys import get_developer_nrel_gov_key
from hybrid.log import hybrid_logger as logger
from hybrid.resource.resource import *


class ElectricityPrices(Resource):
    """

    """
    def __init__(self, lat, lon, year, path_resource="", filepath=""):
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

        self.path_resource = os.path.join(self.path_resource, 'grid')

        if filepath == "":
            filepath = "../../resource_files/grid/dispatch_factors_ts.csv"  # 'default' value
        self.filename = filepath

        if len(str(self.filename)) > 0 and not os.path.isfile(self.filename):
            raise ValueError

        self.format_data()

    def download_resource(self):
        raise NotImplementedError

    def format_data(self):
        if not os.path.isfile(self.filename):
            return
        # TODO: figure out a consistent naming convention
        data = []
        with open(self.filename) as file_in:
            csv_reader = csv.reader(file_in)
            for n, row in enumerate(csv_reader):
                if n == 0:
                    # skip header
                    try:
                        data.append(float(row[0]))
                    except ValueError:
                        pass
                else:
                    data.append(float(row[0]))
        self._data = data

    def data(self):
        if not os.path.isfile(self.filename):
            raise NotImplementedError("File not available as downloading not implemented yet")
        return self._data

    @Resource.data.setter
    def data(self, data_dict):
        pass
