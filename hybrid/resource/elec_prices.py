import csv
from pathlib import Path
import pandas as pd
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
            home_dir = Path(__file__).parent.parent.parent.absolute()
            filepath = os.path.join(str(home_dir), "resource_files", "grid", "dispatch_factors_ts.csv")  # 'default' value
        self.filename = filepath

        if len(str(self.filename)) > 0:
            self.format_data()

    def download_resource(self):
        raise NotImplementedError

    def format_data(self):
        if not os.path.isfile(self.filename):
            raise IOError(f"ElectricityPrices error: {self.filename} does not exist.")
        self._data = np.loadtxt(self.filename)
        self.n_timesteps = len(self._data)

    @property
    def data(self):
        if not os.path.isfile(self.filename):
            raise NotImplementedError("File not available as downloading not implemented yet")
        return self._data

    def resample_data(self, frequency_mins: int):
        """
        Resample the prices given the new frequency in minutes
        """
        n_recs = len(self._data)
        if not n_recs:
            return
        cur_freq = 8760/len(self._data)
        self._data = np.append(self._data, self._data[0])
        n_recs += 1
        start_date = pd.Timestamp(f'2013-01-01 00:00:00')   # choose non-leap-year
        ix = pd.date_range(start=start_date, 
                    end=start_date
                    + pd.offsets.DateOffset(hours=8761),
                    freq=f'{cur_freq}H')[0:n_recs]
        df = pd.DataFrame(self._data, index=ix)
        df = df.resample(frequency_mins).mean().interpolate(method='linear').head(-1)
        self._data = df.to_numpy().ravel().tolist()
        self.n_timesteps = len(self._data)
