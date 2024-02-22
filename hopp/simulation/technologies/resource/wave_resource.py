import pandas as pd
import PySAM.WaveFileReader as wavefile

from hopp.utilities.log import hybrid_logger as logger
from hopp.simulation.technologies.resource.resource import *


class WaveResource(Resource):
    """
    Class to manage Wave Resource data
    """
    def __init__(
        self, 
        lat: float, 
        lon: float, 
        year: int, 
        path_resource: str = "", 
        filepath: str = "", 
        **kwargs
    ):
        """
        lat (float): latitude
        lon (float): longitude
        year (int): year
        path_resource (str): directory where to save downloaded files
        filepath (str): file path of resource file to load

        see 'resource_files/wave/Wave_resource_timeseries.csv' for example wave resource file
        file format for time series for wave energy resource data
            rows 1 and 2: header rows containing info about location
            row 3: headings for time series wave data 
                (month, day, hour, minute, wave height, wave period)
            row 4 and higher: contains data itself
                (significant) wave height in meters
                wave (energy) period in seconds
        """
        super().__init__(lat, lon, year)

        if os.path.isdir(path_resource):
            self.path_resource = path_resource

        self.path_resource = os.path.join(self.path_resource, 'wave')

        self.__dict__.update(kwargs)

        # resource_files files
        self.filename = filepath
        self.format_data()

        logger.info("WaveResource: {}".format(self.filename))

    def download_resource(self):
        #TODO: Add ability to use MHKit for resource downloads
        # https://mhkit-software.github.io/MHKiT/
        raise NotImplementedError

    def format_data(self):
        """
        Format as 'wave_resource_data' dictionary for use in PySAM.
        """
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(self.filename + " does not exist.")

        self.data = self.filename

    @Resource.data.setter
    def data(self, data_file):
        """
        Sets the wave resource data to a dictionary in the SAM Wave format:
            - significant_wave_height: wave height time series data [m]
            - energy period: wave period time series data [s]
            - year
            - month
            - day
            - hour
            - minute
        """
        wavefile_model = wavefile.new()
        #Load resource file
        wavefile_model.WeatherReader.wave_resource_filename_ts = str(self.filename)
        wavefile_model.WeatherReader.wave_resource_model_choice = 1 #Time-series=1 JPD=0

        #Read in resource file, output time series arrays to pass to wave performance module
        wavefile_model.execute() 
        hours = wavefile_model.Outputs.hour

        if len(hours) < 8760:
            # Set up dataframe for data manipulation
            df = pd.DataFrame()
            df['year'] = wavefile_model.Outputs.year 
            df['month'] = wavefile_model.Outputs.month
            df['day'] = wavefile_model.Outputs.day
            df['hour'] = wavefile_model.Outputs.hour
            df['minute'] = wavefile_model.Outputs.minute
            df['date_time'] = pd.to_datetime(dict(year=df.year, month=df.month, day=df.day, hour=df.hour, minute=df.minute))
            df = df.drop(['year','month','day','hour','minute'], axis=1)
            df = df.set_index(['date_time'])
            df['significant_wave_height'] = wavefile_model.Outputs.significant_wave_height
            df['energy_period'] = wavefile_model.Outputs.energy_period

            # Resample data and linearly interpolate to hourly data
            data_df = df.resample("H").mean()
            data_df = data_df.interpolate(method='linear')

            # If data cannot interpolate last hours
            if len(data_df['energy_period']) < 8760:
                last_hour = data_df.index.max()
                missing_hours = 8760 - len(data_df['energy_period'])

                missing_time = pd.date_range(last_hour + pd.Timedelta(hours=1),periods=missing_hours, freq='H')
                missing_rows = pd.DataFrame(index=missing_time, columns=df.columns)
                data_df = pd.concat([data_df, missing_rows]).sort_index()
                data_df = data_df.fillna(method='ffill') # forward fill

            data_df = data_df.reset_index()
            dic = dict()

            # Extract outputs
            dic['significant_wave_height'] = data_df['significant_wave_height']
            dic['energy_period'] = data_df['energy_period']
            dic['year'] = data_df['index'].dt.year
            dic['month'] = data_df['index'].dt.month
            dic['day'] = data_df['index'].dt.day
            dic['hour'] = data_df['index'].dt.hour
            dic['minute'] = data_df['index'].dt.minute

        elif len(hours) == 8760:
            dic = dict()
            dic['significant_wave_height'] = wavefile_model.Outputs.significant_wave_height
            dic['energy_period'] = wavefile_model.Outputs.energy_period
            dic['year'] = wavefile_model.Outputs.year 
            dic['month'] = wavefile_model.Outputs.month
            dic['day'] = wavefile_model.Outputs.day
            dic['hour'] = wavefile_model.Outputs.hour
            dic['minute'] = wavefile_model.Outputs.minute
        else:
            raise ValueError("Resource time-series cannot be subhourly.")

        self._data = dic