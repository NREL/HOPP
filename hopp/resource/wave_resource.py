import pandas as pd
import PySAM.WaveFileReader as wavefile

from hopp.log import hybrid_logger as logger
from hopp.resource.resource import *


class WaveResource(Resource):
    """
    Class to manage Wave Resource data
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

        self.path_resource = os.path.join(self.path_resource, 'wave')

        self.__dict__.update(kwargs)

        # resource_files files
        if filepath == "":
            self.filename = ""
        else:
            self.filename = filepath

        if not os.path.isfile(self.filename):
            # self.download_resource()
            raise ValueError("Wave resource file must be loaded.") # Remove ValueError once resource can be downloaded.
            

        self.format_data()

        logger.info("WaveResource: {}".format(self.filename))

    def download_resource(self):
        #TODO: Add ability to use MHKit for resource downloads
        # https://mhkit-software.github.io/MHKiT/
        pass

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
        Sets the wave resource data to a dictionary in the SAM Wave format

        :key significant_wave_height: sequence, wave height time series data [m]
        :key energy period: sequence, wave period time series data [s]
        :key year: sequence
        :key month: sequence
        :key day: sequence
        :key hour: sequence
        :key minute: sequence
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

            # Extract outputs
            print("data file",data_file)
            data_file['significant_wave_height'] = data_df['significant_wave_height']
            data_file['energy_period'] = data_df['energy_period']
            data_file['year'] = data_df['index'].dt.year
            data_file['month'] = data_df['index'].dt.month
            data_file['day'] = data_df['index'].dt.day
            data_file['hour'] = data_df['index'].dt.hour
            data_file['minute'] = data_df['index'].dt.minute

        if len(hours) == 8760:
            data_file['significant_wave_height'] = wavefile_model.Outputs.significant_wave_height
            data_file['energy_period'] = wavefile_model.Outputs.energy_period
            data_file['year'] = wavefile_model.Outputs.year 
            data_file['month'] = wavefile_model.Outputs.month
            data_file['day'] = wavefile_model.Outputs.day
            data_file['hour'] = wavefile_model.Outputs.hour
            data_file['minute'] = wavefile_model.Outputs.minute
        else:
            raise ValueError("Resource time-series cannot be subhourly.")

        self._data = data_file