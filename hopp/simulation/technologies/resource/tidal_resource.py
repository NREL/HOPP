import os
import pandas as pd
import PySAM.TidalFileReader as tidalfile

from hopp.utilities.log import hybrid_logger as logger
from hopp.simulation.technologies.resource.resource import Resource

class TidalResource(Resource):
    """
    Class to manage Tidal Resource data.

    This class loads, processes, and formats tidal energy resource data, 
    either from a file or a provided dataset, for compatibility with 
    PySAM's tidal energy models.
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
        Initializes the TidalResource object.

        Args:
            lat (float): Latitude of the resource location.
            lon (float): Longitude of the resource location.
            year (int): Year of the resource data.
            path_resource (str, optional): Directory where downloaded files are saved. Defaults to "".
            filepath (str, optional): File path of the resource file to load. Defaults to "".
            **kwargs: Additional keyword arguments.

        Notes:
            The tidal resource data should be in the format:
                - Rows 1 and 2: Header rows with location info.
                - Row 3: Column headings for time-series data 
                    - (`Year`, `Month`, `Day`, `Hour`, `Minute`, `Speed`).
                - Rows 4+: Data values:
                - `Speed` (current speed) in meters/second.

            Example file: 
                `hopp/simulation/resource_files/tidal/Tidal_resource_timeseries.csv`
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
        """
        Placeholder for downloading tidal resource data.

        Raises:
            NotImplementedError: Currently, downloading functionality is not implemented.
        """
        raise NotImplementedError

    def format_data(self):
        """
        Formats tidal resource data as a dictionary for PySAM.

        Raises:
            FileNotFoundError: If the specified resource file does not exist.
        """
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(self.filename + " does not exist.")

        self.data = self.filename

    @Resource.data.setter
    def data(self, data_file):
        """
        Sets the tidal resource data in PySAM's tidal energy format.

        Args:
            data_file (str): File path to the tidal resource data.

        Raises:
            ValueError: If the resource time series contains sub-hourly data.

        The output dictionary includes:
            - `speed` (list[float]): Current speed data [m/s].
            - `year` (list[int]): Year timestamps.
            - `month` (list[int]): Month timestamps.
            - `day` (list[int]): Day timestamps.
            - `hour` (list[int]): Hour timestamps.
            - `minute` (list[int]): Minute timestamps.

        If the time series is incomplete (less than 8760 hours), the function 
        linearly interpolates missing values to create a complete hourly dataset.
        """
        tidalfile_model = tidalfile.new()
        #Load resource file
        tidalfile_model.WeatherReader.tidal_resource_filename = str(self.filename)
        tidalfile_model.WeatherReader.tidal_resource_model_choice = 1 #Time-series=1 JPD=0

        #Read in resource file, output time series arrays to pass to wave performance module
        tidalfile_model.execute() 
        hours = tidalfile_model.Outputs.hour

        if len(hours) < 8760:
            # Set up dataframe for data manipulation
            df = pd.DataFrame()
            df['year'] = tidalfile_model.Outputs.year 
            df['month'] = tidalfile_model.Outputs.month
            df['day'] = tidalfile_model.Outputs.day
            df['hour'] = tidalfile_model.Outputs.hour
            df['minute'] = tidalfile_model.Outputs.minute
            df['date_time'] = pd.to_datetime(dict(year=df.year, month=df.month, day=df.day, hour=df.hour, minute=df.minute))
            df = df.drop(['year','month','day','hour','minute'], axis=1)
            df = df.set_index(['date_time'])
            df['tidal_velocity'] = tidalfile_model.Outputs.tidal_velocity
            
            # Resample data and linearly interpolate to hourly data
            data_df = df.resample("h").mean()
            data_df = data_df.interpolate(method='linear')
            

            # If data cannot interpolate last hours
            if len(data_df['tidal_velocity']) < 8760:
                last_hour = data_df.index.max()
                missing_hours = 8760 - len(data_df['tidal_velocity'])

                missing_time = pd.date_range(last_hour + pd.Timedelta(hours=1),periods=missing_hours, freq='h')
                missing_rows = pd.DataFrame(index=missing_time, columns=df.columns)
                data_df = pd.concat([data_df, missing_rows]).sort_index()
                data_df = data_df.ffill() # forward fill

            data_df = data_df.reset_index()
            dic = dict()

            # Extract outputs
            dic['tidal_velocity'] = data_df['tidal_velocity']
            print(data_df.head())
            dic['year'] = data_df['index'].dt.year
            dic['month'] = data_df['index'].dt.month
            dic['day'] = data_df['index'].dt.day
            dic['hour'] = data_df['index'].dt.hour
            dic['minute'] = data_df['index'].dt.minute

        elif len(hours) == 8760:
            dic = dict()
            # Extract outputs
            dic['tidal_velocity'] = tidalfile_model.Outputs.tidal_velocity
        else:
            raise ValueError("Resource time-series cannot be subhourly.")

        self._data = dic