import os
import csv
from pathlib import Path
from typing import Union
import requests
import json
import time

from hopp.utilities.log import hybrid_logger as logger
from hopp.simulation.technologies.resource.resource import Resource
from hopp import ROOT_DIR

CAMBIUM_BASE_URL = "https://scenarioviewer.nrel.gov/api/get-data-cache/"

class CambiumData(Resource):
    """
    Class to manage Cambium emissions and grid mix generation data

    Args:
        lat: (float) latitude
        lon: (float) longitude
        year: (int) year
        project_uuid: (string) uuid of the cambium project (Cambium 2022 vs 2023), can be found in the scenarioviewer.nrel.gov URL after selecting the project in the viewer
            Default: Cambium 2023 == '0f92fe57-3365-428a-8fe8-0afc326b3b43'
            Available: 
                Cambium 2022 == '82460f06-548c-4954-b2d9-b84ba92d63e2'
                Cambium 2023 == '0f92fe57-3365-428a-8fe8-0afc326b3b43'
        scenario: (string) scenario name to query as it appears in the Cambium Scenario Viewer
            Default: 'Mid-case with 100% decarbonization by 2035'
            Available: 
                'High demand growth', 'High natural gas prices', 'High renewable energy cost', 'Low natural gas prices', 'Low renewable energy cost', 'Mid-case', 'Mid-case with 100% decarbonization by 2035', 'Mid-case with 95% decarbonization by 2050'
    #       Documentation: See pdf for additional information on each scenario https://www.nrel.gov/docs/fy24osti/88507.pdf
        location_type: (string) geographic resolution of cambium emissions and grid generation data
            Default: 'GEA Regions 2023'
            Available:
                'GEA Regions 2023', 'Nations'
            Note: 'GEA Regions 2023' should be used for most accurate emissions results
        time_type: (string) time resolution of data
            Default: 'hourly'
            Available:
                'hourly' == 8760 array for entire queried year
                'annual' == single annual value
        path_resource: directory where to save downloaded files
        filepath: absolute file path of the .csv to load data from, allows users to manually specify which csv file to use for adhoc edits of values. Default = ""
            If the filepath is specified but the file does not exist: Cambium data will be processed / the API will be called, the file will be created, and data will be saved to the specified file
            Note: Cambium data is saved in up to 5 files corresponding to different years (2025:2050:5), specify only one file here and the code will load / create data from / for all relevant cambium years automatically.
        use_api: Make an API call even if there is an existing data file. Default == False
        kwargs: additional keyword arguments

    """

    # Define a map between arguments and appropriate naming conventions for resource files
    filename_map = {'project_uuid':     {'82460f06-548c-4954-b2d9-b84ba92d63e2':'Cambium22',
                                         '0f92fe57-3365-428a-8fe8-0afc326b3b43':'Cambium23'},
                    'scenario':         {'High demand growth':'HighDemandGrowth',
                                         'High natural gas prices':'HighNGPrices',
                                         'High renewable energy cost':'HighRenewableCost',
                                         'Low natural gas prices':'LowNGPrices',
                                         'Low renewable energy cost':'LowRenewableCost',
                                         'Mid-case':'MidCase',
                                         'Mid-case with 100% decarbonization by 2035':'MidCase100by2035', 
                                         'Mid-case with 95% decarbonization by 2050':'MidCase95by2050'},
                    'location_type':    {'GEA Regions 2023':'GEA',
                                         'Nations':'Nation'},
                    }

    # Define Cambium metrics and mappings needed to pull data from the Cambium API for use in LCA calculations
    # Cambium Long-Run Marginal Emissions Rates
    #NOTE: Only lrmer_co2e_c, lrmer_co2e_p, and lrmer_co2e are used in actual LCA analysis
    #TODO: remove extra lrmer_X values if not needed in analysis
    lrmer_metric_cols = ['lrmer_co2_c',             # CO2 from direct combustion (kg/MWh)
                         'lrmer_ch4_c',             # CH4 from direct combustion (g/MWh)
                         'lrmer_n2o_c',             # N2O from direct combustion (g/MWh)
                         'lrmer_co2_p',             # CO2 from precombustion (kg/MWh)
                         'lrmer_ch4_p',             # CH4 from precombustion (g/MWh)
                         'lrmer_n2o_p',             # N2O from precombustion (g/MWh)
                         'lrmer_co2e_c',            # CO2 equivalent from direct combustion (CO2, CH4, N2O with 100 year GWP - kg/MWh)
                         'lrmer_co2e_p',            # CO2 equivalent from precombustion (CO2, CH4, N20 with 100 year GWP - kg/MWh)
                         'lrmer_co2e'               # CO2 equivalent from combustion and precombustion (== lrmer_co2e_c + lrmer_co2e_p - kg/MWh)
                        ]
    # Cambium Energy Generation Values
    gen_metric_cols = ['generation',                # Generation from all technologies (MWh)
                       'battery_MWh',               # Generation from Electric battery storage (MWh)
                       'biomass_MWh',               # Generation from Biopower and landfill gas (MWh)
                       'beccs_MWh',                 # Generation from Biopower with Carbon Capture and Storage (MWh)
                       'canada_MWh',                # Electricity imported from Canada (MWh)
                       'coal_MWh',                  # Generation from Coal (pulverized, integrated gasificiation combined cycle, and cofired - MWh)
                       'coal-ccs_MWh',              # Generation from Coal with Carbon Capture and Storage (MWh)
                       'csp_MWh',                   # Generation from Concentrating Solar Power with and without thermal storage (MWh)
                       'distpv_MWh',                # Generation from customer-sited Rooftop PV (MWh)
                       'gas-cc_MWh',                # Generation from Natural Gas Combined Cycle (MWh)
                       'gas-cc-ccs_MWh',            # Generation from Natural Gas Combined Cycle with Carbon Capture and Storage (MWh)
                       'gas-ct_MWh',                # Generation from Natural Gas Combustion Turbine (MWh)
                       'geothermal_MWh',            # Generation from Geothermal (MWh)
                       'hydro_MWh',                 # Generation from Hydropower (MWh)
                       'nuclear_MWh',               # Generation from Nuclear Power (MWh)
                       'o-g-s_MWh',                 # Generation from Oil-Gas-Steam (MWh)
                       'phs_MWh',                   # Generation from Pumped Hydro Storage (MWh)
                       'upv_MWh',                   # Generation from Utility-Scale PV (MWh)
                       'wind-ons_MWh',              # Generation from Land-Based Wind (MWh)
                       'wind-ofs_MWh'               # Generation from Offshore Wind (MWh)
                    ]
    # Mapping from Cambium metric_col name to technology name, used when pulling technology specific generation metrics from API
    technology_map = {'battery_MWh':'Battery',
                      'biomass_MWh':'Biopower',
                      'beccs_MWh':'Biopower CCS',
                      'canada_MWh':'Canadian Imports',                     
                      'coal_MWh':'Coal',                            
                      'coal-ccs_MWh': 'Coal CCS',
                      'csp_MWh':'Concentrating Solar Power',
                      'distpv_MWh':'Rooftop PV',                  
                      'gas-cc_MWh':'Natural Gas CC',
                      'gas-cc-ccs_MWh':'Natural Gas CC CCS',
                      'gas-ct_MWh':'Natural Gas CT',
                      'geothermal_MWh':'Geothermal',
                      'hydro_MWh':'Hydropower',
                      'nuclear_MWh':'Nuclear',                  
                      'o-g-s_MWh':'Oil-gas-steam',
                      'upv_MWh':'Utility-scale PV',
                      'phs_MWh':'Pumped Hydro Storage',
                      'wind-ons_MWh':'Land-based Wind',
                      'wind-ofs_MWh':'Offshore Wind'
                    }
    def __init__(
        self,
        lat: float,
        lon: float,
        year: int,
        project_uuid: str = '0f92fe57-3365-428a-8fe8-0afc326b3b43',
        scenario: str = 'Mid-case with 100% decarbonization by 2035',
        location_type: str = 'GEA Regions 2023',
        time_type: str = 'hourly',
        path_resource: Union[str, Path] = ROOT_DIR / "simulation" / "resource_files",
        filepath: Union[str, Path] ="",
        use_api: bool = False,
        **kwargs
    ):
        # Run init of Resource parent class
        super().__init__(lat,lon,year)

        # Set attributes not captured by parent class init
        self.project_uuid = project_uuid
        self.scenario = scenario
        self.location_type = location_type
        self.time_type = time_type

        # Define attribute to store resource file names and years of cambium data associated with class instance 
        self.resource_files = []
        self.cambium_years = []

        # Check if path_resource is a directory, if yes define as self.path_resource attribute
        if os.path.isdir(path_resource):
            self.path_resource = path_resource
        
        # update path with cambium directory
        self.path_resource = os.path.join(self.path_resource, 'cambium')

        # Force override internal definitions if kwargs passed in
        self.__dict__.update(kwargs)

        # Define year to start pulling cambium data from
        if year < 2030:
            self.cambium_year = 2025
        elif year < 2035:
            self.cambium_year = 2030
        elif year < 2040:
            self.cambium_year = 2035
        elif year < 2045:
            self.cambium_year = 2040
        elif year < 2050:
            self.cambium_year = 2045
        else:
            self.cambium_year = 2050

        # Define a location attribute for identifying resource files based on geographic resolution of the Cambium Data (GEA region vs Average across Contiguous United States) instead of lat/lon
        if self.location_type == 'GEA Regions 2023':
            self.location = self.lat_lon_to_gea()
        elif self.location_type == 'Nations':
            self.location = 'Contiguous_United_States'
        else:
            raise ValueError("location_type argument must be either 'GEA Regions 2023' or 'Nations'")

        # Define the filepath and filename for the resource file
        if filepath == "":
            filepath = os.path.join(self.path_resource, 
                                    str(self.filename_map['project_uuid'][self.project_uuid]) + "_" +
                                    str(self.filename_map['scenario'][self.scenario]) + "_" + str(self.time_type) + "_" +
                                    str(self.location) + "_" + str(self.cambium_year) + ".csv")
        self.filename = str(filepath)

        # Check if the download directory exists (HOPP/hopp/simulation/resource_files/cambium), if not make the directory
        self.check_download_dir()

        # Loop through years available in cambium data (2025 through 2050 in 5 year intervals)
        # If a resource file does not already exist in the directory or use_api flag == True, download the data
        # Append / save the file name to self.resource_file and the cambium_year downloaded to self.cambium_years
        print("************************************************************")
        print("Checking for Cambium files and/or calling the Cambium API, this may take up to 1+ minutes")
        for year_to_check in range(self.cambium_year, 2055, 5):
            self.year_to_check = year_to_check
            self.filename = self.filename[:-8] + str(self.year_to_check) + self.filename[-4:]
            if not os.path.isfile(self.filename) or use_api:
                self.download_resource()
            self.resource_files.append(str(self.filename))
            self.cambium_years.append(self.year_to_check)
        print("Cambium data processing complete")
        print("************************************************************")
        # self.format_data()

    def lat_lon_to_gea(self):
        # Cambium API handles mapping of lat/lon to GEA Region, returns the GEA region when query is invalid. Call Cambium API with year not included in their data (2031) to return GEA
        url="{base}?project_uuid={project_uuid}&scenario={scenario}&location_type={location_type}&latitude={latitude}&longitude={longitude}&year={year}&time_type={time_type}&metric_col={metric_col}".format(
            base=CAMBIUM_BASE_URL, project_uuid='0f92fe57-3365-428a-8fe8-0afc326b3b43', scenario='Mid-case with 100% decarbonization by 2035', location_type='GEA Regions 2023',
            latitude=self.latitude, longitude=self.longitude, year=2031, time_type=self.time_type, metric_col='generation'
        )
        response = json.loads(requests.get(url).text)
        gea = response['query']['location'].replace(" ","_")

        return gea

    def call_api(self, filename):
        # Instantiate dictionary to hold data for all variables before writing to file
        response_dict = {}

        n_tries = 0 
        success = False
        while n_tries < 5:

            try:
                # Loop through emissions and 'generation' metric and call API to pull data
                for metric in self.lrmer_metric_cols + [self.gen_metric_cols[0]]:
                    # Define URL for emissions metrics and 'generation' metric
                    url="{base}?project_uuid={project_uuid}&scenario={scenario}&location_type={location_type}&latitude={latitude}&longitude={longitude}&year={year}&time_type={time_type}&metric_col={metric_col}".format(
                    base=CAMBIUM_BASE_URL, project_uuid=self.project_uuid, scenario=self.scenario, location_type=self.location_type,
                    latitude=self.latitude, longitude=self.longitude, year=self.year_to_check, time_type=self.time_type, metric_col = metric
                    )
                    r = requests.get(url)
                    if r:
                        response_dict[metric] = json.loads(r.text)['message'][0]['values']
                    elif r.status_code == 400 or r.status_code == 403:
                        print(r.url)
                        err = r.text
                        text_json = json.loads(r.text)
                        if 'errors' in text_json.keys():
                            err = text_json['errors']
                        raise requests.exceptions.HTTPError(err)
                    elif r.status_code == 404:
                        print(filename)
                        raise requests.exceptions.HTTPError
                    elif r.status_code == 429:
                        raise RuntimeError("Maximum API request rate exceeded!")
                    else:
                        n_tries +=1
                # Loop through technology specific generation metrics and call API to pull data
                for metric in self.gen_metric_cols[1:]:
                    # Define URL for technology specific generation metrics (metric_col='*_MWh', additional arg -> technology='<technology_map[metric_col]>')
                    url="{base}?project_uuid={project_uuid}&scenario={scenario}&location_type={location_type}&latitude={latitude}&longitude={longitude}&year={year}&time_type={time_type}&metric_col=*_MWh&technology={technology}".format(
                    base=CAMBIUM_BASE_URL, project_uuid=self.project_uuid, scenario=self.scenario, location_type=self.location_type,
                    latitude=self.latitude, longitude=self.longitude, year=self.year_to_check, time_type=self.time_type, technology=self.technology_map[metric]
                    )
                    r = requests.get(url)
                    if r:
                        response_dict[metric] = json.loads(r.text)['message'][0]['values']
                    elif r.status_code == 400 or r.status_code == 403:
                        print(r.url)
                        err = r.text
                        text_json = json.loads(r.text)
                        if 'errors' in text_json.keys():
                            err = text_json['errors']
                        raise requests.exceptions.HTTPError(err)
                    elif r.status_code == 404:
                        print(filename)
                        raise requests.exceptions.HTTPError
                    elif r.status_code == 429:
                        raise RuntimeError("Maximum API request rate exceeded!")
                    else:
                        n_tries +=1
                # Save the response dict as a csv file
                localfile = open(filename, mode="w+")
                w = csv.writer(localfile)
                w.writerow(list(response_dict.keys()))
                w.writerows(zip(*list(response_dict.values())))
                localfile.close()
                if os.path.isfile(filename):
                    success = True
                    break
            except requests.exceptions.Timeout:
                time.sleep(0.2)
                n_tries +=1

        return success

    def download_resource(self):
        success = self.call_api(filename=self.filename)

        return success

    #NOTE: format_data() and data() not used in current implementation, these were originally setup for processing data from single file, LCA requires up to 5 cambium files and saving all data may be too large / memory intensive
    #NOTE: As an alterative to loading all Cambium data into memory, current logic stores file names in self.resource_files which can be used to load data when needed for LCA calculations
    def format_data(self):
        """
        """
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f"{self.filename} does not exist. Try `download_resource` first.")

        self.data = self.filename

    @Resource.data.setter
    def data(self, data_dict):
        pass

##Adhoc testing
# below coordinates and year used to create the DEFAULT_CAMBIUM_DATA_FILE (Cambium23_MidCase100by2035_hourly_West_Connect_North_<year>.csv) corresponds to Golden, CO campus coordinates
# if __name__ == '__main__':
#     test = CambiumData(lat=39.755, lon=-105.2211, year=2024)
#     print(test.resource_files)
#     print(test.cambium_years)