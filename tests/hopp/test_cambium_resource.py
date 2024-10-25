import pytest
import os
import json
import glob
import pandas as pd
import requests

from hopp import ROOT_DIR
from hopp.simulation.technologies.resource import CambiumData
from hopp.simulation.technologies.resource.cambium_data import CAMBIUM_BASE_URL
from tests.hopp.utils import DEFAULT_CAMBIUM_DATA_FILE

# Obtain list of default cambium files
DEFAULT_CAMBIUM_DATA_FILE_DIR = DEFAULT_CAMBIUM_DATA_FILE.parent
DEFAULT_CAMBIUM_DATA_FILES_LIST = list(glob.glob(str(DEFAULT_CAMBIUM_DATA_FILE_DIR / "Cambium*.csv")))
DEFAULT_CAMBIUM_DATA_FILES_LIST.sort()

# Set default input arguments for CambiumData class (lat/lon corresponds to Golden, CO campus, aligns with other resource tests / files)
year = 2024
lat = 39.7555
lon = -105.2211

# Test functionality of loading Cambium Data from existing resource files
def test_cambium_load_from_file():
    loaded_from_file_cambium = CambiumData(lat=lat, lon=lon, year=year, filepath=DEFAULT_CAMBIUM_DATA_FILE)
    loaded_from_file_cambium_resource_files = loaded_from_file_cambium.resource_files
    loaded_from_file_cambium_resource_files.sort()
    # Check if list of resource files saved as class object attribute is equivalent to list of default files
    assert loaded_from_file_cambium.resource_files == DEFAULT_CAMBIUM_DATA_FILES_LIST
    # Check total # of files loaded / saved == 6 (2025:2050:5)
    assert len(loaded_from_file_cambium.resource_files) == 6
    assert len(loaded_from_file_cambium.cambium_years) == 6

# Test Cambium API server health
@pytest.mark.dependency(name="api_server_health")
def test_cambium_api_server_health():
    url = CAMBIUM_BASE_URL
    r = requests.get(url)
    assert r.status_code != 500

# Test functionality of calling cambium API and saving to non-default file names by providing a filepath that does not exist
@pytest.mark.dependency(name="test_cambium_save_to_file", depends=["api_server_health"])
def test_cambium_save_to_file():
    test_cambium_data_filepath = DEFAULT_CAMBIUM_DATA_FILE_DIR / "test_Cambium_AnyScenario_hourly_AnyLocation_2025.csv"
    saved_to_file_cambium = CambiumData(lat=lat, lon=lon, year=year, filepath=test_cambium_data_filepath)
    # For each resource file saved as class object attribute, check the file exists / was saved
    for resource_file in saved_to_file_cambium.resource_files:
        assert os.path.isfile(resource_file) == True
    # Check total # of files downloaded / saved == 6 (2025:2050:5)
    assert len(saved_to_file_cambium.resource_files) == 6
    assert len(saved_to_file_cambium.cambium_years) == 6

# Test consistency of data between cached default cambium data files and test_Cambium_... files created from test_cambium_save_to_file()
@pytest.mark.dependency(depends=["test_cambium_save_to_file"])
def test_cambium_data_consistency():
    test_resource_files = list(glob.glob(str(DEFAULT_CAMBIUM_DATA_FILE_DIR / "test_*.csv")))
    test_resource_files.sort()
    test_2025_resource_file_df = pd.read_csv(test_resource_files[0])
    test_2050_resource_file_df = pd.read_csv(test_resource_files[-1])
    default_2025_cambium_data_file_df = pd.read_csv(DEFAULT_CAMBIUM_DATA_FILES_LIST[0])
    default_2050_cambium_data_file_df = pd.read_csv(DEFAULT_CAMBIUM_DATA_FILES_LIST[-1])
    # Check column names from respective dataframes are equivalent
    assert list(test_2025_resource_file_df.columns) == list(default_2025_cambium_data_file_df.columns)
    assert list(test_2050_resource_file_df.columns) == list(default_2050_cambium_data_file_df.columns)
    # Check column values are equivalent, sum all rows for each column and compare test vs default
    assert list(test_2025_resource_file_df.sum()) == list(default_2025_cambium_data_file_df.sum())
    assert list(test_2050_resource_file_df.sum()) == list(default_2050_cambium_data_file_df.sum())

    # Clean up files created for testing
    for test_file in test_resource_files:
        os.remove(test_file)

# Test Cambium API calls
@pytest.mark.dependency(depends=["api_server_health"])
def test_cambium_api():
    # Test base URL with no variables / arguments passed
    url = CAMBIUM_BASE_URL
    r = requests.get(url)
    response_dict = json.loads(r.text)
    assert r.status_code == 200
    assert response_dict["status"] == "FAILED"
    assert response_dict["message"] == "Missing columns: ['project_uuid', 'scenario', 'location_type', 'time_type', 'year']"

    # Test a full URL call
    url = 'https://scenarioviewer.nrel.gov/api/get-data-cache/?project_uuid=0f92fe57-3365-428a-8fe8-0afc326b3b43&scenario=Mid-case with 100% decarbonization by 2035&location_type=GEA Regions 2023&latitude=39.755&longitude=-105.2211&year=2025&time_type=hourly&metric_col=lrmer_co2e'
    r = requests.get(url)
    response_dict = json.loads(r.text)
    assert r.status_code == 200
    assert response_dict["status"] == "SUCCESS"
    assert response_dict["message"][0]["metric"] == 'LRMER: CO2e Combustion+Precombustion [kg/MWh]'

    # Test incorrect API URL
    url = (CAMBIUM_BASE_URL + "a")
    r = requests.get(url)
    assert r.status_code == 404

