import os
import yaml
import re

from hopp import ROOT_DIR
from hopp.simulation.technologies.resource import GREETData
from tests.hopp.utils import DEFAULT_GREET_DATA_FILE

# Load default greet data dict from yaml
with open(DEFAULT_GREET_DATA_FILE, mode='r') as f:
    default_greet_data = yaml.load(f, Loader=yaml.SafeLoader)

# Extract year / version of GREET based on DEFAULT_GREET_DATA_FILE
year = int(re.findall(r'\d+',str(DEFAULT_GREET_DATA_FILE))[-1])

# Test functionality of loading GREET data from the greet_<year>_preprocessed.yaml file
def test_greet_load_from_file():
    loaded_from_file_greet = GREETData(filepath=DEFAULT_GREET_DATA_FILE)
    # Assert the data dictionary keys and values from the loaded_from_file_greet instance are equivalent to those loaded from the default greet data file
    assert loaded_from_file_greet.data == default_greet_data

# Test functionality of preprocessing GREET from parent excel docs and saving to non-default file name by providing a filepath that does not exist
def test_greet_save_to_file():
    TEST_GREET_DATA_FILEPATH = DEFAULT_GREET_DATA_FILE.parent / "test_greet_2023_preprocessed.yaml"
    saved_to_file_greet = GREETData(greet_year=year, filepath=TEST_GREET_DATA_FILEPATH)
    assert os.path.isfile(TEST_GREET_DATA_FILEPATH) == True
    assert saved_to_file_greet.data == default_greet_data

    # Clean up file created for testing
    os.remove(TEST_GREET_DATA_FILEPATH)

# Test functionality of preprocessing GREET from parent excel docs by setting input argument preprocess_greet = True
def test_preprocess_greet():
    preprocessed_greet = GREETData(greet_year=year,preprocess_greet=True)
    # Assert the data dictionary keys and values from the preprocessed_greet instance are equivalent to those loaded from the default greet data file
    assert preprocessed_greet.data == default_greet_data


    