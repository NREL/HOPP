from pytest import approx, fixture
import os
import yaml
import re
# import warnings

from hopp import ROOT_DIR
from hopp.simulation.technologies.resource import GREETData
from tests.hopp.utils import DEFAULT_GREET_DATA_FILE

# Ignore / stop print IO of UserWarnings when opening the GREET excel docs with openpyxl
# warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

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

# Test functionality of preprocessing GREET from parent excel documents
def test_preprocess_greet():
    preprocessed_greet = GREETData(greet_year=year,preprocess_greet=True)
    # Assert the data dictionary keys and values from the preprocessed_greet instance are equivalent to those loaded from the default greet data file
    assert preprocessed_greet.data == default_greet_data


    