"""
Pytest configuration file.
"""
import os

from hopp import TEST_ENV_VAR
from hopp.utilities.keys import set_nrel_key_dot_env

def pytest_sessionstart(session):
    os.environ["ENV"] = TEST_ENV_VAR

    # Set a dummy API key
    os.environ["NREL_API_KEY"] = "a" * 40
    set_nrel_key_dot_env()

