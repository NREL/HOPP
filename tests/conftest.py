"""
Pytest configuration file.
"""
import os

from hopp import TEST_ENV_VAR

def pytest_sessionstart(session):
    os.environ["ENV"] = TEST_ENV_VAR

