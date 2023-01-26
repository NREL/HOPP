import pytest
import os
from numpy.testing import assert_almost_equal

from ORBIT import load_config
from hopp.offshore.fixed_platform_h2 import FixedPlatformDesign, FixedPlatformInstallation


@pytest.fixture
def config():
    offshore_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir,'hopp','offshore'))

    return load_config(os.path.join(offshore_path,"example_fixed_project_h2.yaml"))
    
def test_init(config):
    '''
    Test the initialization code from fixed_platform.py
    '''
    print(config)
    pass

def test_calc_platform_capex():
    '''
    Test the code that calculates the CapEx from fixed_platform.py
    '''
    pass

def test_calc_platform_opex():
    '''
    Test the code that calculates the OpEx from fixed_platform.py
    '''
    pass

