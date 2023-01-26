import pytest
import os
from numpy.testing import assert_almost_equal

from hopp.offshore.fixed_platform_h2 import FixedPlatformDesign, FixedPlatformInstallation


@pytest.fixture
def config():
    orbit_config_file = os.path.abspath(__file__).parent.parent.parent
    print(orbit_config_file)
    
def test_init():
    '''
    Test the initialization code from fixed_platform.py
    '''
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

