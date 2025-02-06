
from hopp.tools.resource.wind_tools import calculate_air_density_for_elevation

def test_sea_level_air_density():
    elevation = 0.0 #meters
    air_dens = calculate_air_density_for_elevation(elevation)
    assert round(air_dens,3) == 1.225

def test_mile_high_air_density():
    #test elevation at 1 mile above sea level
    elevation = 1609.34 #meters
    air_dens = calculate_air_density_for_elevation(elevation)
    assert round(air_dens,2) == 1.05