import os
from PySAM.ResourceTools import SRW_to_wind_data
from hopp.tools.resource.wind_tools import (
    calculate_air_density,
    calculate_air_density_losses,
    parse_resource_data,
    weighted_parse_resource_data
)
from hopp.simulation.technologies.resource import WindResource
from hopp import ROOT_DIR
from pytest import fixture, approx
from numpy.testing import assert_array_almost_equal
import numpy as np
from hopp.tools.resource.pysam_wind_tools import combine_wind_files

wind_resource_file_multi_heights = os.path.join(
    ROOT_DIR, "simulation", "resource_files", "wind", 
    "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
)

wind_resource_file_single_height = os.path.join(
    ROOT_DIR, "simulation", "resource_files", "wind", 
    "35.2018863_-101.945027_windtoolkit_2012_60min_100m.srw"
)

@fixture
def wind_resource_data_90m():
    wind_resource_data_dict = SRW_to_wind_data(wind_resource_file_multi_heights)
    return WindResource(
        lat = 35.2018863,
        lon = -101.945027,
        year = 2012,
        wind_turbine_hub_ht = 90,
        resource_data = wind_resource_data_dict
    )

@fixture
def wind_resource_data_85m():
    wind_resource_data_dict = SRW_to_wind_data(wind_resource_file_multi_heights)
    return WindResource(
        lat = 35.2018863,
        lon = -101.945027,
        year = 2012,
        wind_turbine_hub_ht = 85,
        resource_data = wind_resource_data_dict
    )

@fixture
def wind_resource_data_100m():
    wind_resource_data_dict = SRW_to_wind_data(wind_resource_file_single_height)
    return WindResource(
        lat = 35.2018863,
        lon = -101.945027,
        year = 2012,
        wind_turbine_hub_ht = 100,
        resource_data = wind_resource_data_dict
    )

def test_sea_level_air_density():
    elevation = 0.0 #meters
    air_dens = calculate_air_density(elevation)
    assert air_dens == approx(1.225, rel = 1e-3)
   
def test_mile_high_air_density():
    #test elevation at 1 mile above sea level
    elevation = 1609.34 #meters
    air_dens = calculate_air_density(elevation)
    assert air_dens == approx(1.05, rel = 1e-3)

def test_sea_level_air_density_losses():
    elevation = 0.0 #meters
    loss_percent = calculate_air_density_losses(elevation)
    assert loss_percent == 0.0

def test_mile_high_air_density_losses():
    elevation = 1609.34 #meters
    loss_percent = calculate_air_density_losses(elevation)
    assert loss_percent == approx(14.325, rel = 1e-3)

def test_weighted_parsing_100m(wind_resource_data_100m):
    wind_speeds, wind_dirs = weighted_parse_resource_data(wind_resource_data_100m)
    
    assert wind_speeds[0] == approx(wind_resource_data_100m.data['data'][0][2], abs=1e-3)
    assert wind_dirs[0] == approx(wind_resource_data_100m.data['data'][0][3], abs=1e-3)

def test_weighted_parsing_90m(wind_resource_data_90m):
    wind_speeds, wind_dirs = weighted_parse_resource_data(wind_resource_data_90m)
    t0_wind_speeds = [wind_resource_data_90m.data['data'][0][2],wind_resource_data_90m.data['data'][0][6]]
    t0_wind_dirs = [wind_resource_data_90m.data['data'][0][3],wind_resource_data_90m.data['data'][0][7]]

    assert wind_speeds[0] > min(t0_wind_speeds)
    assert wind_speeds[0] < max(t0_wind_speeds)
    assert wind_dirs[0] > min(t0_wind_dirs)
    assert wind_dirs[0] < max(t0_wind_dirs)

def test_weighted_parsing_85m(wind_resource_data_85m):
    wind_speeds, wind_dirs = weighted_parse_resource_data(wind_resource_data_85m)
    ws_frac_80m = wind_speeds[0]/wind_resource_data_85m.data['data'][0][2]
    ws_frac_100m = wind_speeds[0]/wind_resource_data_85m.data['data'][0][6]
    wd_frac_80m = wind_dirs[0]/wind_resource_data_85m.data['data'][0][3]
    wd_frac_100m = wind_dirs[0]/wind_resource_data_85m.data['data'][0][7]
    assert ws_frac_80m > 1
    assert ws_frac_100m < 1
    assert wd_frac_80m > 1
    assert wd_frac_100m < 1

def test_average_parsing_100m(wind_resource_data_100m):
    wind_speeds, wind_dirs = parse_resource_data(wind_resource_data_100m)
    
    assert wind_speeds[0] == approx(wind_resource_data_100m.data['data'][0][2], abs=1e-3)
    assert wind_dirs[0] == approx(wind_resource_data_100m.data['data'][0][3], abs=1e-3)

def test_average_parsing_90m(wind_resource_data_90m):
    wind_speeds, wind_dirs = parse_resource_data(wind_resource_data_90m)
    t0_wind_speeds = [wind_resource_data_90m.data['data'][0][2],wind_resource_data_90m.data['data'][0][6]]
    t0_wind_dirs = [wind_resource_data_90m.data['data'][0][3],wind_resource_data_90m.data['data'][0][7]]

    assert wind_speeds[0] > min(t0_wind_speeds)
    assert wind_speeds[0] < max(t0_wind_speeds)
    assert wind_dirs[0] > min(t0_wind_dirs)
    assert wind_dirs[0] < max(t0_wind_dirs)

def test_average_parsing_85m(wind_resource_data_85m):
    wind_speeds, wind_dirs = parse_resource_data(wind_resource_data_85m)
    t0_wind_speeds = [wind_resource_data_85m.data['data'][0][2],wind_resource_data_85m.data['data'][0][6]]
    t0_wind_dirs = [wind_resource_data_85m.data['data'][0][3],wind_resource_data_85m.data['data'][0][7]]

    assert wind_speeds[0] == approx(np.mean(t0_wind_speeds), rel = 1e-3)
    assert wind_dirs[0] == approx(np.mean(t0_wind_dirs), rel = 1e-3)

def test_weighted_vs_average_parsing_90m(wind_resource_data_90m):
    avg_wind_speeds, avg_wind_dirs = parse_resource_data(wind_resource_data_90m)
    wavg_wind_speeds, wavg_wind_dirs = weighted_parse_resource_data(wind_resource_data_90m)
    assert_array_almost_equal(avg_wind_speeds,wavg_wind_speeds,decimal=3)
    assert_array_almost_equal(avg_wind_dirs,wavg_wind_dirs,decimal=3)

def test_pysam_combine_wind_files_csv():
    alaska_wind_resource_file = os.path.join(
    ROOT_DIR, "simulation", "resource_files", "wind", 
    "66.68_-162.5_WTK_Alaksa_2019_60min_80m_100m.csv"
    )
    resource_heights = [80.0,100.0]
    wind_data = combine_wind_files(alaska_wind_resource_file,resource_heights)
    assert len(wind_data["heights"]) == 7
    
def test_pysam_combine_wind_files_srw():

    resource_heights = [80.0,100.0]
    wind_data = combine_wind_files(wind_resource_file_multi_heights,resource_heights)
    assert len(wind_data["heights"]) == 8