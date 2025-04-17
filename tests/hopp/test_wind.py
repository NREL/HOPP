import math

import PySAM.Windpower as windpower
import pytest
from pytest import fixture,approx
from hopp.simulation.technologies.wind.wind_plant import WindPlant, WindConfig
from hopp.utilities import load_yaml
from tests.hopp.utils import create_default_site_info
from hopp import ROOT_DIR
import os
from hopp.simulation.technologies.sites import SiteInfo


@fixture
def site():
    return create_default_site_info()

wind_default_elevation = 0
wind_default_rated_output = 2000
wind_default_max_cp = 0.45
wind_default_max_tip_speed = 80
wind_default_max_tip_speed_ratio = 8
wind_default_cut_in_speed = 4
wind_default_cut_out_speed = 25
wind_default_drive_train = 0

powercurveKW = (
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 56.9014, 72.8929, 90.7638, 110.618, 132.561,
    156.696, 183.129, 211.962, 243.302, 277.251, 313.915, 353.398, 395.805, 441.239, 489.805,
    541.608, 596.752, 655.341, 717.481, 783.274, 852.826, 926.241, 1003.62, 1088.85, 1174.66,
    1260.47, 1346.28, 1432.09, 1517.9, 1603.71, 1689.53, 1775.34, 1861.15, 1946.96, 2000, 2000,
    2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000,
    2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000,
    2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
)

powercurveWS = (
    0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5,
    4.75, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9, 9.25,
    9.5, 9.75, 10, 10.25, 10.5, 10.75, 11, 11.25, 11.5, 11.75, 12, 12.25, 12.5, 12.75, 13, 13.25,
    13.5, 13.75, 14, 14.25, 14.5, 14.75, 15, 15.25, 15.5, 15.75, 16, 16.25, 16.5, 16.75, 17, 17.25,
    17.5, 17.75, 18, 18.25, 18.5, 18.75, 19, 19.25, 19.5, 19.75, 20, 20.25, 20.5, 20.75, 21, 21.25,
    21.5, 21.75, 22, 22.25, 22.5, 22.75, 23, 23.25, 23.5, 23.75, 24, 24.25, 24.5, 24.75, 25, 25.25,
    25.5, 25.75, 26, 26.25, 26.5, 26.75, 27, 27.25, 27.5, 27.75, 28, 28.25, 28.5, 28.75, 29, 29.25,
    29.5, 29.75, 30, 30.25, 30.5, 30.75, 31, 31.25, 31.5, 31.75, 32, 32.25, 32.5, 32.75, 33, 33.25,
    33.5, 33.75, 34, 34.25, 34.5, 34.75, 35, 35.25, 35.5, 35.75, 36, 36.25, 36.5, 36.75, 37, 37.25,
    37.5, 37.75, 38, 38.25, 38.5, 38.75, 39, 39.25, 39.5, 39.75, 40
)


def test_wind_powercurve_pysam():
    model = windpower.default("WindpowerSingleowner")
    model.Turbine.wind_turbine_rotor_diameter = 75

    # calculate system capacity.  To evaluate other turbines, update the defaults dictionary
    model.Turbine.calculate_powercurve(
        wind_default_rated_output,
        int(model.Turbine.wind_turbine_rotor_diameter),
        wind_default_elevation,
        wind_default_max_cp,
        wind_default_max_tip_speed,
        wind_default_max_tip_speed_ratio,
        wind_default_cut_in_speed,
        wind_default_cut_out_speed,
        wind_default_drive_train
    )

    windspeeds_truth = [round(x, 2) for x in powercurveWS]
    windspeeds_calc = [round(x, 2) for x in model.Turbine.wind_turbine_powercurve_windspeeds]
    powercurve_truth = [round(x, 0) for x in powercurveKW]
    powercurve_calc = [round(x, 0) for x in model.Turbine.wind_turbine_powercurve_powerout]

    assert all([a == b for a, b in zip(windspeeds_truth, windspeeds_calc)])
    assert all([a == b for a, b in zip(powercurve_truth, powercurve_calc)])


def test_changing_n_turbines_pysam(site):
    # test with gridded layout
    config = WindConfig.from_dict({'num_turbines': 10, "turbine_rating_kw": 2000})
    model = WindPlant(site, config=config)
    assert(model.system_capacity_kw == 20000)
    for n in range(1, 20):
        model.num_turbines = n
        assert model.num_turbines == n, "n turbs should be " + str(n)
        assert model.system_capacity_kw == approx(20000, 1), "system capacity different when n turbs " + str(n)

    # test with row layout


def test_changing_rotor_diam_recalc_pysam(site):
    config = WindConfig.from_dict({'num_turbines': 10, "turbine_rating_kw": 2000})
    model = WindPlant(site, config=config)
    assert model.system_capacity_kw == 20000
    diams = range(50, 70, 140)
    for d in diams:
        model.rotor_diameter = d
        assert model.rotor_diameter == d, "rotor diameter should be " + str(d)
        assert model.turb_rating == 2000, "new rating different when rotor diameter is " + str(d)


def test_changing_turbine_rating_pysam(site):
    # powercurve scaling
    config = WindConfig.from_dict({'num_turbines': 24, "turbine_rating_kw": 2000})
    model = WindPlant(site, config=config)
    n_turbs = model.num_turbines
    for n in range(1000, 3000, 150):
        model.turb_rating = n
        assert model.system_capacity_kw == model.turb_rating * n_turbs, "system size error when rating is " + str(n)


def test_changing_powercurve_pysam(site):
    # with power curve recalculation requires diameter changes
    config = WindConfig.from_dict({'num_turbines': 24, "turbine_rating_kw": 2000})
    model = WindPlant(site, config=config)
    n_turbs = model.num_turbines
    d_to_r = model.rotor_diameter / model.turb_rating
    for n in range(1000, 3001, 500):
        d = math.ceil(n * d_to_r * 1)
        model.modify_powercurve(d, n)
        assert model.turb_rating == approx(n, 0.1), "turbine rating should be " + str(n)
        assert model.system_capacity_kw == approx(model.turb_rating * n_turbs, 0.1), "size error when rating is " + str(n)


def test_changing_system_capacity_pysam(site):
    # adjust number of turbines, system capacity won't be exactly as requested
    config = WindConfig.from_dict({'num_turbines': 20, "turbine_rating_kw": 1000})
    model = WindPlant(site, config=config)
    rating = model.turb_rating
    for n in range(1000, 20000, 1000):
        model.system_capacity_by_num_turbines(n)
        assert model.turb_rating == rating, str(n)
        assert model.system_capacity_kw == rating * round(n/rating)

    # adjust turbine rating first, system capacity will be exact
    model = WindPlant(site, config=config)
    for n in range(40000, 60000, 1000):
        model.system_capacity_by_rating(n)
        assert model.system_capacity_kw == approx(n)


#################### FLORIS tests ################
def test_floris_num_turbines(site):
    floris_config_path = (
        ROOT_DIR.parent / "tests" / "hopp" / "inputs" / "floris_config.yaml"
    )
    
    site.wind_resource.hub_height_meters = 90.0
    config = WindConfig.from_dict({'num_turbines': 16, "turbine_rating_kw": 5000, "model_name": "floris", "timestep": [1, 8760], "floris_config": floris_config_path})
    model = WindPlant(site, config=config)
    xcoords, ycoords = model._system_model.wind_farm_layout
    assert len(xcoords) == config.num_turbines
    assert len(ycoords) == model.num_turbines
    assert model._system_model.nTurbs == model.num_turbines
    

def test_changing_rotor_diam_recalc_floris(site):
    floris_config_path = (
        ROOT_DIR.parent / "tests" / "hopp" / "inputs" / "floris_config.yaml"
    )
    site.wind_resource.hub_height_meters = 90.0
    config = WindConfig.from_dict({'num_turbines': 4, "turbine_rating_kw": 5000, "model_name": "floris", "timestep": [1, 8760], "floris_config": floris_config_path})
    model = WindPlant(site, config=config)
    assert model._system_model.system_capacity  == 20000
    diams = range(50, 70, 140)
    for d in diams:
        model._system_model.wind_turbine_rotor_diameter = d
        assert model._system_model.wind_turbine_rotor_diameter == d, "rotor diameter should be " + str(d)

def test_changing_turbine_rating_floris(site):
    
    floris_config_path = (ROOT_DIR.parent / "tests" / "hopp" / "inputs" / "floris_config.yaml")
    config = WindConfig.from_dict(
        {
            'num_turbines': 4,
            "turbine_rating_kw": 1000,
            "model_name": "floris",
            "timestep": [1, 8760],
            "floris_config": floris_config_path
        }
    )
    site.wind_resource.hub_height_meters = 90.0
    config = WindConfig.from_dict({'num_turbines': 4, "turbine_rating_kw": 1000, "model_name": "floris", "timestep": [1, 8760], "floris_config": floris_config_path})
    with pytest.raises(ValueError) as err:
        model = WindPlant(site, config=config)
    
    err_str = "Input turbine rating (1000 kW) does not match rating from floris power-curve (5000.0 kW)"
    assert err_str in str(err.value)
   

def test_changing_system_capacity_floris(site):
    # this will fail now
    floris_config_path = (
        ROOT_DIR.parent / "tests" / "hopp" / "inputs" / "floris_config.yaml"
    )
    site.wind_resource.hub_height_meters = 90.0
    config = WindConfig.from_dict({'num_turbines': 4, "turbine_rating_kw": 5000, "model_name": "floris", "timestep": [1, 8760], "floris_config": floris_config_path})
    model = WindPlant(site, config=config)
    
    new_num_turbs = 16
    new_capacity_kW = new_num_turbs*config.turbine_rating_kw
    rating = model._system_model.turb_rating
    
    assert model._system_model.nTurbs == 4
    assert model._system_model.turb_rating == rating
    assert model._system_model.system_capacity == 20000
    
    model.system_capacity_by_num_turbines(new_capacity_kW)
    assert model._system_model.nTurbs == new_num_turbs
    assert model._system_model.system_capacity == new_capacity_kW 

def test_alaska_wind_pysam():
    site_data = {
        "lat": 66.68,
        "lon": -162.5,
        "year": 2019,
        "site_details":
            {
            "site_area_km2": 1.0,
            "site_shape":"square",
            }
    }
    alaska_wind_resource_file = os.path.join(
    ROOT_DIR, "simulation", "resource_files", "wind", 
    "66.68_-162.5_WTK_Alaksa_2019_60min_80m_100m.csv"
    )
    site_info = {
        "data": site_data,
        "wind_resource_file": alaska_wind_resource_file,
        "wind_resource_region": "ak",
        "wind": True,
        "solar": False,
        "hub_height": 90.0,
    }
    site = SiteInfo.from_dict(site_info)
    config = WindConfig.from_dict({'num_turbines': 5, "turbine_rating_kw": 2000})
    model = WindPlant(site, config=config)
    model._system_model.execute(1)
    assert model._system_model.Outputs.capacity_factor == approx(45.85,abs = 0.1)


def test_alaska_wind_floris():
    site_data = {
        "lat": 66.68,
        "lon": -162.5,
        "year": 2019,
        "site_details":
            {
            "site_area_km2": 1.0,
            "site_shape":"square",
            }
    }
    alaska_wind_resource_file = os.path.join(
    ROOT_DIR, "simulation", "resource_files", "wind", 
    "66.68_-162.5_WTK_Alaksa_2019_60min_80m_100m.csv"
    )
    site_info = {
        "data": site_data,
        "wind_resource_file": alaska_wind_resource_file,
        "wind_resource_region": "ak",
        "wind": True,
        "solar":False,
        "hub_height": 90.0,
    }
    site = SiteInfo.from_dict(site_info)
    floris_config_path = (
        ROOT_DIR.parent / "tests" / "hopp" / "inputs" / "floris_config.yaml"
    )
    wind_config_input = {
        'num_turbines': 4,
        "turbine_rating_kw": 5000,
        "model_name": "floris",
        "timestep": [1, 8760],
        "resource_parse_method":"weighted_average",
        "floris_config": floris_config_path
    }
    config = WindConfig.from_dict(wind_config_input)
    model = WindPlant(site, config=config)
    model._system_model.execute(1)
    assert model._system_model.annual_energy == approx(78514174,rel=1e-6)

def test_bchrrr_wind_pysam():
    site_data = {
        "lat": 35.2018863,
        "lon": -101.945027,
        "elev": 1099,
        "year": 2015,
        "tz": -6,
        "site_details":
            {
            "site_area_km2": 1.0,
            "site_shape":"square",
            }
    }
    bchrrr_wind_resource_file = os.path.join(
    ROOT_DIR, "simulation", "resource_files", "wind", 
    "35.2018863_-101.945027_BC_HRRR_2015_60min_80m_100m.csv"
    )
    site_info = {
        "data": site_data,
        "wind_resource_file": bchrrr_wind_resource_file,
        "wind_resource_origin": "BC-HRRR",
        "wind": True,
        "solar": False,
        "hub_height": 90.0,
    }
    site = SiteInfo.from_dict(site_info)
    config = WindConfig.from_dict({'num_turbines': 5, "turbine_rating_kw": 2000})

    model = WindPlant(site, config=config)
    model._system_model.execute(1)
    assert model._system_model.Outputs.capacity_factor == approx(35.97,abs = 0.1) 
    
def test_bchrrr_wind_floris():
    site_data = {
        "lat": 35.2018863,
        "lon": -101.945027,
        "elev": 1099,
        "year": 2015,
        "tz": -6,
        "site_details":
            {
            "site_area_km2": 1.0,
            "site_shape":"square",
            }
    }
    bchrrr_wind_resource_file = os.path.join(
    ROOT_DIR, "simulation", "resource_files", "wind", 
    "35.2018863_-101.945027_BC_HRRR_2015_60min_80m_100m.csv"
    )
    site_info = {
        "data": site_data,
        "wind_resource_file": bchrrr_wind_resource_file,
        "wind_resource_origin": "BC-HRRR",
        "wind": True,
        "solar":False,
        "hub_height": 90.0,
    }
    site = SiteInfo.from_dict(site_info)
    floris_config_path = (
        ROOT_DIR.parent / "tests" / "hopp" / "inputs" / "floris_config.yaml"
    )
    wind_config_input = {
        'num_turbines': 4,
        "turbine_rating_kw": 5000,
        "model_name": "floris",
        "timestep": [1, 8760],
        "resource_parse_method":"weighted_average",
        "floris_config": floris_config_path
    }
    config = WindConfig.from_dict(wind_config_input)
    model = WindPlant(site, config=config)
    model._system_model.execute(1)
    assert model._system_model.annual_energy == approx(69526231,rel=1e-6)    
