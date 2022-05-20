import pytest
from pytest import approx
import os
from pathlib import Path

from hybrid.resource import SolarResource, WindResource
from hybrid.keys import set_nrel_key_dot_env

import PySAM.Windpower as wp
import PySAM.Pvwattsv8 as pv


set_nrel_key_dot_env()

dir_path = os.path.dirname(os.path.realpath(__file__))

year = 2012
lat = 39.7555
lon = -105.2211
hubheight = 80


@pytest.fixture
def solar_resource():
    return SolarResource(lat=lat, lon=lon, year=year, api='nrel')


@pytest.fixture
def wind_resource():
    return WindResource(lat=lat, lon=lon, year=year, api='nrel', wind_turbine_hub_ht=hubheight)


def test_solar_nsrdb(solar_resource):
    data = solar_resource.data
    for key in ('df', 'dn', 'wspd', 'tdry', 'year', 'month', 'day', 'hour', 'minute', 'tz'):
        assert(key in data)
    model = pv.default("PVWattsNone")
    model.SolarResource.solar_resource_file = solar_resource.filename
    model.execute(0)
    assert(model.Outputs.annual_energy == approx(9275, 0.1))
    model = pv.default("PVWattsNone")
    model.SolarResource.solar_resource_data = solar_resource.data
    model.execute(1)
    assert(model.Outputs.annual_energy == approx(9275, 0.1))


def test_nsrdb_download(solar_resource):
    solar_resource.download_resource()


def test_wind_windtoolkit(wind_resource):
    data = wind_resource.data
    for key in ('heights', 'fields', 'data'):
        assert (key in data)
    model = wp.default("WindPowerNone")
    model.Resource.wind_resource_filename = wind_resource.filename
    model.execute(0)
    aep = model.Outputs.annual_energy
    assert(aep > 70e6)
    model = wp.default("WindPowerNone")
    model.Resource.wind_resource_data = wind_resource.data
    model.execute(0)
    assert(model.Outputs.annual_energy == approx(aep))

def test_windtoolkit_download(wind_resource):
    assert(wind_resource.download_resource())

@pytest.fixture
def solar_resource_nasa():
    return SolarResource(lat=lat, lon=lon, year=year, api='nasa')


@pytest.fixture
def wind_resource_nasa():
    return WindResource(lat=lat, lon=lon, year=year, api='nasa', nasa_vegtype='vegtype_8', wind_turbine_hub_ht=hubheight)

def test_solar_nasa(solar_resource_nasa):
    data = solar_resource_nasa.data
    for key in ('df', 'dn', 'wspd', 'tdry', 'year', 'month', 'day', 'hour', 'minute', 'tz'):
        assert(key in data)
    ### NASA provides leap year 2/29 data in file, we should always use post processed solar_resource_nasa.data not .filename for NASA solar
    # annual energy output in line 86 is different than line 90 because it includes 2/29 data
    model = pv.default("PVWattsNone")
    model.SolarResource.solar_resource_file = solar_resource_nasa.filename
    model.execute(0)
    assert(model.Outputs.annual_energy == approx(10373.6, 0.1))
    model = pv.default("PVWattsNone")
    model.SolarResource.solar_resource_data = solar_resource_nasa.data
    model.execute(1)
    assert(model.Outputs.annual_energy == approx(10362.1, 0.1))


def test_nasa_solar_download(solar_resource_nasa):
    assert solar_resource_nasa.download_resource()


def test_wind_nasa(wind_resource_nasa):
    data = wind_resource_nasa.data
    for key in ('heights', 'fields', 'data'):
        assert (key in data)

    assert (data['fields'] == [1, 2, 3, 4])
    assert (data['heights'] == [80.0, 80.0, 80.0, 80.0])
    assert (data['data'][0] == [-3.205, 1.008, 9.499, 304.202])
    ### NASA provides leap year data and non formated data, we should alway use post processed wind_resource_nasa.data not .filename for NASA Wind
    # No testing for wind.filename because we must use post processed data for NASA POWER wind
    model = wp.default("WindPowerNone")
    model.Resource.wind_resource_data = wind_resource_nasa.data
    model.execute(0)
    assert(model.Outputs.annual_energy == approx(138e6,1e5))


def test_nasa_wind_download(wind_resource_nasa):
    assert(wind_resource_nasa.download_resource())


def test_wind_combine():
    path_file = os.path.dirname(os.path.abspath(__file__))

    kwargs = {'path_resource': os.path.join(path_file, 'data')}

    wind_resource = WindResource(lat=lat, lon=lon, year=year, wind_turbine_hub_ht=70, **kwargs)

    if os.path.isfile(wind_resource.filename):
        os.remove(wind_resource.filename)

    assert(wind_resource.combine_wind_files())


def test_from_file():
    windfile = Path(__file__).parent.parent.parent / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m.srw"
    wind_resource = WindResource(lat=lat, lon=lon, year=year, api='nrel', wind_turbine_hub_ht=70, filepath=windfile)
    assert(len(wind_resource.data['data']) > 0)

    solarfile = Path(__file__).parent.parent.parent / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
    solar_resource = SolarResource(lat=lat, lon=lon, year=year, api='nrel', filepath=solarfile)
    assert(len(solar_resource.data['gh']) > 0)

    windfile = Path(__file__).parent.parent.parent / "resource_files" / "wind" / "35.2018863_-101.945027_nasa_2012_60min_80m.srw"
    wind_resource_nasa = WindResource(lat=lat, lon=lon, year=year, api='nasa', wind_turbine_hub_ht=70, filepath=windfile)
    assert(len(wind_resource_nasa.data['data']) > 0)

    solarfile = Path(__file__).parent.parent.parent / "resource_files" / "solar" / "35.2018863_-101.945027_nasa_60_2012.csv"
    solar_resource_nasa = SolarResource(lat=lat, lon=lon, year=year, api='nasa', filepath=solarfile)
    assert(len(solar_resource_nasa.data['gh']) > 0)
