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
    return SolarResource(lat=lat, lon=lon, year=year)


@pytest.fixture
def wind_resource():
    return WindResource(lat=lat, lon=lon, year=year, wind_turbine_hub_ht=hubheight)


def test_solar(solar_resource):
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


def test_nsrdb(solar_resource):
    solar_resource.download_resource()


def test_wind(wind_resource):
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


def test_wind_toolkit(wind_resource):
    assert(wind_resource.download_resource())


def test_wind_combine():
    path_file = os.path.dirname(os.path.abspath(__file__))

    kwargs = {'path_resource': os.path.join(path_file, 'data')}

    wind_resource = WindResource(lat=lat, lon=lon, year=year, wind_turbine_hub_ht=70, **kwargs)

    if os.path.isfile(wind_resource.filename):
        os.remove(wind_resource.filename)

    assert(wind_resource.combine_wind_files())


def test_from_file():
    windfile = Path(__file__).parent.parent.parent / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m.srw"
    wind_resource = WindResource(lat=lat, lon=lon, year=year, wind_turbine_hub_ht=70, filepath=windfile)
    assert(len(wind_resource.data['data']) > 0)

    solarfile = Path(__file__).parent.parent.parent / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
    solar_resource = SolarResource(lat=lat, lon=lon, year=year, filepath=solarfile)
    assert(len(solar_resource.data['gh']) > 0)
