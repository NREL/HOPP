import pytest
from pytest import approx
import os

from hybrid.resource import SolarResource, WindResource

import PySAM.Windpower as wp
import PySAM.Pvwattsv5 as pv

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
    model.LocationAndResource.solar_resource_file = solar_resource.filename
    model.execute(0)
    assert(model.Outputs.annual_energy == approx(5771.589))
    model = pv.default("PVWattsNone")
    model.LocationAndResource.solar_resource_data = solar_resource.data
    model.execute(1)
    assert(model.Outputs.annual_energy == approx(5771.803))


def test_nsrdb(solar_resource):
    assert(solar_resource.download_solar_resource(force_download=True))
    assert(solar_resource.download_solar_resource(force_download=False))


def test_wind(wind_resource):
    data = wind_resource.data
    for key in ('heights', 'fields', 'data'):
        assert (key in data)
    model = wp.default("WindPowerNone")
    model.Resource.wind_resource_filename = wind_resource.filename
    model.execute(0)
    assert(model.Outputs.annual_energy == approx(85049139.587))
    model = wp.default("WindPowerNone")
    model.Resource.wind_resource_data = wind_resource.data
    model.execute(0)
    assert(model.Outputs.annual_energy == approx(85049139.587))


def test_wind_toolkit(wind_resource):
    assert(wind_resource.download_resource(force_download=True))
    assert(wind_resource.download_resource(force_download=False))


def test_wind_combine():
    path_file = os.path.dirname(os.path.abspath(__file__))

    kwargs = {'path_resource': os.path.join(path_file, 'data')}

    wind_resource = WindResource(lat=lat, lon=lon, year=year, wind_turbine_hub_ht=70, **kwargs)

    if os.path.isfile(wind_resource.filename):
        os.remove(wind_resource.filename)

    assert(wind_resource.combine_wind_files())
