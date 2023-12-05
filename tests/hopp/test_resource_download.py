from pytest import approx, fixture
import responses
import os

from hopp import ROOT_DIR
from hopp.simulation.technologies.resource.solar_resource import BASE_URL as SOLAR_URL
from hopp.simulation.technologies.resource.wind_resource import WTK_BASE_URL, TAP_BASE_URL
from hopp.simulation.technologies.resource import SolarResource, WindResource, Resource
from hopp.utilities.utils_for_tests import DEFAULT_WIND_RESOURCE_FILE

import PySAM.Windpower as wp
import PySAM.Pvwattsv8 as pv

dir_path = os.path.dirname(os.path.realpath(__file__))

year = 2012
lat = 39.7555
lon = -105.2211
hubheight = 80

with open(DEFAULT_WIND_RESOURCE_FILE, 'r') as f:
    wind_body = f.read()

solar_file = ROOT_DIR.parent / "resource_files" / "solar" / "39.7555_-105.2211_psmv3_60_2012.csv"

with open(solar_file, 'r') as f:
    solar_body = f.read()

@responses.activate
def test_solar():
    resp = responses.add(
        responses.GET,
        SOLAR_URL,
        body=solar_body
    )
    solar_resource = SolarResource(
        lat=lat, 
        lon=lon, 
        year=year,
        use_api=True
    )
    data = solar_resource.data
    for key in ('df', 'dn', 'wspd', 'tdry', 'year', 'month', 'day', 'hour', 'minute', 'tz'):
        assert(key in data)
    model = pv.default("PVWattsNone")
    model.SolarResource.solar_resource_file = solar_resource.filename
    model.execute(0)
    assert(model.Outputs.annual_energy == approx(143852209, 0.1))
    model = pv.default("PVWattsNone")
    model.SolarResource.solar_resource_data = solar_resource.data
    model.execute(1)
    assert(model.Outputs.annual_energy == approx(143852209, 0.1))
    assert resp.call_count == 1


@responses.activate
def test_wind():
    resp = responses.add(
        responses.GET,
        WTK_BASE_URL,
        body=wind_body
    )
    wind_resource = WindResource(
        lat=lat, 
        lon=lon, 
        year=year, 
        wind_turbine_hub_ht=hubheight, 
        use_api=True
    )
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
    assert resp.call_count == 1


@responses.activate
def test_wind_toolkit():
    resp = responses.add(
        responses.GET,
        WTK_BASE_URL,
        body=wind_body
    )
    WindResource(
        lat=lat, 
        lon=lon, 
        year=year, 
        wind_turbine_hub_ht=hubheight,
        use_api=True
    )
    assert resp.call_count == 1


@responses.activate
def test_tap():
    resp = responses.add(
        responses.GET,
        TAP_BASE_URL,
        body=wind_body
    )
    WindResource(
        lat=lat, 
        lon=lon, 
        year=year, 
        wind_turbine_hub_ht=hubheight,
        source="TAP",
        use_api=True
    )
    assert resp.call_count == 1


@responses.activate
def test_wind_combine():
    resp = responses.add(
        responses.GET,
        WTK_BASE_URL,
        wind_body
    )

    wind_resource = WindResource(
        lat=lat, 
        lon=lon, 
        year=year, 
        wind_turbine_hub_ht=70, 
        use_api=True,
    )

    assert(wind_resource.combine_wind_files())
    assert resp.call_count == 2


def test_from_file():
    wind_resource = WindResource(
        lat=lat, 
        lon=lon, 
        year=year, 
        wind_turbine_hub_ht=70, 
        filepath=str(DEFAULT_WIND_RESOURCE_FILE),
    )
    assert(len(wind_resource.data['data']) > 0)

    solar_resource = SolarResource(
        lat=lat, 
        lon=lon, 
        year=year, 
        filepath=str(solar_file)
    )
    assert(len(solar_resource.data['gh']) > 0)
