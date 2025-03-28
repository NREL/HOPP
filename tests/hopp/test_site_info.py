import os
import copy
from pathlib import Path

import pytest
from pytest import fixture, approx
from shapely.geometry import Polygon
import numpy as np
from numpy.testing import assert_array_equal

from hopp.simulation.technologies.sites import SiteInfo, flatirons_site
from hopp import ROOT_DIR

from PySAM.ResourceTools import SRW_to_wind_data, SAM_CSV_to_solar_data

solar_resource_file = os.path.join(
    ROOT_DIR, "simulation", "resource_files", "solar", 
    "35.2018863_-101.945027_psmv3_60_2012.csv"
)
wind_resource_file = os.path.join(
    ROOT_DIR, "simulation", "resource_files", "wind", 
    "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
)
grid_resource_file = os.path.join(
    ROOT_DIR, "simulation", "resource_files", "grid", 
    "pricing-data-2015-IronMtn-002_factors.csv"
)
kml_filepath = Path(__file__).absolute().parent / "layout_example.kml"
from hopp.simulation.technologies.resource import AlaskaWindData

@fixture
def site():
    return SiteInfo(
        flatirons_site,
        solar_resource_file=solar_resource_file,
        wind_resource_file=wind_resource_file,
        grid_resource_file=grid_resource_file
    )


def test_site_init(site):
    """Site should initialize properly."""
    assert site is not None

    # data
    assert site.lat == flatirons_site["lat"]
    assert site.lon == flatirons_site["lon"]
    assert site.year == flatirons_site["year"]
    assert site.tz == flatirons_site["tz"]
    assert site.urdb_label == flatirons_site["urdb_label"]

    # resources
    assert site.solar_resource is not None
    assert site.wind_resource is not None
    assert site.elec_prices is not None

    # time periods
    assert site.n_timesteps == 8760
    assert site.n_periods_per_day == 24
    assert site.interval == 60
    assert_array_equal(site.capacity_hours, [False] * site.n_timesteps)
    assert_array_equal(site.desired_schedule, [])

    # polygon
    assert site.polygon is not None
    assert site.vertices is not None

    # unset
    assert site.kml_data is None


def test_site_init_kml_read():
    """Should initialize via kml file."""
    site = SiteInfo({"kml_file": kml_filepath}, solar_resource_file=solar_resource_file, wind_resource_file=wind_resource_file)

    assert site.kml_data is not None
    assert site.polygon is not None


def test_site_init_missing_coords():
    """Should fail if lat/lon missing."""
    data = copy.deepcopy(flatirons_site)
    del data["lat"]
    del data["lon"]

    with pytest.raises(ValueError):
        SiteInfo(data)

    data["lat"] = flatirons_site["lat"]

    # should still fail because lon is missing
    with pytest.raises(ValueError):
        SiteInfo(data)


def test_site_init_improper_schedule():
    """Should fail if the desired schedule mismatches the number of timesteps."""
    data = copy.deepcopy(flatirons_site)

    with pytest.raises(ValueError):
        SiteInfo(
            data, 
            solar_resource_file=solar_resource_file,
            wind_resource_file=wind_resource_file,
            grid_resource_file=grid_resource_file,
            desired_schedule=np.array([1])
        )


def test_site_init_no_wind():
    """Should initialize without pulling wind data."""
    data = copy.deepcopy(flatirons_site)

    site = SiteInfo(
        data, 
        solar_resource_file=solar_resource_file,
        wind_resource_file=wind_resource_file,
        grid_resource_file=grid_resource_file,
        wind=False
    )

    assert site.wind_resource is None

    
def test_site_init_no_solar():
    """Should initialize without pulling wind data."""
    data = copy.deepcopy(flatirons_site)

    site = SiteInfo(
        data, 
        solar_resource_file=solar_resource_file,
        wind_resource_file=wind_resource_file,
        grid_resource_file=grid_resource_file,
        solar=False
    )

    assert site.solar_resource is None


def test_site_kml_file_read():
    site_data = {'kml_file': kml_filepath}
    site = SiteInfo(site_data, solar_resource_file=solar_resource_file, wind_resource_file=wind_resource_file)
    assert np.array_equal(np.round(site.polygon.bounds), [ 681175., 4944970.,  686386., 4949064.])
    assert site.polygon.area * 3.86102e-7 == pytest.approx(2.3393, abs=0.01) # m2 to mi2


def test_site_kml_file_append():
    site_data = {'kml_file': kml_filepath}
    site = SiteInfo(site_data, solar_resource_file=solar_resource_file, wind_resource_file=wind_resource_file)

    x = site.polygon.centroid.x
    y = site.polygon.centroid.y
    turb_coords = [x - 500, y - 500]
    solar_region = Polygon(((x, y), (x, y + 5000), (x + 5000, y), (x + 5000, y + 5000)))

    filepath_new = Path(__file__).absolute().parent / "layout_example2.kml"
    site.kml_write(filepath_new, turb_coords, solar_region)
    assert filepath_new.exists()
    k, valid_region, lat, lon = SiteInfo.kml_read(kml_filepath)
    assert valid_region.area > 0
    os.remove(filepath_new)

def test_site_wind_resource_input_filename():
    data = copy.deepcopy(flatirons_site)
    wind_resource_data_dict = SRW_to_wind_data(wind_resource_file)
    site = SiteInfo(
        data, 
        hub_height = 90,
        wind = True,
        solar = False,
        wind_resource = wind_resource_data_dict
    )
    assert site.wind_resource.filename is None

def test_site_wind_resource_input_data_length():
    data = copy.deepcopy(flatirons_site)
    wind_resource_data_dict = SRW_to_wind_data(wind_resource_file)
    site = SiteInfo(
        data, 
        hub_height = 90,
        wind = True,
        solar = False,
        wind_resource = wind_resource_data_dict
    )
    assert len(site.wind_resource.data['data'])==8760

def test_site_wind_resource_input_data_format():
    data = copy.deepcopy(flatirons_site)
    wind_resource_data_dict = SRW_to_wind_data(wind_resource_file)
    site = SiteInfo(
        data, 
        hub_height = 90,
        wind = True,
        solar = False,
        wind_resource = wind_resource_data_dict
    )
    assert int(site.wind_resource.data['heights'][0])==80

def test_site_solar_resource_input_filename():
    data = copy.deepcopy(flatirons_site)
    solar_resource_data_dict = SAM_CSV_to_solar_data(solar_resource_file)
    site = SiteInfo(
        data, 
        wind = False,
        solar = True,
        solar_resource = solar_resource_data_dict
    )
    assert site.solar_resource.filename is None

def test_site_solar_resource_input_data_length():
    data = copy.deepcopy(flatirons_site)
    solar_resource_data_dict = SAM_CSV_to_solar_data(solar_resource_file)
    site = SiteInfo(
        data, 
        wind = False,
        solar = True,
        solar_resource = solar_resource_data_dict
    )
    assert len(site.solar_resource.data['dn'])==8760

def test_site_solar_resource_input_data_format():
    data = copy.deepcopy(flatirons_site)
    solar_resource_data_dict = SAM_CSV_to_solar_data(solar_resource_file)
    site = SiteInfo(
        data, 
        wind = False,
        solar = True,
        solar_resource = solar_resource_data_dict
    )
    assert site.solar_resource.data['tz']==-6

def test_site_polygon_valid_verts():
    site_data = {
        "lat": 35.2018863,
        "lon": -101.945027,
        "elev": 1099,
        "year": 2012,
        "tz": -6,
        "site_boundaries":
            {
            "verts":
            [[0.0,0.0],[500.0,0.0],[500.0,500.0],[0.0,500.0]]
            }
    }
    site =  SiteInfo(
        site_data,
        solar_resource_file=solar_resource_file,
        wind_resource_file=wind_resource_file,
        grid_resource_file=grid_resource_file
    )
    
    assert site.polygon.area == approx(250000,rel = 1e-3)
    assert site.vertices[0][0] == 0.0
    assert site.vertices[0][1] == 0.0
    assert site.vertices[1][0] == 500.0
    assert site.vertices[1][1] == 0.0
    

def test_site_polygon_invalid_verts():
    site_data = {
        "lat": 35.2018863,
        "lon": -101.945027,
        "elev": 1099,
        "year": 2012,
        "tz": -6,
        "site_boundaries":
            {
            "verts":
            [[0.0,0.0],[500.0,500.0],[500.0,0.0],[0.0,500.0]]
            }
    }
    site =  SiteInfo(
        site_data,
        solar_resource_file=solar_resource_file,
        wind_resource_file=wind_resource_file,
        grid_resource_file=grid_resource_file
    )
    assert site.polygon.area != approx(250000,rel = 1e-3)

def test_site_polygon_square_defaults():
    site_area_km2 = 2.5
    site_data = {
        "lat": 35.2018863,
        "lon": -101.945027,
        "elev": 1099,
        "year": 2012,
        "tz": -6,
        "site_details":
            {
            "site_area_km2": site_area_km2,
            "site_shape":"square",
            }
    }
    site =  SiteInfo(
        site_data,
        solar_resource_file=solar_resource_file,
        wind_resource_file=wind_resource_file,
        grid_resource_file=grid_resource_file
    )
    assert site.polygon.area == approx(site_area_km2*1e6,rel=1e-3)


def test_site_polygon_square_offset():
    site_area_km2 = 2.5
    x0 = 25.0
    site_data = {
        "lat": 35.2018863,
        "lon": -101.945027,
        "elev": 1099,
        "year": 2012,
        "tz": -6,
        "site_details":
            {
            "site_area_km2": site_area_km2,
            "site_shape":"Square",
            "x0": x0,
            }
    }
    site =  SiteInfo(
        site_data,
        solar_resource_file=solar_resource_file,
        wind_resource_file=wind_resource_file,
        grid_resource_file=grid_resource_file
    )
    x_verts,y_verts = site.polygon.exterior.coords.xy
    assert site.polygon.area == approx(site_area_km2*1e6,rel=1e-3)
    assert min(x_verts) == approx(x0,rel=1e-3)
    assert min(y_verts) == approx(0.0,abs=1e-8)
    

def test_site_polygon_rectangle_default():
    site_area_km2 = 2.5
    site_data = {
        "lat": 35.2018863,
        "lon": -101.945027,
        "elev": 1099,
        "year": 2012,
        "tz": -6,
        "site_details":
            {
            "site_area_km2": site_area_km2,
            "site_shape":"rectangle",
            }
    }
    site =  SiteInfo(
        site_data,
        solar_resource_file=solar_resource_file,
        wind_resource_file=wind_resource_file,
        grid_resource_file=grid_resource_file
    )
    x_verts,y_verts = site.polygon.exterior.coords.xy
    dx = max(x_verts) - min(x_verts)
    dy = max(y_verts) - min(y_verts)
    width_to_height = dx/dy

    assert site.polygon.area == approx(site_area_km2*1e6,rel=1e-3)
    assert width_to_height == approx(1.5,rel=1e-3)


def test_site_polygon_rectangle_aspect_ratio():
    site_area_km2 = 2.5
    aspect_ratio = 2.0
    site_data = {
        "lat": 35.2018863,
        "lon": -101.945027,
        "elev": 1099,
        "year": 2012,
        "tz": -6,
        "site_details":
            {
            "site_area_km2": site_area_km2,
            "site_shape":"rectangle",
            "aspect_ratio": aspect_ratio,
            }
    }
    site =  SiteInfo(
        site_data,
        solar_resource_file=solar_resource_file,
        wind_resource_file=wind_resource_file,
        grid_resource_file=grid_resource_file
    )
    x_verts,y_verts = site.polygon.exterior.coords.xy
    dx = max(x_verts) - min(x_verts)
    dy = max(y_verts) - min(y_verts)
    width_to_height = dx/dy

    assert site.polygon.area == approx(site_area_km2*1e6,rel=1e-3)
    assert width_to_height == approx(aspect_ratio,rel=1e-3)
    

def test_site_polygon_circle_default():
    site_area_km2 = 2.5
    site_data = {
        "lat": 35.2018863,
        "lon": -101.945027,
        "elev": 1099,
        "year": 2012,
        "tz": -6,
        "site_details":
            {
            "site_area_km2": site_area_km2,
            "site_shape":"circle",
            }
    }
    site =  SiteInfo(
        site_data,
        solar_resource_file=solar_resource_file,
        wind_resource_file=wind_resource_file,
        grid_resource_file=grid_resource_file
    )

    assert site.polygon.area == approx(site_area_km2*1e6,rel=1e-2)
    assert len(site.vertices) == 36


def test_site_polygon_circle_detail():
    site_area_km2 = 2.5
    site_data = {
        "lat": 35.2018863,
        "lon": -101.945027,
        "elev": 1099,
        "year": 2012,
        "tz": -6,
        "site_details":
            {
            "site_area_km2": site_area_km2,
            "site_shape":"circle",
            "degrees_between_points":1.0
            }
    }
    site =  SiteInfo(
        site_data,
        solar_resource_file=solar_resource_file,
        wind_resource_file=wind_resource_file,
        grid_resource_file=grid_resource_file
    )

    assert site.polygon.area == approx(site_area_km2*1e6,rel=1e-3)
    assert len(site.vertices) == 360


def test_site_polygon_hexagon_default():
    site_area_km2 = 2.5
    site_data = {
        "lat": 35.2018863,
        "lon": -101.945027,
        "elev": 1099,
        "year": 2012,
        "tz": -6,
        "site_details":
            {
            "site_area_km2": site_area_km2,
            "site_shape":"hexagon",
            }
    }
    site =  SiteInfo(
        site_data,
        solar_resource_file=solar_resource_file,
        wind_resource_file=wind_resource_file,
        grid_resource_file=grid_resource_file
    )

    assert site.polygon.area == approx(site_area_km2*1e6,rel=1e-3)


def test_site_polygon_hexagon_m2():
    site_area_km2 = 2.5
    site_area_m2 = site_area_km2*1e6
    site_data = {
        "lat": 35.2018863,
        "lon": -101.945027,
        "elev": 1099,
        "year": 2012,
        "tz": -6,
        "site_details":
            {
            "site_area_m2": site_area_m2,
            "site_shape":"hexagon",
            }
    }
    site =  SiteInfo(
        site_data,
        solar_resource_file=solar_resource_file,
        wind_resource_file=wind_resource_file,
        grid_resource_file=grid_resource_file
    )

    assert site.polygon.area == approx(site_area_km2*1e6,rel=1e-3)


def test_site_invalid_shape():
    site_area_km2 = 2.5
    site_data = {
        "lat": 35.2018863,
        "lon": -101.945027,
        "elev": 1099,
        "year": 2012,
        "tz": -6,
        "site_details":
            {
            "site_area_km2": site_area_km2,
            "site_shape":"triangle",
            }
    }
    with pytest.raises(ValueError) as err:
        site =  SiteInfo(
            site_data,
            solar_resource_file=solar_resource_file,
            wind_resource_file=wind_resource_file,
            grid_resource_file=grid_resource_file
        )
    assert str(err.value) == "invalid entry for `site_shape`, site_shape must be either 'circle', 'rectangle', 'square' or 'hexagon'"
    

def test_site_none_shape():
    site_area_km2 = 2.5
    site_data = {
        "lat": 35.2018863,
        "lon": -101.945027,
        "elev": 1099,
        "year": 2012,
        "tz": -6,
        "site_details": {}
    }
    site =  SiteInfo(
        site_data,
        solar_resource_file=solar_resource_file,
        wind_resource_file=wind_resource_file,
        grid_resource_file=grid_resource_file
    )
    assert site.polygon is None

def test_alaska_wind_resource():
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
    assert isinstance(site.wind_resource,AlaskaWindData)
