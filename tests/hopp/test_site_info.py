import os
import copy
from pathlib import Path

import pytest
from pytest import fixture
from shapely.geometry import Polygon
import numpy as np
from numpy.testing import assert_array_equal

from hopp.simulation.technologies.sites import SiteInfo, flatirons_site
from hopp import ROOT_DIR

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