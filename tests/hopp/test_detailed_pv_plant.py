from pathlib import Path

import json
import pytest
from pytest import fixture

from hopp.simulation.technologies.pv.detailed_pv_plant import (
    DetailedPVConfig,
    DetailedPVPlant,
)
from hopp.simulation.technologies.financial.custom_financial_model import (
    CustomFinancialModel,
)
from hopp.simulation.technologies.layout.pv_layout import PVGridParameters
from tests.hopp.utils import create_default_site_info, DEFAULT_FIN_CONFIG

config_data = {"system_capacity_kw": 100, "tech_config": {"subarray2_enable": 0}}


@fixture
def site():
    return create_default_site_info()


def test_detailed_pv_plant_initialization(site, subtests):
    """Test simple instantiation (no layout params)."""
    config = DetailedPVConfig.from_dict(config_data)
    pv_plant = DetailedPVPlant(site=site, config=config)
    with subtests.test("Site: lat and lon"):
        assert pv_plant.site.lat == site.lat
        assert pv_plant.site.lon == site.lon
    with subtests.test("Site: elev"):
        assert pv_plant.site.data["elev"] == site.data["elev"]
    with subtests.test("Site: year"):
        assert pv_plant.site.data["year"] == site.data["year"]
    with subtests.test("Site: tz"):
        assert pv_plant.site.data["tz"] == site.data["tz"]
    with subtests.test("Site: site boundaries"):
        assert (
            pv_plant.site.data["site_boundaries"]["verts"]
            == site.data["site_boundaries"]["verts"]
        )
    with subtests.test("Site: site boundaries simple"):
        assert (
            pv_plant.site.data["site_boundaries"]["verts_simple"]
            == site.data["site_boundaries"]["verts_simple"]
        )
    assert pv_plant._financial_model is not None
    assert pv_plant.layout is not None
    assert pv_plant.layout.parameters is None
    assert pv_plant.config is not None


def test_single_subarray_limitation(site):
    """Ensure only one subarray is allowed."""
    config_with_multiple_subarrays = {
        "system_capacity_kw": 100,
        "tech_config": {"subarray2_enable": 1},
    }
    config = DetailedPVConfig.from_dict(config_with_multiple_subarrays)
    with pytest.raises(
        Exception, match=r"Detailed PV plant currently only supports one subarray."
    ):
        DetailedPVPlant(site=site, config=config)


def test_processed_assign(site, subtests):
    """Test more detailed instantiation with `tech_config`."""
    pvsamv1_defaults_file = (
        Path(__file__).absolute().parent / "pvsamv1_basic_params.json"
    )
    with open(pvsamv1_defaults_file, "r") as f:
        tech_config = json.load(f)

    with subtests.test("With Pvsamv1 configuration file"):
        config = DetailedPVConfig.from_dict({"tech_config": tech_config})
        pv_plant = DetailedPVPlant(site=site, config=config)
        assert pv_plant.system_capacity_kw is not None


def test_layout_parameters(site):
    """Ensure layout parameters are set properly if provided."""
    config_with_layout_params = {
        "system_capacity_kw": 100,
        "layout_params": PVGridParameters(
            x_position=0.5,
            y_position=0.5,
            aspect_power=0,
            gcr=0.5,
            s_buffer=2,
            x_buffer=2,
        ),
    }
    config = DetailedPVConfig.from_dict(config_with_layout_params)
    pv_plant = DetailedPVPlant(site=site, config=config)
    assert pv_plant.layout.parameters == config_with_layout_params["layout_params"]


def test_custom_financial(site):
    """Test with a non-default financial model."""
    config = DetailedPVConfig.from_dict(
        {
            "system_capacity_kw": 100,
            "fin_model": CustomFinancialModel(DEFAULT_FIN_CONFIG, name="Test"),
        }
    )
    pv_plant = DetailedPVPlant(site=site, config=config)
    assert pv_plant._financial_model is not None
    assert isinstance(pv_plant._financial_model, CustomFinancialModel)
