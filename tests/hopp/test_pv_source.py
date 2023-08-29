from pytest import fixture
import pytest

from numpy.testing import assert_array_equal
import PySAM.Pvwattsv8 as Pvwatts
import PySAM.Singleowner as Singleowner

from hopp import ROOT_DIR
from hopp.simulation.technologies.sites import SiteInfo, flatirons_site
from hopp.simulation.technologies.pv_source import PVConfig, PVPlant
from hopp.simulation.technologies.layout.pv_layout import PVLayout

solar_resource_file = ROOT_DIR.parent / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"


@fixture
def site():
    return SiteInfo(flatirons_site, solar_resource_file=solar_resource_file, wind=False)


@fixture
def sample_pv_config():
    return PVConfig(system_capacity_kw=100.0)


def test_pv_config_initialization():
    pv_config = PVConfig(system_capacity_kw=100.0)
    assert pv_config.system_capacity_kw == 100.0
    assert pv_config.layout_params is None
    assert pv_config.layout_model is None
    assert pv_config.fin_model is None


def test_pv_plant_init(site):
    pv_plant = PVPlant(site=site, config={'system_capacity_kw': 100.0})
    assert isinstance(pv_plant.pv_config, PVConfig)
    assert isinstance(pv_plant.system_model, Pvwatts.Pvwattsv8)
    assert isinstance(pv_plant.financial_model, Singleowner.Singleowner)
    assert isinstance(pv_plant.layout, PVLayout)
    assert pv_plant.config_name == "PVWattsSingleOwner"


def test_pv_plant_bad_config(site, subtests):
    with subtests.test("No system capacity"):
        with pytest.raises(TypeError):
            PVPlant(site=site, config={})


def test_pv_plant_system_capacity(site, subtests):
    pv_plant = PVPlant(site=site, config={'system_capacity_kw': 100.0})

    with subtests.test("getter"):
        assert pv_plant.system_capacity_kw == 100.0

    with subtests.test("setter"):
        pv_plant.system_capacity_kw = 200.0
        assert pv_plant.system_capacity_kw == 200.0


def test_pv_plant_dc_degradation(site, subtests):
    pv_plant = PVPlant(site=site, config={'system_capacity_kw': 100.0})

    with subtests.test("getter"):
        assert_array_equal(pv_plant.dc_degradation, [0])

    with subtests.test("setter"):
        pv_plant.dc_degradation = [1]
        assert_array_equal(pv_plant.dc_degradation, [1])
