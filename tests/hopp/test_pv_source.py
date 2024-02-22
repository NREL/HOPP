import pytest
from pytest import fixture

from numpy.testing import assert_array_equal

from hopp.simulation.technologies.pv.pv_plant import PVConfig, PVPlant
from tests.hopp.utils import create_default_site_info


@fixture
def site():
    return create_default_site_info()


@fixture
def sample_pv_config():
    return PVConfig(system_capacity_kw=100.0)


def test_pv_config_initialization(subtests):
    pv_config = PVConfig(system_capacity_kw=100.0)
    assert pv_config.system_capacity_kw == 100.0
    assert pv_config.layout_params is None
    assert pv_config.layout_model is None
    assert pv_config.fin_model is None

    with subtests.test("with invalid system_capacity_kw"):
        with pytest.raises(ValueError):
            PVConfig(system_capacity_kw=0.0)


def test_pv_plant_initialization(site, subtests):
    system_capacity_kw = 100.0
    config_data = {'system_capacity_kw': system_capacity_kw}
    config = PVConfig.from_dict(config_data)

    pv_plant = PVPlant(site=site, config=config)

    assert pv_plant.config_name == "PVWattsSingleOwner"
    assert pv_plant.name == "PVPlant"
    assert pv_plant.system_capacity_kw == system_capacity_kw
    assert_array_equal(pv_plant.dc_degradation, [0])

def test_module_type(site, subtests):
    system_capacity_kw = 100.0
    config_data = {'system_capacity_kw': system_capacity_kw}
    config = PVConfig.from_dict(config_data)

    pv_plant = PVPlant(site=site, config=config)

    with subtests.test("initial module type"):
        assert pv_plant.module_type == 0
        assert pv_plant.approx_nominal_efficiency == 0.19

    with subtests.test("change module type"):
        pv_plant.module_type = 2
        assert pv_plant.approx_nominal_efficiency == 0.18

    with subtests.test("module type not found"):
        with pytest.raises(Exception):
            pv_plant.module_type = 3

def test_pv_plant_area(site, subtests):
    system_capacity_kw = 100.0
    config_data = {'system_capacity_kw': system_capacity_kw}
    config = PVConfig.from_dict(config_data)

    pv_plant = PVPlant(site=site, config=config)
    
    with subtests.test("plant footprint area"):
        assert pv_plant.footprint_area == pytest.approx(457.94, 0.1)

def test_pv_plant_mass(site, subtests):
    system_capacity_kw = 100.0
    config_data = {'system_capacity_kw': system_capacity_kw}
    config = PVConfig.from_dict(config_data)

    pv_plant = PVPlant(site=site, config=config)

    with subtests.test("plant mass"):
        assert pv_plant.system_mass == pytest.approx(5079.51,0.01)