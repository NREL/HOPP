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