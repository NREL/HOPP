from pytest import fixture

from numpy.testing import assert_array_equal
import PySAM.Pvwattsv8 as Pvwatts
import PySAM.Singleowner as Singleowner

from hopp.simulation.technologies.pv_source import PVConfig, PVPlant
from hopp.simulation.technologies.layout.pv_layout import PVLayout
from tests.hopp.utils import create_default_site_info


@fixture
def site():
    return create_default_site_info()


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