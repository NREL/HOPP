import pytest
from pytest import fixture
from pathlib import Path

from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.technologies.financial.mhk_cost_model import MHKCostModelInputs
from hopp.simulation.technologies.financial.custom_financial_model import CustomFinancialModel
from hopp.utilities import load_yaml
from hopp import ROOT_DIR
from tests.hopp.utils import DEFAULT_FIN_CONFIG
from hopp.simulation.technologies.tidal.mhk_tidal_plant import MHKTidalPlant, MHKTidalConfig

@fixture
def site():
    data = {
        "lat": 44.6899,
        "lon": 124.1346,
        "year": 2010,
        "tz": -7,
    }
    tidal_resource_file = Path.joinpath(ROOT_DIR / "simulation" / "resource_files" / "tidal" / "Tidal_resource_timeseries.csv")
    site = SiteInfo(data, solar=False, wind=False, tidal=True, tidal_resource_file=tidal_resource_file)
	
    return site

@fixture
def mhk_config():
    mhk_yaml_path = Path(__file__).absolute().parent.parent.parent / "tests" / "hopp" / "inputs" / "tidal" / "tidal_device.yaml"
    mhk_config = load_yaml(mhk_yaml_path)

    return mhk_config

@fixture
def tidalplant(mhk_config, site):
    financial_model = {'fin_model': DEFAULT_FIN_CONFIG}
    mhk_config.update(financial_model)
    config = MHKTidalConfig.from_dict(mhk_config)
    
    cost_model_input = MHKCostModelInputs.from_dict({
        'reference_model_num':1,
        'water_depth': 100,
        'distance_to_shore': 80,
        'number_rows': 2,
        'device_spacing':600,
        'row_spacing': 600,
        'cable_system_overbuild': 20
    })
    return MHKTidalPlant(site, config, cost_model_input)

def test_mhk_config(mhk_config, subtests):
    with subtests.test("with basic params"):
        financial_model = {'fin_model': DEFAULT_FIN_CONFIG}
        mhk_config.update(financial_model)

        config = MHKTidalConfig.from_dict(mhk_config)

        assert config.device_rating_kw == 1115.
        assert config.num_devices == 20
        assert config.fin_model is not None
        # defaults
        assert config.loss_array_spacing == 0.
        assert config.loss_resource_overprediction == 0.
        assert config.loss_transmission == 0.
        assert config.loss_downtime == 0.
        assert config.loss_additional == 0.
        
def test_system_outputs(tidalplant,subtests):
    tidalplant.simulate(25)

    with subtests.test("annual energy kwh"):
        assert tidalplant.annual_energy_kwh == pytest.approx(60625516, 1e-3)

    with subtests.test("capacity factor"):
        assert tidalplant.capacity_factor == pytest.approx(31.03, 1e-3)
        
def test_cost_outputs(tidalplant,subtests):
    tidalplant.simulate(25)
    with subtests.test("structural assembly cost"):
        assert tidalplant.mhk_costs.cost_outputs['structural_assembly_cost_modeled'] == pytest.approx(8495602, 1e-3)
    with subtests.test("power_takeoff_system_cost"):
        assert tidalplant.mhk_costs.cost_outputs['power_takeoff_system_cost_modeled']== pytest.approx(41212670, 1e-3)

def test_changing_n_devices(tidalplant, subtests):
    with subtests.test("less devices than rows"):
        with pytest.raises(Exception):
            tidalplant.number_devices = 9

    with subtests.test("not grid shape"):
        with pytest.raises(Exception):
            tidalplant.number_devices = 11
    
    tidalplant.number_devices = 50
    with subtests.test("change system capacity"):
        assert tidalplant.system_capacity_kw == pytest.approx(1115*50,0)

    with subtests.test("update cost model - number_devices"):
        assert tidalplant.mhk_costs.number_devices == tidalplant.number_devices

    with subtests.test("update cost model - system_capacity"):
        assert tidalplant.mhk_costs.system_capacity_kw == tidalplant.system_capacity_kw

def test_changing_device_rating(tidalplant, subtests):
    tidalplant.device_rated_power = 150
    with subtests.test("change system capacity"):
        assert tidalplant.system_capacity_kw == tidalplant.device_rated_power * tidalplant.number_devices
    
    with subtests.test("update cost model - device rated power"):
        assert tidalplant.mhk_costs.device_rated_power == tidalplant.device_rated_power
    
    with subtests.test("update cost model - system capacity"):
        assert tidalplant.mhk_costs.system_capacity_kw == tidalplant.system_capacity_kw