from unittest.mock import MagicMock
from copy import deepcopy

import pytest
from pytest import fixture

from hopp.simulation.technologies.battery import Battery, BatteryConfig
from tests.hopp.utils import create_default_site_info

from tests.hopp.utils import DEFAULT_FIN_CONFIG

batt_kw = 5e3

config_data = {
    'system_capacity_kwh': batt_kw * 4,
    'system_capacity_kw': batt_kw,
    'system_model_source': "hopp",
    'chemistry': "LDES",
	"fin_model": DEFAULT_FIN_CONFIG,
}

@fixture
def site():
    return create_default_site_info()


def test_battery_config(subtests):

    config = BatteryConfig.from_dict(config_data)

    with subtests.test("with minimal params batt_kw"):
        assert config.system_capacity_kw == batt_kw
    with subtests.test("with minimal params system_capacity_kwh"):
        assert config.system_capacity_kwh == batt_kw * 4
    with subtests.test("with minimal params tracking"):
        assert config.tracking is True
    with subtests.test("with minimal params minimum_SOC"):
        assert config.minimum_SOC == 10.
    with subtests.test("with minimal params maximum_SOC"):
        assert config.maximum_SOC == 90.
    with subtests.test("with minimal params initial_SOC"):
        assert config.initial_SOC == 10.
    with subtests.test("with minimal params system_model_source"):
        assert config.system_model_source is "hopp"

    with subtests.test("with invalid capacity"):
        with pytest.raises(ValueError):
            data = deepcopy(config_data)
            data["system_capacity_kw"] = -1.
            BatteryConfig.from_dict(data)

        with pytest.raises(ValueError):
            data = deepcopy(config_data)
            data["system_capacity_kwh"] = -1.
            BatteryConfig.from_dict(data)

    with subtests.test("with invalid SOC"):
        # SOC values must be between 0-100
        with pytest.raises(ValueError):
            data = deepcopy(config_data)
            data["minimum_SOC"] = -1.
            BatteryConfig.from_dict(data)

        with pytest.raises(ValueError):
            data = deepcopy(config_data)
            data["maximum_SOC"] = 120.
            BatteryConfig.from_dict(data)

        with pytest.raises(ValueError):
            data = deepcopy(config_data)
            data["initial_SOC"] = 120.
            BatteryConfig.from_dict(data)


def test_battery_initialization(site, subtests):
    config = BatteryConfig.from_dict(config_data)
    battery = Battery(site, config=config)

    with subtests.test("battery attribute not None _financial_model"):
        assert battery._financial_model is not None
    with subtests.test("battery attribute not None _system_model"):
        assert battery._system_model is not None
    with subtests.test("battery attribute not None outputs"):
        assert battery.outputs is not None
    with subtests.test("battery attribute chemistry"):
        assert battery.chemistry == "LDES"
    with subtests.test("battery attribute system_capacity_kw"):
        assert battery.system_capacity_kw == config.system_capacity_kw
    with subtests.test("battery attribute system_capacity_kwh"):
        assert battery.system_capacity_kwh == config.system_capacity_kwh

    with subtests.test("with custom financial model"):
        data = deepcopy(config_data)
        fin_model = MagicMock() # duck type a financial model for simplicity
        data["fin_model"] = fin_model

        config = BatteryConfig.from_dict(data)
        battery = Battery(site, config=config)

        assert battery._financial_model == fin_model

def test_battery_initialization_with_replacement_schedule(site, subtests):

    config_data_local = deepcopy(config_data)
    config_data_local["fin_model"]["battery_system"]["batt_replacement_option"] = 2
    length = 25
    refurb = [0]*length
    n = 10
    for i in range(n-1, length, n):
        refurb[i] = 0.5
    config_data_local["fin_model"]["battery_system"]["batt_replacement_schedule_percent"] = refurb

    config_data_local["fin_model"]["name"] = "LDES"

    config = BatteryConfig.from_dict(config_data_local)
    battery = Battery(site, config=config)

    with subtests.test("battery attribute not None _financial_model"):
        assert battery._financial_model is not None
    with subtests.test("battery attribute not None _system_model"):
        assert battery._system_model is not None
    with subtests.test("battery attribute not None outputs"):
        assert battery.outputs is not None
    with subtests.test("battery attribute chemistry"):
        assert battery.chemistry == "LDES"
    with subtests.test("battery attribute system_capacity_kw"):
        assert battery.system_capacity_kw == config.system_capacity_kw
    with subtests.test("battery attribute system_capacity_kwh"):
        assert battery.system_capacity_kwh == config.system_capacity_kwh
    with subtests.test("financial model attribute batt_replacement_option"):
        assert battery._financial_model.BatterySystem.batt_replacement_option == 2
    with subtests.test("financial model attribute batt_replacement_option"):
        assert battery._financial_model.BatterySystem.batt_replacement_schedule_percent == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0]

    with subtests.test("with custom financial model"):
        data = deepcopy(config_data)
        fin_model = MagicMock() # duck type a financial model for simplicity
        data["fin_model"] = fin_model

        config = BatteryConfig.from_dict(data)
        battery = Battery(site, config=config)

        assert battery._financial_model == fin_model