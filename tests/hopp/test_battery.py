from unittest.mock import MagicMock
from copy import deepcopy

import pytest
from pytest import fixture

from hopp.simulation.technologies.battery import Battery, BatteryConfig
from tests.hopp.utils import create_default_site_info


batt_kw = 5e3

config_data = {
    'system_capacity_kwh': batt_kw * 4,
    'system_capacity_kw': batt_kw
}

@fixture
def site():
    return create_default_site_info()


def test_battery_config(subtests):
    with subtests.test("with minimal params"):
        config = BatteryConfig.from_dict(config_data)

        assert config.system_capacity_kw == batt_kw
        assert config.system_capacity_kwh == batt_kw * 4
        assert config.tracking is True
        assert config.minimum_SOC == 10.
        assert config.maximum_SOC == 90.
        assert config.initial_SOC == 10.
        assert config.fin_model is None

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

    assert battery._financial_model is not None
    assert battery._system_model is not None
    assert battery.outputs is not None
    assert battery.chemistry == "LFPGraphite"
    assert battery.system_capacity_kw == config.system_capacity_kw
    assert battery.system_capacity_kwh == config.system_capacity_kwh

    with subtests.test("with custom financial model"):
        data = deepcopy(config_data)
        fin_model = MagicMock() # duck type a financial model for simplicity
        data["fin_model"] = fin_model

        config = BatteryConfig.from_dict(data)
        battery = Battery(site, config=config)

        assert battery._financial_model == fin_model

    with subtests.test("battery mass"):
        assert battery.system_mass == pytest.approx(304454.0,1e-3) #TODO: verify system mass. Current value is just based on output at writing.

    with subtests.test("battery footprint area"):
        assert battery.footprint_area == pytest.approx(250.0, 1e-3) #TODO: verify system mass. Current value is just based on output at writing.
