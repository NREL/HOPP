from unittest.mock import MagicMock
from copy import deepcopy

import pytest
from pytest import fixture

from hopp.simulation.technologies.battery import (
    BatteryStateless, BatteryStatelessConfig
)
from tests.hopp.utils import create_default_site_info


batt_kw = 5e3

fin_model = MagicMock() # duck typing a financial model for simplicity
config_data = {
    'system_capacity_kwh': batt_kw * 4,
    'system_capacity_kw': batt_kw,
    'fin_model': fin_model
}

@fixture
def site():
    return create_default_site_info()


def test_battery_config(subtests):

    with subtests.test("with minimal params"):
        config = BatteryStatelessConfig.from_dict(config_data)

        assert config.system_capacity_kw == batt_kw
        assert config.system_capacity_kwh == batt_kw * 4
        assert config.tracking is False
        assert config.minimum_SOC == 10.
        assert config.maximum_SOC == 90.
        assert config.initial_SOC == 10.
        assert config.fin_model == fin_model

    with subtests.test("with invalid capacity"):
        with pytest.raises(ValueError):
            data = deepcopy(config_data)
            data["system_capacity_kw"] = -1.
            BatteryStatelessConfig.from_dict(data)

        with pytest.raises(ValueError):
            data = deepcopy(config_data)
            data["system_capacity_kwh"] = -1.
            BatteryStatelessConfig.from_dict(data)

    with subtests.test("with invalid SOC"):
        # SOC values must be between 0-100
        with pytest.raises(ValueError):
            data = deepcopy(config_data)
            data["minimum_SOC"] = -1.
            BatteryStatelessConfig.from_dict(data)

        with pytest.raises(ValueError):
            data = deepcopy(config_data)
            data["maximum_SOC"] = 120.
            BatteryStatelessConfig.from_dict(data)

        with pytest.raises(ValueError):
            data = deepcopy(config_data)
            data["initial_SOC"] = 120.
            BatteryStatelessConfig.from_dict(data)


def test_battery_initialization(site):
    config = BatteryStatelessConfig.from_dict(config_data)
    battery = BatteryStateless(site, config=config)

    assert battery.financial_model == fin_model
    assert battery.outputs is not None
    assert battery.system_capacity_kw == config.system_capacity_kw
    assert battery.system_capacity_kwh == config.system_capacity_kwh