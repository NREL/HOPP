from copy import deepcopy

import pytest
from pytest import fixture

from hopp.simulation.technologies.battery import Battery, BatteryConfig
from tests.hopp.utils import create_default_site_info


batt_kw = 5e3


@fixture
def site():
    return create_default_site_info()


def test_battery_config(subtests):
    config_data = {
        'system_capacity_kwh': batt_kw * 4,
        'system_capacity_kw': batt_kw
    }

    with subtests.test("with minimal params"):
        config = BatteryConfig.from_dict(config_data)

        assert config.system_capacity_kw == batt_kw
        assert config.system_capacity_kwh == batt_kw * 4
        assert config.tracking is False
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


def test_battery_initialization(site):
    config = BatteryConfig.from_dict({
        'system_capacity_kwh': batt_kw * 4,
        'system_capacity_kw': batt_kw
    })
    battery = Battery(site, config=config)

    assert battery.financial_model is not None
    assert battery.system_model is not None
    assert battery.outputs is not None
    assert battery.chemistry == "lfpgraphite"
    assert battery.system_capacity_kw == config.system_capacity_kw
    assert battery.system_capacity_kwh == config.system_capacity_kwh