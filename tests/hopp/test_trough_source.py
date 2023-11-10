import pytest
from pytest import fixture

from numpy.testing import assert_array_equal

from hopp.simulation.technologies.csp.trough_plant import TroughConfig, TroughPlant
from tests.hopp.utils import create_default_site_info


config_data = {
    'cycle_capacity_kw': 15 * 1000,
    'solar_multiple': 2.0,
    'tes_hours': 6.0
}


@fixture
def site():
    return create_default_site_info()


def test_trough_config(subtests):
    with subtests.test("with default params"):
        config = TroughConfig.from_dict(config_data)

        assert config.tech_name == "trough_physical"
        assert config.cycle_capacity_kw == config_data["cycle_capacity_kw"]
        assert config.solar_multiple == config_data["solar_multiple"]
        assert config.tes_hours == config_data["tes_hours"]
        assert config.fin_model is None
        assert config.name == "TroughPlant"

    with subtests.test("with invalid tech_name"):
        data = config_data.copy()
        data["tech_name"] = "bad"

        with pytest.raises(ValueError):
            config = TroughConfig.from_dict(data)


def test_trough_init(site, subtests):
    config = TroughConfig.from_dict(config_data)

    with subtests.test("with default params"):
        trough = TroughPlant(site, config=config)

        assert trough._financial_model is not None
        param_files_keys = ["tech_model_params_path", "cf_params_path", "wlim_series_path"]
        assert_array_equal(list(trough.param_files.keys()), param_files_keys)