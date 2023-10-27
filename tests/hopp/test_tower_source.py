import pytest
from pytest import fixture

from numpy.testing import assert_array_equal

from hopp.simulation.technologies.csp.tower_plant import TowerConfig, TowerPlant
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
        config = TowerConfig.from_dict(config_data)

        assert config.tech_name == "tcsmolten_salt"
        assert config.cycle_capacity_kw == config_data["cycle_capacity_kw"]
        assert config.solar_multiple == config_data["solar_multiple"]
        assert config.tes_hours == config_data["tes_hours"]
        assert config.fin_model is None
        assert config.optimize_field_before_sim == True
        assert config.scale_input_params == False
        assert config.name == "TowerPlant"

    with subtests.test("with invalid tech_name"):
        data = config_data.copy()
        data["tech_name"] = "bad"

        with pytest.raises(ValueError):
            config = TowerConfig.from_dict(data)


def test_trough_init(site, subtests):
    config = TowerConfig.from_dict(config_data)

    with subtests.test("with default params"):
        trough = TowerPlant(site, config=config)

        assert trough._financial_model is not None
        param_files_keys = [
            "tech_model_params_path", 
            "cf_params_path", 
            "wlim_series_path", 
            "helio_positions_path"
        ]
        assert_array_equal(list(trough.param_files.keys()), param_files_keys)

    with subtests.test("with scale input"):
        data = config_data.copy()
        data["scale_input_params"] = True

        TowerPlant(site, config=config)