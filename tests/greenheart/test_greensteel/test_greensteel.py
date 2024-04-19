import os

from pytest import approx, warns, raises
import yaml
from yamlinclude import YamlIncludeConstructor
import warnings
import pathlib

from greenheart.simulation.greenheart_simulation import (
    run_simulation,
    GreenHeartSimulationConfig,
)

from hopp.utilities.keys import set_nrel_key_dot_env
from greenheart.tools.eco.utilities import visualize_plant

dirname = os.path.dirname(__file__)
path = os.path.join(dirname, "input/")

def test_onshore_steel_mn_no_policy_2030(subtests):
    # load inputs as needed
    turbine_model = "lbw_6MW"
    filename_turbine_config = os.path.join(path, f"turbines/{turbine_model}.yaml")
    filename_floris_config = os.path.join(path, "floris/floris_input_lbw_6MW.yaml")
    filename_hopp_config = os.path.join(path, "plant/hopp_config_mn.yaml")
    filename_greenheart_config = os.path.join(path,"plant/greenheart_config_onshore_mn.yaml")

    config = GreenHeartSimulationConfig(
        filename_hopp_config,
        filename_greenheart_config,
        filename_turbine_config,
        filename_floris_config,
        verbose=True,
        show_plots=False,
        save_plots=True,
        use_profast=True,
        post_processing=True,
        incentive_option=1,
        plant_design_scenario=9,
        output_level=7,
    )

    config.hopp_config["technologies"]["wind"]["fin_model"]["system_costs"]["om_fixed"][
        0
    ] = config.hopp_config["config"]["cost_info"]["wind_om_per_kw"]
    config.hopp_config["technologies"]["pv"]["fin_model"]["system_costs"]["om_fixed"][
        0
    ] = config.hopp_config["config"]["cost_info"]["pv_om_per_kw"]
    config.hopp_config["technologies"]["battery"]["fin_model"]["system_costs"][
        "om_batt_fixed_cost"
    ] = config.hopp_config["config"]["cost_info"]["battery_om_per_kw"]

    lcoe, lcoh, steel_finance, _ = run_simulation(config)

    with subtests.test("lcoh"):
        assert lcoh == approx(
            4.2986685034417045
        )

    with subtests.test("lcos"):
        assert steel_finance.sol.get("price") == approx(961.2866791076059)

def test_onshore_ammonia_tx_2030_no_policy(subtests):
    # load inputs as needed
    turbine_model = "lbw_6MW"
    filename_turbine_config = os.path.join(path, f"turbines/{turbine_model}.yaml")
    filename_floris_config = os.path.join(path, "floris/floris_input_lbw_6MW.yaml")
    filename_hopp_config = os.path.join(path, "plant/hopp_config_tx.yaml")
    filename_greenheart_config = os.path.join(path,"plant/greenheart_config_onshore_tx.yaml")

    config = GreenHeartSimulationConfig(
        filename_hopp_config,
        filename_greenheart_config,
        filename_turbine_config,
        filename_floris_config,
        verbose=True,
        show_plots=False,
        save_plots=True,
        use_profast=True,
        post_processing=True,
        incentive_option=1,
        plant_design_scenario=9,
        output_level=7,
    )

    config.hopp_config["technologies"]["wind"]["fin_model"]["system_costs"]["om_fixed"][
        0
    ] = config.hopp_config["config"]["cost_info"]["wind_om_per_kw"]
    config.hopp_config["technologies"]["pv"]["fin_model"]["system_costs"]["om_fixed"][
        0
    ] = config.hopp_config["config"]["cost_info"]["pv_om_per_kw"]
    config.hopp_config["technologies"]["battery"]["fin_model"]["system_costs"][
        "om_batt_fixed_cost"
    ] = config.hopp_config["config"]["cost_info"]["battery_om_per_kw"]

    lcoe, lcoh, _, ammonia_finance = run_simulation(config)

    with subtests.test("lcoh"):
        assert lcoh == approx(
            4.023963541118758
        )

    with subtests.test("lcos"):
        assert ammonia_finance.sol.get("price") == approx(0.8839797787889466)

def test_onshore_ammonia_tx_2030_base_policy(subtests):
    # load inputs as needed
    # NOTE: GreenSteel used Wind PTC, H2 PTC and Storage ITC for base policy and incentive_option 2 does not include the Storage ITC
    turbine_model = "lbw_6MW"
    filename_turbine_config = os.path.join(path, f"turbines/{turbine_model}.yaml")
    filename_floris_config = os.path.join(path, "floris/floris_input_lbw_6MW.yaml")
    filename_hopp_config = os.path.join(path, "plant/hopp_config_tx.yaml")
    filename_greenheart_config = os.path.join(path,"plant/greenheart_config_onshore_tx.yaml")

    config = GreenHeartSimulationConfig(
        filename_hopp_config,
        filename_greenheart_config,
        filename_turbine_config,
        filename_floris_config,
        verbose=True,
        show_plots=False,
        save_plots=True,
        use_profast=True,
        post_processing=True,
        incentive_option=2,
        plant_design_scenario=9,
        output_level=7,
    )

    config.hopp_config["technologies"]["wind"]["fin_model"]["system_costs"]["om_fixed"][
        0
    ] = config.hopp_config["config"]["cost_info"]["wind_om_per_kw"]
    config.hopp_config["technologies"]["pv"]["fin_model"]["system_costs"]["om_fixed"][
        0
    ] = config.hopp_config["config"]["cost_info"]["pv_om_per_kw"]
    config.hopp_config["technologies"]["battery"]["fin_model"]["system_costs"][
        "om_batt_fixed_cost"
    ] = config.hopp_config["config"]["cost_info"]["battery_om_per_kw"]

    lcoe, lcoh, _, ammonia_finance = run_simulation(config)

    with subtests.test("lcoh"):
        assert lcoh == approx(
            3.2231088737405846
        )

    with subtests.test("lcos"):
        assert ammonia_finance.sol.get("price") == approx(0.7242460501735473)

def test_onshore_ammonia_tx_2030_max_policy(subtests):
    # load inputs as needed
    # NOTE: GreenSteel used Wind PTC, H2 PTC and Storage ITC for max policy and incentive_option 3 does not include the Storage ITC
    turbine_model = "lbw_6MW"
    filename_turbine_config = os.path.join(path, f"turbines/{turbine_model}.yaml")
    filename_floris_config = os.path.join(path, "floris/floris_input_lbw_6MW.yaml")
    filename_hopp_config = os.path.join(path, "plant/hopp_config_tx.yaml")
    filename_greenheart_config = os.path.join(path,"plant/greenheart_config_onshore_tx.yaml")

    config = GreenHeartSimulationConfig(
        filename_hopp_config,
        filename_greenheart_config,
        filename_turbine_config,
        filename_floris_config,
        verbose=True,
        show_plots=False,
        save_plots=True,
        use_profast=True,
        post_processing=True,
        incentive_option=3,
        plant_design_scenario=9,
        output_level=7,
    )

    config.hopp_config["technologies"]["wind"]["fin_model"]["system_costs"]["om_fixed"][
        0
    ] = config.hopp_config["config"]["cost_info"]["wind_om_per_kw"]
    config.hopp_config["technologies"]["pv"]["fin_model"]["system_costs"]["om_fixed"][
        0
    ] = config.hopp_config["config"]["cost_info"]["pv_om_per_kw"]
    config.hopp_config["technologies"]["battery"]["fin_model"]["system_costs"][
        "om_batt_fixed_cost"
    ] = config.hopp_config["config"]["cost_info"]["battery_om_per_kw"]

    lcoe, lcoh, _, ammonia_finance = run_simulation(config)

    with subtests.test("lcoh"):
        assert lcoh == approx(
            -0.3719231321573331
        )

    with subtests.test("lcos"):
        assert ammonia_finance.sol.get("price") == approx(0.007202259061804786)

def test_onshore_ammonia_tx_2025_no_policy(subtests):
    # load inputs as needed
    turbine_model = "lbw_6MW"
    filename_turbine_config = os.path.join(path, f"turbines/{turbine_model}.yaml")
    filename_floris_config = os.path.join(path, "floris/floris_input_lbw_6MW.yaml")
    filename_hopp_config = os.path.join(path, "plant/hopp_config_tx.yaml")
    filename_greenheart_config = os.path.join(path,"plant/greenheart_config_onshore_tx_2025.yaml")

    config = GreenHeartSimulationConfig(
        filename_hopp_config,
        filename_greenheart_config,
        filename_turbine_config,
        filename_floris_config,
        verbose=True,
        show_plots=False,
        save_plots=True,
        use_profast=True,
        post_processing=True,
        incentive_option=1,
        plant_design_scenario=9,
        output_level=7,
    )

    config.hopp_config["technologies"]["wind"]["fin_model"]["system_costs"]["om_fixed"][
        0
    ] = config.hopp_config["config"]["cost_info"]["wind_om_per_kw"]
    config.hopp_config["technologies"]["pv"]["fin_model"]["system_costs"]["om_fixed"][
        0
    ] = config.hopp_config["config"]["cost_info"]["pv_om_per_kw"]
    config.hopp_config["technologies"]["battery"]["fin_model"]["system_costs"][
        "om_batt_fixed_cost"
    ] = config.hopp_config["config"]["cost_info"]["battery_om_per_kw"]

    lcoe, lcoh, _, ammonia_finance = run_simulation(config)

    with subtests.test("lcoh"):
        assert lcoh == approx(
            5.215316325627814
        )

    with subtests.test("lcos"):
        assert ammonia_finance.sol.get("price") == approx(1.1211844274252674)

def test_onshore_ammonia_tx_2035_no_policy(subtests):
    # load inputs as needed
    turbine_model = "lbw_6MW"
    filename_turbine_config = os.path.join(path, f"turbines/{turbine_model}.yaml")
    filename_floris_config = os.path.join(path, "floris/floris_input_lbw_6MW.yaml")
    filename_hopp_config = os.path.join(path, "plant/hopp_config_tx.yaml")
    filename_greenheart_config = os.path.join(path,"plant/greenheart_config_onshore_tx_2035.yaml")

    config = GreenHeartSimulationConfig(
        filename_hopp_config,
        filename_greenheart_config,
        filename_turbine_config,
        filename_floris_config,
        verbose=True,
        show_plots=False,
        save_plots=True,
        use_profast=True,
        post_processing=True,
        incentive_option=1,
        plant_design_scenario=9,
        output_level=7,
    )

    config.hopp_config["technologies"]["wind"]["fin_model"]["system_costs"]["om_fixed"][
        0
    ] = config.hopp_config["config"]["cost_info"]["wind_om_per_kw"]
    config.hopp_config["technologies"]["pv"]["fin_model"]["system_costs"]["om_fixed"][
        0
    ] = config.hopp_config["config"]["cost_info"]["pv_om_per_kw"]
    config.hopp_config["technologies"]["battery"]["fin_model"]["system_costs"][
        "om_batt_fixed_cost"
    ] = config.hopp_config["config"]["cost_info"]["battery_om_per_kw"]

    lcoe, lcoh, _, ammonia_finance = run_simulation(config)

    with subtests.test("lcoh"):
        assert lcoh == approx(
            3.68491615716891
        )

    with subtests.test("lcos"):
        assert ammonia_finance.sol.get("price") == approx(0.8161748484435717)