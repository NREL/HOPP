import os

from pytest import approx, warns
import yaml
from yamlinclude import YamlIncludeConstructor
import warnings

from greenheart.simulation.greenheart_simulation import (
    run_simulation,
    GreenHeartSimulationConfig,
)

from hopp.utilities.keys import set_nrel_key_dot_env

set_nrel_key_dot_env()

from ORBIT.core.library import initialize_library

dirname = os.path.dirname(__file__)
orbit_library_path = os.path.join(dirname, "input_files/")

YamlIncludeConstructor.add_to_loader_class(
    loader_class=yaml.FullLoader, base_dir=os.path.join(orbit_library_path, "floris/")
)
YamlIncludeConstructor.add_to_loader_class(
    loader_class=yaml.FullLoader, base_dir=os.path.join(orbit_library_path, "turbines/")
)

initialize_library(orbit_library_path)

turbine_model = "osw_18MW"
filename_turbine_config = os.path.join(
    orbit_library_path, f"turbines/{turbine_model}.yaml"
)
filename_orbit_config = os.path.join(
    orbit_library_path, f"plant/orbit-config-{turbine_model}-stripped.yaml"
)
filename_floris_config = os.path.join(
    orbit_library_path, f"floris/floris_input_{turbine_model}.yaml"
)
filename_greenheart_config = os.path.join(
    orbit_library_path, f"plant/greenheart_config.yaml"
)
filename_greenheart_config_onshore = os.path.join(
        orbit_library_path, f"plant/greenheart_config_onshore.yaml"
    )
filename_hopp_config = os.path.join(orbit_library_path, f"plant/hopp_config.yaml")


def test_simulation_wind(subtests):
    config = GreenHeartSimulationConfig(
        filename_hopp_config=filename_hopp_config,
        filename_greenheart_config=filename_greenheart_config,
        filename_turbine_config=filename_turbine_config,
        filename_orbit_config=filename_orbit_config,
        filename_floris_config=filename_floris_config,
        verbose=False,
        show_plots=False,
        save_plots=False,
        use_profast=True,
        post_processing=True,
        incentive_option=1,
        plant_design_scenario=1,
        output_level=5,
    )
    lcoe, lcoh, _, hi = run_simulation(config)

    with subtests.test("lcoh"):
        assert lcoh == approx(
            7.057994298481547
        )  # TODO base this test value on something

    with subtests.test("lcoe"):
        assert lcoe == approx(
            0.10816180445700445
        )  # TODO base this test value on something

    with subtests.test("energy sources"):
        expected_annual_energy_hybrid = hi.system.annual_energies.wind
        assert hi.system.annual_energies.hybrid == approx(expected_annual_energy_hybrid)

    with subtests.test("num_turbines conflict raise warning"):
        config.orbit_config["plant"]["num_turbines"] = 400
        with warns(UserWarning, match=f"The 'num_turbines' value"):
            lcoe, lcoh, _, hi = run_simulation(config)
            
    with subtests.test("depth conflict raise warning"):
        config.orbit_config["site"]["depth"] = 4000
        with warns(UserWarning, match=f"The site depth value"):
            lcoe, lcoh, _, hi = run_simulation(config)

    with subtests.test("turbine_spacing conflict raise warning"):
        config.orbit_config["plant"]["turbine_spacing"] = 400
        with warns(UserWarning, match=f"The 'turbine_spacing' value"):
            lcoe, lcoh, _, hi = run_simulation(config)

    with subtests.test("row_spacing conflict raise warning"):
        config.orbit_config["plant"]["row_spacing"] = 400
        with warns(UserWarning, match=f"The 'row_spacing' value"):
            lcoe, lcoh, _, hi = run_simulation(config)

def test_simulation_wind_wave(subtests):
    filename_hopp_config_wind_wave = os.path.join(
        orbit_library_path, f"plant/hopp_config_wind_wave.yaml"
    )

    config = GreenHeartSimulationConfig(
        filename_hopp_config=filename_hopp_config_wind_wave,
        filename_greenheart_config=filename_greenheart_config,
        filename_turbine_config=filename_turbine_config,
        filename_orbit_config=filename_orbit_config,
        filename_floris_config=filename_floris_config,
        verbose=False,
        show_plots=False,
        save_plots=False,
        use_profast=True,
        post_processing=True,
        incentive_option=1,
        plant_design_scenario=1,
        output_level=5,
    )

    lcoe, lcoh, _, hi = run_simulation(config)

    # TODO base this test value on something
    with subtests.test("lcoh"):
        assert lcoh == approx(8.133894926897252)

    # prior to 20240207 value was approx(0.11051228251811765) # TODO base this test value on something
    with subtests.test("lcoe"):
        assert lcoe == approx(0.12887769358919945)


def test_simulation_wind_wave_solar(subtests):
    filename_hopp_config_wind_wave_solar = os.path.join(
        orbit_library_path, f"plant/hopp_config_wind_wave_solar.yaml"
    )

    config = GreenHeartSimulationConfig(
        filename_hopp_config=filename_hopp_config_wind_wave_solar,
        filename_greenheart_config=filename_greenheart_config,
        filename_turbine_config=filename_turbine_config,
        filename_orbit_config=filename_orbit_config,
        filename_floris_config=filename_floris_config,
        verbose=False,
        show_plots=False,
        save_plots=False,
        use_profast=True,
        post_processing=True,
        incentive_option=1,
        plant_design_scenario=11,
        output_level=5,
    )

    lcoe, lcoh, _, hi = run_simulation(config)

    # prior to 20240207 value was approx(10.823798551850347)
    # TODO base this test value on something. Currently just based on output at writing.
    with subtests.test("lcoh"):
        assert lcoh == approx(12.597232748295273)

    # prior to 20240207 value was approx(0.11035426429749774)
    # TODO base this test value on something. Currently just based on output at writing.
    with subtests.test("lcoe"):
        assert lcoe == approx(0.12868090262684384)


def test_simulation_wind_wave_solar_battery(subtests):
    filename_hopp_config_wind_wave_solar_battery = os.path.join(
        orbit_library_path, f"plant/hopp_config_wind_wave_solar_battery.yaml"
    )

    config = GreenHeartSimulationConfig(
        filename_hopp_config=filename_hopp_config_wind_wave_solar_battery,
        filename_greenheart_config=filename_greenheart_config,
        filename_turbine_config=filename_turbine_config,
        filename_orbit_config=filename_orbit_config,
        filename_floris_config=filename_floris_config,
        verbose=False,
        show_plots=False,
        save_plots=False,
        use_profast=True,
        post_processing=True,
        incentive_option=1,
        plant_design_scenario=10,
        output_level=5,
    )

    lcoe, lcoh, _, hi = run_simulation(config)

    with subtests.test("lcoh"):
        # TODO base this test value on something. Currently just based on output at writing.
        assert lcoh == approx(16.984071652636903)

    # TODO base this test value on something. Currently just based on output at writing.
    with subtests.test("lcoe"):
        # TODO base this test value on something. Currently just based on output at writing.
        assert lcoe == approx(0.12936583137325117)  


def test_simulation_wind_onshore(subtests):

    config = GreenHeartSimulationConfig(
        filename_hopp_config=filename_hopp_config,
        filename_greenheart_config=filename_greenheart_config_onshore,
        filename_turbine_config=filename_turbine_config,
        filename_orbit_config=filename_orbit_config,
        filename_floris_config=filename_floris_config,
        verbose=False,
        show_plots=False,
        save_plots=False,
        use_profast=True,
        post_processing=False,
        incentive_option=1,
        plant_design_scenario=9,
        output_level=5,
    )
    # based on 2023 ATB moderate case for onshore wind
    config.hopp_config["config"]["cost_info"]["wind_installed_cost_mw"] = 1434000.0 
    # based on 2023 ATB moderate case for onshore wind
    config.hopp_config["config"]["cost_info"]["wind_om_per_kw"] = 29.567
    # set skip_financial to false for onshore wind
    config.hopp_config["config"]["simulation_options"]["wind"]["skip_financial"] = False

    lcoe, lcoh, _, _ = run_simulation(config)

    # TODO base this test value on something
    with subtests.test("lcoh"):
        assert lcoh == approx(3.040736244214041)  

    # TODO base this test value on something
    with subtests.test("lcoe"):
        assert lcoe == approx(0.034869608896094494)


def test_simulation_wind_onshore_steel_ammonia(subtests):

    config = GreenHeartSimulationConfig(
        filename_hopp_config=filename_hopp_config,
        filename_greenheart_config=filename_greenheart_config_onshore,
        filename_turbine_config=filename_turbine_config,
        filename_orbit_config=filename_orbit_config,
        filename_floris_config=filename_floris_config,
        verbose=False,
        show_plots=False,
        save_plots=True,
        use_profast=True,
        post_processing=True,
        incentive_option=1,
        plant_design_scenario=9,
        output_level=7,
    )
    # based on 2023 ATB moderate case for onshore wind
    config.hopp_config["config"]["cost_info"]["wind_installed_cost_mw"] = 1434000.0 
    # based on 2023 ATB moderate case for onshore wind
    config.hopp_config["config"]["cost_info"]["wind_om_per_kw"] = 29.567
    config.hopp_config["technologies"]["wind"]["fin_model"]["system_costs"]["om_fixed"][0] = config.hopp_config["config"]["cost_info"]["wind_om_per_kw"]
    # set skip_financial to false for onshore wind
    config.hopp_config["config"]["simulation_options"]["wind"]["skip_financial"] = False
    lcoe, lcoh, steel_finance, ammonia_finance = run_simulation(config)

    # TODO base this test value on something
    with subtests.test("lcoh"):
        assert lcoh == approx(3.040736244214041)

    # TODO base this test value on something
    with subtests.test("lcoe"):
        assert lcoe == approx(0.034869649135212274)

    # TODO base this test value on something
    with subtests.test("steel_finance"):
        lcos_expected = 1348.5863267221866

        assert steel_finance.sol.get("price") == approx(lcos_expected)

    # TODO base this test value on something
    with subtests.test("ammonia_finance"):
        lcoa_expected = 1.0419316870652462

        assert ammonia_finance.sol.get("price") == approx(lcoa_expected)
