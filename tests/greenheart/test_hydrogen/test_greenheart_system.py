from re import S
from greenheart.simulation.greenheart_simulation import run_simulation
from pytest import approx
import unittest

import os

from hopp.utilities.keys import set_nrel_key_dot_env

set_nrel_key_dot_env()

import yaml
from yamlinclude import YamlIncludeConstructor

from pathlib import Path
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
    orbit_library_path, f"plant/orbit-config-{turbine_model}.yaml"
)
filename_floris_config = os.path.join(
    orbit_library_path, f"floris/floris_input_{turbine_model}.yaml"
)
filename_greenheart_config = os.path.join(
    orbit_library_path, f"plant/greenheart_config.yaml"
)
filename_hopp_config = os.path.join(orbit_library_path, f"plant/hopp_config.yaml")


def test_simulation_wind(subtests):
    lcoe, lcoh, _ = run_simulation(
        filename_hopp_config,
        filename_greenheart_config,
        filename_turbine_config,
        filename_orbit_config,
        filename_floris_config,
        verbose=False,
        show_plots=False,
        save_plots=False,
        use_profast=True,
        post_processing=False,
        incentive_option=1,
        plant_design_scenario=1,
        output_level=4,
    )

    with subtests.test("lcoh"):
        assert lcoh == approx(
            7.057994298481547
        )  # TODO base this test value on something

    with subtests.test("lcoe"):
        assert lcoe == approx(
            0.10816180445700445
        )  # TODO base this test value on something


def test_simulation_wind_wave(subtests):
    filename_hopp_config_wind_wave = os.path.join(
        orbit_library_path, f"plant/hopp_config_wind_wave.yaml"
    )

    lcoe, lcoh, _ = run_simulation(
        filename_hopp_config_wind_wave,
        filename_greenheart_config,
        filename_turbine_config,
        filename_orbit_config,
        filename_floris_config,
        verbose=False,
        show_plots=False,
        save_plots=False,
        use_profast=True,
        post_processing=False,
        incentive_option=1,
        plant_design_scenario=1,
        output_level=4,
    )

    with subtests.test("lcoh"):
        assert lcoh == approx(
            8.120065296802442
        )  # TODO base this test value on something

    with subtests.test("lcoe"):
        assert lcoe == approx(
            0.12863386719193057
        )  # prior to 20240207 value was approx(0.11051228251811765) # TODO base this test value on something


def test_simulation_wind_wave_solar(subtests):
    filename_hopp_config_wind_wave_solar = os.path.join(
        orbit_library_path, f"plant/hopp_config_wind_wave_solar.yaml"
    )

    lcoe, lcoh, _ = run_simulation(
        filename_hopp_config_wind_wave_solar,
        filename_greenheart_config,
        filename_turbine_config,
        filename_orbit_config,
        filename_floris_config,
        verbose=False,
        show_plots=False,
        save_plots=False,
        use_profast=True,
        post_processing=False,
        incentive_option=1,
        plant_design_scenario=7,
        output_level=4,
    )

    with subtests.test("lcoh"):
        assert lcoh == approx(
            12.583155204831298
        )  # prior to 20240207 value was approx(10.823798551850347) #TODO base this test value on something. Currently just based on output at writing.

    with subtests.test("lcoe"):
        assert lcoe == approx(
            0.1284376127848134
        )  # prior to 20240207 value was approx(0.11035426429749774) # TODO base this test value on something. Currently just based on output at writing.


def test_simulation_wind_wave_solar_battery(subtests):
    filename_hopp_config_wind_wave_solar_battery = os.path.join(
        orbit_library_path, f"plant/hopp_config_wind_wave_solar_battery.yaml"
    )

    lcoe, lcoh, _ = run_simulation(
        filename_hopp_config_wind_wave_solar_battery,
        filename_greenheart_config,
        filename_turbine_config,
        filename_orbit_config,
        filename_floris_config,
        verbose=False,
        show_plots=False,
        save_plots=False,
        use_profast=True,
        post_processing=False,
        incentive_option=1,
        plant_design_scenario=7,
        output_level=4,
    )

    with subtests.test("lcoh"):
        assert lcoh == approx(
            13.22669818008385
        )  # TODO base this test value on something. Currently just based on output at writing.

    with subtests.test("lcoe"):
        assert lcoe == approx(
            0.13955940183722207
        )  # TODO base this test value on something. Currently just based on output at writing.


def test_simulation_wind_onshore(subtests):
    filename_greenheart_config_onshore = os.path.join(
        orbit_library_path, f"plant/greenheart_config_onshore.yaml"
    )

    lcoe, lcoh, _ = run_simulation(
        filename_hopp_config,
        filename_greenheart_config_onshore,
        filename_turbine_config,
        filename_orbit_config,
        filename_floris_config,
        verbose=False,
        show_plots=False,
        save_plots=False,
        use_profast=True,
        post_processing=False,
        incentive_option=1,
        plant_design_scenario=1,
        output_level=4,
    )

    with subtests.test("lcoh"):
        assert lcoh == approx(
            7.057994298481547
        )  # TODO base this test value on something

    with subtests.test("lcoe"):
        assert lcoe == approx(
            0.10816180445700445
        )  # TODO base this test value on something


def test_simulation_wind_onshore_steel_ammonia(subtests):
    turbine_model = "osw_18MW"
    filename_turbine_config = os.path.join(
        orbit_library_path, f"turbines/{turbine_model}.yaml"
    )
    filename_orbit_config = os.path.join(
        orbit_library_path, f"plant/orbit-config-{turbine_model}.yaml"
    )
    filename_floris_config = os.path.join(
        orbit_library_path, f"floris/floris_input_{turbine_model}.yaml"
    )
    filename_greenheart_config = os.path.join(
        orbit_library_path, f"plant/greenheart_config_onshore.yaml"
    )
    filename_hopp_config = os.path.join(orbit_library_path, f"plant/hopp_config.yaml")

    lcoe, lcoh, steel_finance, ammonia_finance = run_simulation(
        filename_hopp_config,
        filename_greenheart_config,
        filename_turbine_config,
        filename_orbit_config,
        filename_floris_config,
        verbose=False,
        show_plots=False,
        save_plots=False,
        use_profast=True,
        post_processing=False,
        incentive_option=1,
        plant_design_scenario=1,
        output_level=5,
    )

    with subtests.test("lcoh"):
        assert lcoh == approx(
            7.057994298481547
        )  # TODO base this test value on something

    with subtests.test("lcoe"):
        assert lcoe == approx(
            0.10816180445700445
        )  # TODO base this test value on something

    with subtests.test("steel_finance"):
        lcos_expected = 1626.5484389152123

        assert steel_finance.sol.get("price") == approx(lcos_expected)

    with subtests.test("ammonia_finance"):
        lcoa_expected = 1.0441398120394485

        assert ammonia_finance.sol.get("price") == approx(lcoa_expected)
