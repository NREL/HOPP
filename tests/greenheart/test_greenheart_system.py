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

from greenheart.tools.optimization.gc_run_greenheart import run_greenheart

from hopp.utilities.keys import set_nrel_key_dot_env
from greenheart.tools.eco.utilities import visualize_plant, ceildiv

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
filename_hopp_config = os.path.join(
    orbit_library_path, f"plant/hopp_config.yaml"
)
filename_hopp_config_wind_wave = os.path.join(
        orbit_library_path, f"plant/hopp_config_wind_wave.yaml"
)
filename_hopp_config_wind_wave_solar = os.path.join(
        orbit_library_path, f"plant/hopp_config_wind_wave_solar.yaml"
)
filename_hopp_config_wind_wave_solar_battery = os.path.join(
        orbit_library_path, f"plant/hopp_config_wind_wave_solar_battery.yaml"
)

rtol = 1E-5

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
            7.248875250895419
        )  # TODO base this test value on something

    with subtests.test("lcoe"):
        assert lcoe == approx(
            0.10813733018947481
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
        assert lcoh == approx(8.228321816022078, rel=rtol)

    # prior to 20240207 value was approx(0.11051228251811765) # TODO base this test value on something
    with subtests.test("lcoe"):
        assert lcoe == approx(0.12885017943224733, rel=rtol)


def test_simulation_wind_wave_solar(subtests):

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
        assert lcoh == approx(12.719884986359553, rel=rtol)

    # prior to 20240207 value was approx(0.11035426429749774)
    # TODO base this test value on something. Currently just based on output at writing.
    with subtests.test("lcoe"):
        assert lcoe == approx(0.12865548678206473, rel=rtol)


def test_simulation_wind_wave_solar_battery(subtests):

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
        output_level=8,
    )

    results = run_simulation(config)

    with subtests.test("lcoh"):
        # TODO base this test value on something. Currently just based on output at writing.
        assert results.lcoh == approx(17.11063907134404, rel=rtol)

    # TODO base this test value on something. Currently just based on output at writing.
    with subtests.test("lcoe"):
        # TODO base this test value on something. Currently just based on output at writing.
        assert results.lcoe == approx(0.1294193054583137, rel=rtol)  

    with subtests.test("no conflict in om cost does not raise warning"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")

    with subtests.test("wind_om_per_kw conflict raise warning"):
        config.hopp_config["technologies"]["wind"]["fin_model"]["system_costs"]["om_capacity"][0] = 1.0
        with warns(UserWarning, match=f"The 'om_capacity' value in the wind 'fin_model'"):
            _ = run_simulation(config)
    
    with subtests.test("pv_om_per_kw conflict raise warning"):
        config.hopp_config["technologies"]["pv"]["fin_model"]["system_costs"]["om_capacity"][0] = 1.0
        with warns(UserWarning, match=f"The 'om_capacity' value in the pv 'fin_model'"):
            _ = run_simulation(config)

    with subtests.test("battery_om_per_kw conflict raise warning"):
        config.hopp_config["technologies"]["battery"]["fin_model"]["system_costs"]["om_capacity"][0] = 1.0
        with warns(UserWarning, match=f"The 'om_capacity' value in the battery 'fin_model'"):
            _ = run_simulation(config)

def test_simulation_wind_onshore(subtests):

    config = GreenHeartSimulationConfig(
        filename_hopp_config=filename_hopp_config,
        filename_greenheart_config=filename_greenheart_config_onshore,
        filename_turbine_config=filename_turbine_config,
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
        assert lcoh == approx(3.1691092704830357, rel=rtol)  

    # TODO base this test value on something
    with subtests.test("lcoe"):
        assert lcoe == approx(0.03486192934806013, rel=rtol)


def test_simulation_wind_onshore_steel_ammonia(subtests):

    config = GreenHeartSimulationConfig(
        filename_hopp_config=filename_hopp_config,
        filename_greenheart_config=filename_greenheart_config_onshore,
        filename_turbine_config=filename_turbine_config,
        filename_floris_config=filename_floris_config,
        verbose=False,
        show_plots=False,
        save_plots=True,
        output_dir=os.path.abspath(pathlib.Path(__file__).parent.resolve()) + "/output/",
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
        assert lcoh == approx(3.1691092704830357, rel=rtol)

    # TODO base this test value on something
    with subtests.test("lcoe"):
        assert lcoe == approx(0.03486192934806013, rel=rtol)

    # TODO base this test value on something
    with subtests.test("steel_finance"):
        lcos_expected = 1357.046163641118

        assert steel_finance.sol.get("price") == approx(lcos_expected, rel=rtol)

    # TODO base this test value on something
    with subtests.test("ammonia_finance"):
        lcoa_expected = 1.0419096226034346

        assert ammonia_finance.sol.get("price") == approx(lcoa_expected, rel=rtol)

def test_simulation_wind_battery_pv_onshore_steel_ammonia(subtests):

    plant_design_scenario = 12

    config = GreenHeartSimulationConfig(
        filename_hopp_config=filename_hopp_config_wind_wave_solar_battery,
        filename_greenheart_config=filename_greenheart_config_onshore,
        filename_turbine_config=filename_turbine_config,
        filename_orbit_config=filename_orbit_config,
        filename_floris_config=filename_floris_config,
        verbose=False,
        show_plots=False,
        save_plots=True,
        output_dir=os.path.abspath(pathlib.Path(__file__).parent.resolve()) + "/output/",
        use_profast=True,
        post_processing=True,
        incentive_option=1,
        plant_design_scenario=plant_design_scenario,
        output_level=8,
    )
    
    # based on 2023 ATB moderate case for onshore wind
    config.hopp_config["config"]["cost_info"]["wind_installed_cost_mw"] = 1434000.0 
    # based on 2023 ATB moderate case for onshore wind
    config.hopp_config["config"]["cost_info"]["wind_om_per_kw"] = 29.567
    config.hopp_config["technologies"]["wind"]["fin_model"]["system_costs"]["om_fixed"][0] = config.hopp_config["config"]["cost_info"]["wind_om_per_kw"]
    # set skip_financial to false for onshore wind
    config.hopp_config["config"]["simulation_options"]["wind"]["skip_financial"] = False
    # exclude wave
    config.hopp_config["technologies"].pop("wave")
    config.hopp_config["site"]["wave"] = False
    # colocated end-use
    config.greenheart_config["plant_design"][f"scenario{plant_design_scenario}"]["transportation"] = "colocated" 

    # run the simulation
    greenheart_output = run_simulation(config)

    # TODO base this test value on something
    with subtests.test("lcoh"):
        assert greenheart_output.lcoh == approx(3.1509756450008752, rel=rtol)

    # TODO base this test value on something
    with subtests.test("lcoe"):
        assert greenheart_output.lcoe == approx(0.03476011434910082, rel=rtol)

    # TODO base this test value on something
    with subtests.test("steel_finance"):
        lcos_expected = 1349.3364242679354

        assert greenheart_output.steel_finance.sol.get("price") == approx(lcos_expected, rel=rtol)

    # TODO base this test value on something
    with subtests.test("ammonia_finance"):
        lcoa_expected = 1.0404837286866984

        assert greenheart_output.ammonia_finance.sol.get("price") == approx(lcoa_expected, rel=rtol)

    with subtests.test("check time series lengths"):
        expected_length = 8760
        
        for key in greenheart_output.hourly_energy_breakdown.keys():
            assert len(greenheart_output.hourly_energy_breakdown[key]) == expected_length

def test_simulation_wind_onshore_steel_ammonia_ss_h2storage(subtests):

    config = GreenHeartSimulationConfig(
        filename_hopp_config=filename_hopp_config,
        filename_greenheart_config=filename_greenheart_config_onshore,
        filename_turbine_config=filename_turbine_config,
        filename_floris_config=filename_floris_config,
        verbose=False,
        show_plots=False,
        save_plots=True,
        output_dir=os.path.abspath(pathlib.Path(__file__).parent.resolve()) + "/output/",
        use_profast=True,
        post_processing=True,
        incentive_option=1,
        plant_design_scenario=9,
        output_level=7,
    )

    config.greenheart_config['h2_storage']['size_capacity_from_demand']['flag'] = True
    config.greenheart_config['h2_storage']['type'] = 'pipe'
    
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
        assert lcoh == approx(10.0064010897151, rel=rtol)

    # TODO base this test value on something
    with subtests.test("lcoe"):
        assert lcoe == approx(0.03486192934806013, rel=rtol)

    # TODO base this test value on something
    with subtests.test("steel_finance"):
        lcos_expected = 1812.985744428756 

        assert steel_finance.sol.get("price") == approx(lcos_expected, rel=rtol)

    # TODO base this test value on something
    with subtests.test("ammonia_finance"):
        lcoa_expected = 1.0419096226034346

        assert ammonia_finance.sol.get("price") == approx(lcoa_expected, rel=rtol)