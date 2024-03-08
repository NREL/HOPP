from greenheart.simulation.greenheart_simulation import run_simulation
from pytest import approx
import unittest

import os

from hopp.utilities.keys import set_nrel_key_dot_env
set_nrel_key_dot_env()

import yaml
from yamlinclude import YamlIncludeConstructor 

from ORBIT.core.library import initialize_library

dirname = os.path.dirname(__file__)
orbit_library_path = os.path.join(dirname, "input_files/")

YamlIncludeConstructor.add_to_loader_class(
    loader_class=yaml.FullLoader,
    base_dir=os.path.join(orbit_library_path,'floris/')
)
YamlIncludeConstructor.add_to_loader_class(
    loader_class=yaml.FullLoader,
    base_dir=os.path.join(orbit_library_path, 'turbines/')
)

initialize_library(orbit_library_path)

class TestSimulationWind(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    @classmethod
    def setUpClass(self):
        super(TestSimulationWind, self).setUpClass()

        turbine_model = "osw_18MW"
        filename_turbine_config = os.path.join(
            orbit_library_path,
            f"turbines/{turbine_model}.yaml"
        )
        filename_orbit_config = os.path.join(
            orbit_library_path,
            f"plant/orbit-config-{turbine_model}.yaml"
        )
        filename_floris_config = os.path.join(
            orbit_library_path,
            f"floris/floris_input_{turbine_model}.yaml"
        )
        filename_greenheart_config = os.path.join(
            orbit_library_path,
            f"plant/greenheart_config.yaml"
        )
        filename_hopp_config = os.path.join(
            orbit_library_path,
            f"plant/hopp_config.yaml"
        )

        self.lcoe, self.lcoh, _, self.hi = run_simulation(
            filename_hopp_config, 
            filename_greenheart_config, 
            filename_turbine_config, 
            filename_orbit_config, 
            filename_floris_config, 
            verbose=False, 
            show_plots=False, 
            save_plots=False,  
            use_profast=True, 
            post_processing=True,
            incentive_option=1, 
            plant_design_scenario=1, 
            output_level=5
        )

    def test_lcoh(self):
        assert self.lcoh == approx(7.057994298481547) # TODO base this test value on something

    def test_lcoe(self):
        assert self.lcoe == approx(0.10816180445700445) # TODO base this test value on something

    def test_energy_sources(self):
        expected_annual_energy_hybrid = self.hi.system.annual_energies.wind
        assert self.hi.system.annual_energies.hybrid == approx(expected_annual_energy_hybrid)

class TestSimulationWindWave(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    @classmethod
    def setUpClass(self):
        super(TestSimulationWindWave, self).setUpClass()

        turbine_model = "osw_18MW"
        filename_turbine_config = os.path.join(
            orbit_library_path,
            f"turbines/{turbine_model}.yaml"
        )
        filename_orbit_config = os.path.join(
            orbit_library_path,
            f"plant/orbit-config-{turbine_model}.yaml"
        )
        filename_floris_config = os.path.join(
            orbit_library_path,
            f"floris/floris_input_{turbine_model}.yaml"
        )
        filename_greenheart_config = os.path.join(
            orbit_library_path,
            f"plant/greenheart_config.yaml"
        )
        filename_hopp_config = os.path.join(
            orbit_library_path,f"plant/hopp_config_wind_wave.yaml"
        )

        self.lcoe, self.lcoh, _, self.hi = run_simulation(
            filename_hopp_config, 
            filename_greenheart_config, 
            filename_turbine_config, 
            filename_orbit_config, 
            filename_floris_config, 
            verbose=False, 
            show_plots=False, 
            save_plots=False,  
            use_profast=True, 
            incentive_option=1, 
            post_processing=True,
            plant_design_scenario=1, 
            output_level=5
        )

    def test_lcoh(self):
        assert self.lcoh == approx(8.120065296802442) #TODO base this test value on something
    def test_lcoe(self):
        assert self.lcoe == approx(0.12863386719193057) # prior to 20240207 value was
        # approx(0.11051228251811765) # TODO base this test value on something
    def test_energy_sources(self):
        expected_annual_energy_hybrid = \
            self.hi.system.annual_energies.wind + self.hi.system.annual_energies.wave
        assert self.hi.system.annual_energies.hybrid == approx(expected_annual_energy_hybrid)

class TestSimulationWindWaveSolar(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    @classmethod
    def setUpClass(self):
        super(TestSimulationWindWaveSolar, self).setUpClass()

        turbine_model = "osw_18MW"
        filename_turbine_config = os.path.join(
            orbit_library_path,
            f"turbines/{turbine_model}.yaml"
        )
        filename_orbit_config = os.path.join(
            orbit_library_path,
            f"plant/orbit-config-{turbine_model}.yaml"
        )
        filename_floris_config = os.path.join(
            orbit_library_path,
            f"floris/floris_input_{turbine_model}.yaml"
        )
        filename_greenheart_config = os.path.join(
            orbit_library_path,
            f"plant/greenheart_config.yaml"
        )
        filename_hopp_config = os.path.join(
            orbit_library_path,
            f"plant/hopp_config_wind_wave_solar.yaml"
        )

        self.lcoe, self.lcoh, _, self.hi  = run_simulation(
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
            plant_design_scenario=9, 
            output_level=5
        )

    def test_lcoh(self):
        assert self.lcoh == approx(12.583155204831298) # prior to 20240207 value was
        # approx(10.823798551850347) #TODO base this test value on something.
        # Currently just based on output at writing.
    def test_lcoe(self):
        assert self.lcoe == approx(0.1284376127848134) # prior to 20240207 value was
        # approx(0.11035426429749774) # TODO base this test value on something.
        # Currently just based on output at writing.
    def test_energy_sources(self):
        expected_annual_energy_hybrid = \
            self.hi.system.annual_energies.wind \
            + self.hi.system.annual_energies.wave \
            + self.hi.system.annual_energies.pv
        assert self.hi.system.annual_energies.hybrid == approx(expected_annual_energy_hybrid)

class TestSimulationWindWaveSolarBattery(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    @classmethod
    def setUpClass(self):
        super(TestSimulationWindWaveSolarBattery, self).setUpClass()

        turbine_model = "osw_18MW"
        filename_turbine_config = os.path.join(
            orbit_library_path,
            f"turbines/{turbine_model}.yaml"
        )
        filename_orbit_config = os.path.join(
            orbit_library_path,
            f"plant/orbit-config-{turbine_model}.yaml"
        )
        filename_floris_config = os.path.join(
            orbit_library_path,
            f"floris/floris_input_{turbine_model}.yaml"
        )
        filename_greenheart_config = os.path.join(
            orbit_library_path,
            f"plant/greenheart_config.yaml"
        )
        filename_hopp_config = os.path.join(
            orbit_library_path,
            f"plant/hopp_config_wind_wave_solar_battery.yaml"
        )

        self.lcoe, self.lcoh, _, self.hi = run_simulation(
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
            plant_design_scenario=10, 
            output_level=5
        )

    def test_lcoh(self):
        assert self.lcoh == approx(16.96997513319437) #TODO base this test value on something.
        # Currently just based on output at writing.

    def test_lcoe(self):
        assert self.lcoe == approx(0.12912145788428933) # TODO base this test value on something.
        # Currently just based on output at writing.

    # def test_energy_sources(self): # TODO why is hybrid energy different than the sum of the parts when battery is being used.
    #     expected_annual_energy_hybrid = \
    #         self.hi.system.annual_energies.wind \
    #         + self.hi.system.annual_energies.wave \
    #         + self.hi.system.annual_energies.pv
    #     assert self.hi.system.annual_energies.hybrid == expected_annual_energy_hybrid

class TestSimulationWindOnshore(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    @classmethod
    def setUpClass(self):
        super(TestSimulationWindOnshore, self).setUpClass()

        turbine_model = "osw_18MW"
        filename_turbine_config = os.path.join(
            orbit_library_path,
            f"turbines/{turbine_model}.yaml"
        )
        filename_orbit_config = os.path.join(
            orbit_library_path,
            f"plant/orbit-config-{turbine_model}.yaml"
        )
        filename_floris_config = os.path.join(
            orbit_library_path,
            f"floris/floris_input_{turbine_model}.yaml"
        )
        filename_greenheart_config = os.path.join(
            orbit_library_path,
            f"plant/greenheart_config_onshore.yaml"
        )
        filename_hopp_config = os.path.join(
            orbit_library_path,
            f"plant/hopp_config.yaml"
        )

        self.lcoe, self.lcoh, _, self.hi = run_simulation(
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
            output_level=5
        )

    def test_lcoh(self):
        assert self.lcoh == approx(7.057994298481547) # TODO base this test value on something

    def test_lcoe(self):
        assert self.lcoe == approx(0.10816180445700445) # TODO base this test value on something
