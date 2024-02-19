from greenheart.tools.eco.hybrid_system import run_simulation
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

YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=os.path.join(orbit_library_path, 'floris/'))
YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=os.path.join(orbit_library_path, 'turbines/'))

initialize_library(orbit_library_path)

class TestSimulationWind(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    @classmethod
    def setUpClass(self):
        super(TestSimulationWind, self).setUpClass()

        turbine_model = "osw_18MW"
        filename_turbine_config = os.path.join(orbit_library_path, f"turbines/{turbine_model}.yaml")
        filename_orbit_config = os.path.join(orbit_library_path, f"plant/orbit-config-{turbine_model}.yaml")
        filename_floris_config = os.path.join(orbit_library_path, f"floris/floris_input_{turbine_model}.yaml")
        filename_eco_config = os.path.join(orbit_library_path, f"plant/eco_config.yaml")
        filename_hopp_config = os.path.join(orbit_library_path, f"plant/hopp_config.yaml")

        self.lcoe, self.lcoh, _, self.hi = run_simulation(filename_hopp_config, 
                                                          filename_eco_config, 
                                                          filename_turbine_config, 
                                                          filename_orbit_config, 
                                                          filename_floris_config, 
                                                          verbose=False, 
                                                          show_plots=False, 
                                                          save_plots=False,  
                                                          use_profast=True, 
                                                          incentive_option=1, 
                                                          plant_design_scenario=1, 
                                                          output_level=5)

    def test_lcoh(self):
        assert self.lcoh == approx(5.70230272215567) # TODO base this test value on something
    def test_lcoe(self):
        assert self.lcoe == approx(0.08062438550749312) # TODO base this test value on something
    def test_energy_sources(self):
        expected_annual_energy_hybrid = self.hi.system.annual_energies.wind
        assert self.hi.system.annual_energies.hybrid == expected_annual_energy_hybrid

class TestSimulationWindWave(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    @classmethod
    def setUpClass(self):
        super(TestSimulationWindWave, self).setUpClass()

        turbine_model = "osw_18MW"
        filename_turbine_config = os.path.join(orbit_library_path, f"turbines/{turbine_model}.yaml")
        filename_orbit_config = os.path.join(orbit_library_path, f"plant/orbit-config-{turbine_model}.yaml")
        filename_floris_config = os.path.join(orbit_library_path, f"floris/floris_input_{turbine_model}.yaml")
        filename_eco_config = os.path.join(orbit_library_path, f"plant/eco_config.yaml")
        filename_hopp_config = os.path.join(orbit_library_path, f"plant/hopp_config_wind_wave.yaml")

        self.lcoe, self.lcoh, _, self.hi = run_simulation(filename_hopp_config, 
                                                 filename_eco_config, 
                                                 filename_turbine_config, 
                                                 filename_orbit_config, 
                                                 filename_floris_config, 
                                                 verbose=False, 
                                                 show_plots=False, 
                                                 save_plots=False,  
                                                 use_profast=True, 
                                                 incentive_option=1, 
                                                 plant_design_scenario=1, 
                                                 output_level=5)

    def test_lcoh(self):
        assert self.lcoh == approx(7.01523894727639) #TODO base this test value on something
    def test_lcoe(self):
        assert self.lcoe == approx(0.09962345766436045) # prior to 20240207 value was approx(0.11051228251811765) # TODO base this test value on something
    def test_energy_sources(self):
        expected_annual_energy_hybrid = self.hi.system.annual_energies.wind + self.hi.system.annual_energies.wave
        assert self.hi.system.annual_energies.hybrid == expected_annual_energy_hybrid

class TestSimulationWindWaveSolar(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    @classmethod
    def setUpClass(self):
        super(TestSimulationWindWaveSolar, self).setUpClass()

        turbine_model = "osw_18MW"
        filename_turbine_config = os.path.join(orbit_library_path, f"turbines/{turbine_model}.yaml")
        filename_orbit_config = os.path.join(orbit_library_path, f"plant/orbit-config-{turbine_model}.yaml")
        filename_floris_config = os.path.join(orbit_library_path, f"floris/floris_input_{turbine_model}.yaml")
        filename_eco_config = os.path.join(orbit_library_path, f"plant/eco_config.yaml")
        filename_hopp_config = os.path.join(orbit_library_path, f"plant/hopp_config_wind_wave_solar.yaml")

        self.lcoe, self.lcoh, _, self.hi = run_simulation(filename_hopp_config, 
                                    filename_eco_config, 
                                    filename_turbine_config, 
                                    filename_orbit_config, 
                                    filename_floris_config, 
                                    verbose=False, 
                                    show_plots=False, 
                                    save_plots=False,  
                                    use_profast=True, 
                                    incentive_option=1, 
                                    plant_design_scenario=7, 
                                    output_level=5)

    def test_lcoh(self):
        assert self.lcoh == approx(10.765330694539326) # prior to 20240207 value was approx(10.823798551850347) #TODO base this test value on something. Currently just based on output at writing.
    def test_lcoe(self):
        assert self.lcoe == approx(0.09951895075981732) # prior to 20240207 value was approx(0.11035426429749774) # TODO base this test value on something. Currently just based on output at writing.
    def test_energy_sources(self):
        expected_annual_energy_hybrid = self.hi.system.annual_energies.wind + self.hi.system.annual_energies.wave + self.hi.system.annual_energies.pv
        assert self.hi.system.annual_energies.hybrid == expected_annual_energy_hybrid

class TestSimulationWindWaveSolarBattery(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    @classmethod
    def setUpClass(self):
        super(TestSimulationWindWaveSolarBattery, self).setUpClass()

        turbine_model = "osw_18MW"
        filename_turbine_config = os.path.join(orbit_library_path, f"turbines/{turbine_model}.yaml")
        filename_orbit_config = os.path.join(orbit_library_path, f"plant/orbit-config-{turbine_model}.yaml")
        filename_floris_config = os.path.join(orbit_library_path, f"floris/floris_input_{turbine_model}.yaml")
        filename_eco_config = os.path.join(orbit_library_path, f"plant/eco_config.yaml")
        filename_hopp_config = os.path.join(orbit_library_path, f"plant/hopp_config_wind_wave_solar_battery.yaml")

        self.lcoe, self.lcoh, _, self.hi = run_simulation(filename_hopp_config, 
                                    filename_eco_config, 
                                    filename_turbine_config, 
                                    filename_orbit_config, 
                                    filename_floris_config, 
                                    verbose=False, 
                                    show_plots=False, 
                                    save_plots=False,  
                                    use_profast=True, 
                                    incentive_option=1, 
                                    plant_design_scenario=7, 
                                    output_level=5)

    def test_lcoh(self):
        assert self.lcoh == approx(11.31588977975476) #TODO base this test value on something. Currently just based on output at writing.
    def test_lcoe(self):
        assert self.lcoe == approx(0.10813686278768041) # TODO base this test value on something. Currently just based on output at writing.
    def test_energy_sources(self):
        expected_annual_energy_hybrid = self.hi.system.annual_energies.wind + self.hi.system.annual_energies.wave + self.hi.system.annual_energies.pv
        assert self.hi.system.annual_energies.hybrid == expected_annual_energy_hybrid
