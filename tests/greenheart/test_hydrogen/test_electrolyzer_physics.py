from greenheart.simulation.technologies.hydrogen.electrolysis.run_h2_PEM import run_h2_PEM
from pytest import approx
import numpy as np
import greenheart.tools.eco.electrolysis as he_elec


from hopp.utilities.keys import set_nrel_key_dot_env
set_nrel_key_dot_env()

project_life_years = 30

#1: load eco_config
eco_config["electrolyzer"]["rating"]
eco_config["project_parameters"]["grid_connection"]
hopp_results["combined_hybrid_power_production_hopp"]


electrolyzer_physics_results = he_elec.run_electrolyzer_physics(hopp_results, project_life_years, eco_config, wind_resource, design_scenario, show_plots=False, save_plots=False, verbose=False)

def create_inputs():
    lat = []
    lon = []



class TestPEMPhysics():

    def test_run_TX_off_grid_wind(self):
        pass
    def test_run_TX_hybrid_grid(self):
        pass
    def test_run_TX_grid_only(self):
        pass

