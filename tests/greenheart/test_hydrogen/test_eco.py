from greenheart.tools.eco.hybrid_system import run_simulation
from pytest import approx

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

# class TestSimulationWind():
#     turbine_model = "osw_18MW"
#     filename_turbine_config = os.path.join(orbit_library_path, f"turbines/{turbine_model}.yaml")
#     filename_orbit_config = os.path.join(orbit_library_path, f"plant/orbit-config-{turbine_model}.yaml")
#     filename_floris_config = os.path.join(orbit_library_path, f"floris/floris_input_{turbine_model}.yaml")
#     filename_eco_config = os.path.join(orbit_library_path, f"plant/eco_config.yaml")
#     filename_hopp_config = os.path.join(orbit_library_path, f"plant/hopp_config.yaml")

#     lcoe, lcoh, _ = run_simulation(filename_hopp_config, filename_eco_config, filename_turbine_config, filename_orbit_config, filename_floris_config, verbose=False, show_plots=False, save_plots=False,  use_profast=True, incentive_option=1, plant_design_scenario=1, output_level=4)

#     def test_lcoh(self):
#         assert self.lcoh == approx(5.70230272215567) # TODO base this test value on something
#     def test_lcoe(self):
#         assert self.lcoe == approx(0.08608837821899562) # TODO base this test value on something
    
# class TestSimulationWindWave():
#     turbine_model = "osw_18MW"
#     filename_turbine_config = os.path.join(orbit_library_path, f"turbines/{turbine_model}.yaml")
#     filename_orbit_config = os.path.join(orbit_library_path, f"plant/orbit-config-{turbine_model}.yaml")
#     filename_floris_config = os.path.join(orbit_library_path, f"floris/floris_input_{turbine_model}.yaml")
#     filename_eco_config = os.path.join(orbit_library_path, f"plant/eco_config.yaml")
#     filename_hopp_config = os.path.join(orbit_library_path, f"plant/hopp_config_wind_wave.yaml")

#     lcoe, lcoh, _ = run_simulation(filename_hopp_config, filename_eco_config, filename_turbine_config, filename_orbit_config, filename_floris_config, verbose=False, show_plots=False, save_plots=False,  use_profast=True, incentive_option=1, plant_design_scenario=1, output_level=4)

#     def test_lcoh(self):
#         assert self.lcoh == approx(6.605069697477719) #TODO base this test value on something
#     def test_lcoe(self):
#         assert self.lcoe == approx(0.10329095004231528) # TODO base this test value on something
    
class TestSimulationWindWaveSolar():
    turbine_model = "osw_18MW"
    filename_turbine_config = os.path.join(orbit_library_path, f"turbines/{turbine_model}.yaml")
    filename_orbit_config = os.path.join(orbit_library_path, f"plant/orbit-config-{turbine_model}.yaml")
    filename_floris_config = os.path.join(orbit_library_path, f"floris/floris_input_{turbine_model}.yaml")
    filename_eco_config = os.path.join(orbit_library_path, f"plant/eco_config.yaml")
    filename_hopp_config = os.path.join(orbit_library_path, f"plant/hopp_config_wind_wave_solar.yaml")

    lcoe, lcoh, _ = run_simulation(filename_hopp_config, 
                                   filename_eco_config, 
                                   filename_turbine_config, 
                                   filename_orbit_config, 
                                   filename_floris_config, 
                                   verbose=False, 
                                   show_plots=False, 
                                   save_plots=True,  
                                   use_profast=True, 
                                   incentive_option=1, 
                                   plant_design_scenario=7, 
                                   output_level=4)

    def test_lcoh(self):
        assert self.lcoh == approx(9.389832471053667) #TODO base this test value on something
    def test_lcoe(self):
        assert self.lcoe == approx(0.0855851556213723) # TODO base this test value on something
    
# run the stuff
if __name__ == "__main__":
    ts = TestSimulation()
    # run_simulation(verbose=True, show_plots=False, save_plots=True,  use_profast=True, incentive_option=1, plant_design_scenario=1)
    # # quit()
    # try:
    #     run_design_options(show_plots=False)
    # except:
    #     print("failed run_design_options")
    # # quit()
    # try:
    #     run_simulation(verbose=True, show_plots=False, save_plots=True,  use_profast=True, incentive_option=1, plant_design_scenario=1)
    # except:
    #     print("failed run_simulation")
    # try:
    #     run_storage_options()
    # except:
    #     print("failed run_storage_options")
    # try:
    #     for i in range(1,8):
    #         run_simulation(verbose=False, show_plots=False, save_plots=True, use_profast=True, incentive_option=1, plant_design_scenario=i)
    # except:
    #     print("failed run_simulation across all scenarios")
    # try:
    #     run_sweeps(simulate=True)
    # except:
    #     print("faild run_sweeps")
    # try:
    #     run_policy_options_storage_types(verbose=False, show_plots=False, save_plots=True,  use_profast=True)
    # except:
    #     print("failed run_policy_options_storage_types")
    # # run_design_options(verbose=False)

    # # # # # process_design_options()

    # try:
    #     run_policy_storage_design_options()
    #     # colors = ["#0079C2", "#00A4E4", "#F7A11A", "#FFC423", "#5D9732", "#8CC63F", "#5E6A71", "#D1D5D8", "#933C06", "#D9531E"]
    #     # colors = ["#0079C2",                         "#FFC423", "#5D9732",            "#5E6A71", "#D1D5D8", "#933C06", "#D9531E"]
    #     # plot_policy_storage_design_options(colors, normalized=True)
    # except:
    #     print("failed run_policy_storage_design_options")
    

    # #### notes
    # - double check capacity factor
    # - fill in detail on slides or in the notes in prep for client meeting
    # - check lower opex electrolyzer cases 3, 4, and 5
    # - make some sort of icon level visual for aggregate slides