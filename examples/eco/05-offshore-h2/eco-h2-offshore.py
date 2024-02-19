# general imports
import os

# # yaml imports
import yaml
from yamlinclude import YamlIncludeConstructor
from pathlib import Path

PATH = Path(__file__).parent
YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=PATH / './input/floris/')
YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=PATH / './input/turbines/')

# ORBIT imports
from ORBIT.core.library import initialize_library
initialize_library(os.path.join(os.getcwd(), "./input/"))

# HOPP imports
from greenheart.tools.eco.hybrid_system import run_simulation

# run the stuff
if __name__ == "__main__":
    ## this should result in  5.16 $/kg LCOH

    # load inputs as needed
    turbine_model="osw_18MW"
    filename_orbit_config= "./input/plant/orbit-config-"+turbine_model+".yaml"
    filename_turbine_config = "./input/turbines/"+turbine_model+".yaml"
    filename_floris_config = "./input/floris/floris_input_osw_18MW.yaml"
    filename_hopp_config = "./input/plant/hopp_config.yaml"
    filename_eco_config = "./input/plant/eco_config.yaml"
    
    run_simulation(filename_hopp_config, filename_eco_config, filename_turbine_config, filename_orbit_config, filename_floris_config, verbose=True, show_plots=False, save_plots=True,  use_profast=True, incentive_option=1, plant_design_scenario=1)
