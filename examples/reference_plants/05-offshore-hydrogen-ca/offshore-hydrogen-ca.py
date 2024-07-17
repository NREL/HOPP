# general imports
import os

# # yaml imports
import yaml
from yamlinclude import YamlIncludeConstructor
from pathlib import Path

PATH = Path(__file__).parent
YamlIncludeConstructor.add_to_loader_class(
    loader_class=yaml.FullLoader, base_dir=PATH / "./input/floris/"
)
YamlIncludeConstructor.add_to_loader_class(
    loader_class=yaml.FullLoader, base_dir=PATH / "./input/turbines/"
)

# HOPP imports
from greenheart.simulation.greenheart_simulation import (
    run_simulation,
    GreenHeartSimulationConfig,
)
from greenheart.tools.optimization.gc_run_greenheart import run_greenheart

# ORBIT imports
from ORBIT.core.library import initialize_library

initialize_library(os.path.join(os.getcwd(), "./input/"))

# run the stuff
if __name__ == "__main__":
    # load inputs as needed
    turbine_model = "osw_15MW"
    filename_turbine_config = "./input/turbines/" + turbine_model + ".yaml"
    filename_floris_config = "./input/floris/floris_input_osw_15MW.yaml"
    filename_hopp_config = "./input/plant/hopp_config_ca.yaml"
    filename_orbit_config = "./input/plant/orbit-config-" + turbine_model + "-ca.yaml"
    filename_greenheart_config = "./input/plant/greenheart_config_offshore_ca.yaml"

    config = GreenHeartSimulationConfig(
        filename_hopp_config=filename_hopp_config,
        filename_greenheart_config=filename_greenheart_config,
        filename_turbine_config=filename_turbine_config,
        filename_orbit_config=filename_orbit_config,
        filename_floris_config=filename_floris_config,
        verbose=True,
        show_plots=False,
        save_plots=True,
        use_profast=True,
        post_processing=True,
        incentive_option=1,
        plant_design_scenario=1,
        output_level=5,
    )

    # for analysis
    prob, config = run_greenheart(config, run_only=True)

    # for optimization
    # prob, config = run_greenheart(config, run_only=False)
    
    lcoe = prob.get_val("lcoe", units="USD/(MW*h)")
    lcoh = prob.get_val("lcoh", units="USD/kg")

    print("LCOE: ", lcoe, "[$/MWh]")
    print("LCOH: ", lcoh, "[$/kg]")
