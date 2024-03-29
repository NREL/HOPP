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

# ORBIT imports
from ORBIT.core.library import initialize_library
initialize_library(os.path.join(os.getcwd(), "./input/"))

# run the stuff
if __name__ == "__main__":
    # load inputs as needed
    turbine_model = "osw_15MW"
    filename_turbine_config = "./input/turbines/" + turbine_model + ".yaml"
    filename_floris_config = "./input/floris/floris_input_osw_15MW.yaml"
    filename_hopp_config = "./input/plant/hopp_config_ny.yaml"
    filename_orbit_config= "./input/plant/orbit-config-"+turbine_model+"-ny.yaml"
    filename_greenheart_config = "./input/plant/greenheart_config_offshore_ny.yaml"

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
        post_processing=False,
        incentive_option=1,
        plant_design_scenario=1,
        output_level=5,
    )

    config.hopp_config["technologies"]["pv"]["fin_model"]["system_costs"]["om_fixed"][
        0
    ] = config.hopp_config["config"]["cost_info"]["pv_om_per_kw"]
    config.hopp_config["technologies"]["battery"]["fin_model"]["system_costs"][
        "om_batt_fixed_cost"
    ] = config.hopp_config["config"]["cost_info"]["battery_om_per_kw"]

    lcoe, lcoh, _, _ = run_simulation(config)

    print("LCOE: ", lcoe * 1e3, "[$/MWh]")
    print("LCOH: ", lcoh, "[$/kg]")