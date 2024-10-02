# general imports
import os

# # yaml imports
import yaml
from pathlib import Path

# HOPP imports
from greenheart.simulation.greenheart_simulation import (
    run_simulation,
    GreenHeartSimulationConfig,
)
from greenheart.tools.optimization.gc_run_greenheart import run_greenheart

# run the stuff
if __name__ == "__main__":
    # load inputs as needed
    turbine_model = "lbw_6MW"
    filename_turbine_config = "./input/turbines/" + turbine_model + ".yaml"
    filename_floris_config = "./input/floris/floris_input_lbw_6MW.yaml"
    filename_hopp_config = "./input/plant/hopp_config_mn.yaml"
    filename_greenheart_config = "./input/plant/greenheart_config_onshore_mn.yaml"

    config = GreenHeartSimulationConfig(
        filename_hopp_config,
        filename_greenheart_config,
        filename_turbine_config,
        filename_floris_config,
        verbose=True,
        show_plots=False,
        save_plots=True,
        use_profast=True,
        post_processing=True,
        incentive_option=1,
        plant_design_scenario=9,
        output_level=7,
    )

    # for analysis
    prob, config = run_greenheart(config, run_only=True)

    # for optimization
    # prob, config = run_greenheart(config, run_only=False)
    
    lcoe = prob.get_val("lcoe", units="USD/(MW*h)")
    lcoh = prob.get_val("lcoh", units="USD/kg")
    lcos = prob.get_val("lcos", units="USD/t")

    print("LCOE: ", lcoe, "[$/MWh]")
    print("LCOH: ", lcoh, "[$/kg]")
    print("LCOS: ", lcos, "[$/metric-tonne]")
