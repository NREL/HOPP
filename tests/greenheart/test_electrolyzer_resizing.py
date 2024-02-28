import os
from hopp.utilities import load_yaml
import greenheart.tools.plant_sizing_estimation as gh_sizing
dirname = os.path.dirname(__file__)
input_library_path = os.path.join(dirname, "test_hydrogen","input_files","plant")
greenheart_config_filename = os.path.join(input_library_path,"GS_greenheart_config.yaml")
greenheart_config = load_yaml(greenheart_config_filename)

greenheart_config["end_use"]['estimated_cf'] = 0.9
greenheart_config["end_use"]['annual_production_target'] = 1000000
greenheart_config["end_use"]["product"] = 'steel'

def test_electrolyzer_size_from_steel_gridconnected():
    greenheart_config["project_parameters"]["grid_connection"] = True
    greenheart_config["component_sizing"]["hybrid_cf_est"] = 1.0
    gh_test = gh_sizing.run_resizing_estimation(greenheart_config)
    assert gh_test["electrolyzer"]["rating"] == 480

def test_electrolyzer_size_from_steel_offgrid():
    greenheart_config["project_parameters"]["grid_connection"] = False
    greenheart_config["component_sizing"]["hybrid_cf_est"] = 0.492
    gh_test = gh_sizing.run_resizing_estimation(greenheart_config)
    assert gh_test["electrolyzer"]["rating"] == 960

