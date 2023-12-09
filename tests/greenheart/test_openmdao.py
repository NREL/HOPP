from greenheart.openmdao import HOPPComponent, TurbineDistanceComponent, BoundaryDistanceComponent
import openmdao.api as om
from hopp.simulation.hopp_interface import HoppInterface
from hopp.utilities import load_yaml
from pytest import approx, raises
from pathlib import Path
import numpy as np

solar_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
wind_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
floris_input_file = Path(__file__).absolute().parent / "inputs" / "floris_input.yaml"
hopp_config_filename = Path(__file__).absolute().parent / "inputs" / "hopp_config.yaml"

# test that we get the results we got when the code was received
class TestTurbineDistanceComponent():

    turbine_x = np.array([2.0, 4.0, 6.0, 2.0, 4.0, 6.0])*100.0
    turbine_y = np.array([2.0, 2.0, 2.0, 4.0, 4.0, 4.0])*100.0
    
    config_dict = load_yaml(hopp_config_filename)
    config_dict["site"]["solar_resource_file"] = solar_resource_file
    config_dict["site"]["wind_resource_file"] = wind_resource_file
    config_dict["technologies"]["wind"]["floris_config"] = floris_input_file
    hi = HoppInterface(config_dict)

    model = om.Group()

    model.add_subsystem('boundary_constraint', BoundaryDistanceComponent(hopp_interface=hi, turbine_x_init=turbine_x, turbine_y_init=turbine_y), promotes=["*"])

    prob = om.Problem(model)
    prob.setup()

    prob.run_model()
    
    def test_distance_inside(self):
        
        assert self.prob["boundary_distance_vec"][0] == 200.0

    # def test_derivative_inside(self):

        
    #     assert total_capex == 680590.3412708649
    
    def test_distance_outside(self):

        self.prob.set_val('boundary_constraint.turbine_x', np.array([-2.0, 4.0, 6.0, 2.0, 4.0, 6.0])*100.0)
        self.prob.set_val('boundary_constraint.turbine_y', np.array([2.0, 2.0, 2.0, 4.0, 4.0, 4.0])*100.0)
        self.prob.run_model()
        
        assert self.prob["boundary_distance_vec"][0] == -200.0

    # def test_derivative_outside(self):
        
    #     assert total_capex == 680590.3412708649
