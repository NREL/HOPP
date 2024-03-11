from pytest import approx, raises, fixture
from pathlib import Path
import numpy as np
import copy

import openmdao.api as om

from hopp.simulation import HoppInterface
from hopp.utilities import load_yaml

from greenheart.tools.optimization.openmdao import HOPPComponent, TurbineDistanceComponent, BoundaryDistanceComponent

solar_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
wind_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
floris_input_file = Path(__file__).absolute().parent / "inputs" / "floris_input.yaml"
hopp_config_filename = Path(__file__).absolute().parent / "inputs" / "hopp_config.yaml"

class TestBoundaryDistanceComponent():

    turbine_x = np.array([2.0, 4.0, 6.0, 2.0, 4.0, 6.0])*100.0
    turbine_y = np.array([2.0, 2.0, 2.0, 4.0, 4.0, 4.0])*100.0
    
    config_dict = load_yaml(hopp_config_filename)
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

class TestTurbineDistanceComponent():

    turbine_x = np.array([2.0, 4.0, 6.0])*100.0
    turbine_y = np.array([2.0, 2.0, 4.0])*100.0

    model = om.Group()

    model.add_subsystem('boundary_constraint', TurbineDistanceComponent(turbine_x_init=turbine_x, turbine_y_init=turbine_y), promotes=["*"])

    prob = om.Problem(model)
    prob.setup()

    prob.run_model()
    
    def test_distance_between_turbines(self, subtests):
        expected_distances = np.array([200.0, np.sqrt(400**2 + 200**2), np.sqrt(2*200**2)])
        for i in range(len(self.turbine_x)):
            with subtests.test(f"for element {i}"):
                assert self.prob["spacing_vec"][i] == expected_distances[i]

class TestHoppComponent():

    def setup(self):
        self.turbine_x = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])*100.0
        self.turbine_y = np.array([2.0, 2.0, 4.0, 4.0, 8.0, 8.0])*100.0

        self.hybrid_config_dict = load_yaml(hopp_config_filename)
        self.hybrid_config_dict["site"]["solar_resource_file"] = solar_resource_file
        self.hybrid_config_dict["site"]["solar"] = "true"
        self.hybrid_config_dict["site"]["wind_resource_file"] = wind_resource_file
        self.hybrid_config_dict["technologies"]["wind"]["floris_config"] = floris_input_file

        self.design_variables = ["pv_capacity_kw", "turbine_x"]

        technologies = self.hybrid_config_dict["technologies"]
        self.solar_wind_hybrid = {key: technologies[key] for key in ('pv', 'wind', 'grid')}
        self.hybrid_config_dict["technologies"] = self.solar_wind_hybrid
        self.hi = HoppInterface(self.hybrid_config_dict)

        model = om.Group()

        model.add_subsystem('hopp', HOPPComponent(hi=self.hi, verbose=False, turbine_x_init=self.turbine_x, turbine_y_init=self.turbine_y, design_variables=self.design_variables), promotes=["*"])

        self.prob = om.Problem(model)
        self.prob.setup()

        self.prob.run_model()
    
    def test_inputs(self, subtests):
        with subtests.test("turbine_x"):
            assert self.prob.get_val('turbine_x')[0] == approx(self.turbine_x[0])
        with subtests.test("pv_capacity_kw"):
            assert self.prob.get_val('pv_capacity_kw')[0] == approx(self.hybrid_config_dict["technologies"]["pv"]["system_capacity_kw"])

    def test_changes(self, subtests):
        new_pv_capacity_kw = 50
        new_x = copy.deepcopy(self.turbine_x)
        new_x[0] = 0.0
        self.prob.set_val("pv_capacity_kw", new_pv_capacity_kw)
        self.prob.set_val("turbine_x", new_x)
        self.prob.run_model()
        with subtests.test("turbine_x"):
            assert self.prob.get_val('turbine_x')[0] == approx(new_x[0])
        with subtests.test("pv_capacity_kw_new"):
            assert self.prob.get_val('pv_capacity_kw')[0] == new_pv_capacity_kw
