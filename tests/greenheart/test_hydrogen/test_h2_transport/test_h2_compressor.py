from greenheart.simulation.technologies.hydrogen.h2_transport.h2_compression import Compressor
from pytest import approx, raises

# test that we get the results we got when the code was received
class TestH2Compressor():
    p_inlet = 20 # bar
    p_outlet = 68 # bar
    flow_rate_kg_d = 9311
    n_compressors = 2
    
    def test_capex(self):
        comp = Compressor(self.p_outlet, self.flow_rate_kg_d, p_inlet=self.p_inlet, n_compressors=self.n_compressors)
        comp.compressor_power()
        total_capex, total_OM = comp.compressor_costs()
        assert total_capex == 680590.3412708649

    def test_opex(self):
        comp = Compressor(self.p_outlet, self.flow_rate_kg_d, p_inlet=self.p_inlet, n_compressors=self.n_compressors)
        comp.compressor_power()
        total_capex, total_OM = comp.compressor_costs()
        assert total_OM == 200014.00244504173

    def test_system_power(self):
        comp = Compressor(self.p_outlet, self.flow_rate_kg_d, p_inlet=self.p_inlet, n_compressors=self.n_compressors)
        comp.compressor_power()
        _motor_rating, total_system_power = comp.compressor_system_power()
        assert total_system_power == 246.27314443197918

    def test_system_power_report(self):
        """
        H2A Hydrogen Delivery Infrastructure Analysis Models and Conventional Pathway Options Analysis Results
        DE-FG36-05GO15032
        Interim Report
        Nexant, Inc., Air Liquide, Argonne National Laboratory, Chevron Technology Venture, Gas Technology Institute, National Renewable Energy Laboratory, Pacific Northwest National Laboratory, and TIAX LLC
        May 2008
        Table 2-21
        """
        n_compressors = 1
        flow_rate_kg_per_day = 126593.0 #[kg/day]
        p_outlet = 1227.0*0.0689476 # convert from psi to bar
        p_inlet = 265.0*0.0689476 # convert from psi to bar
        sizing_safety_factor = 1.0 # default is 1.1, for a 10% oversizing of the compressor
        comp = Compressor(p_outlet, flow_rate_kg_per_day, p_inlet=p_inlet, n_compressors=n_compressors, sizing_safety_factor=sizing_safety_factor)
        comp.compressor_power()
        _motor_rating, total_system_power = comp.compressor_system_power()
        assert total_system_power == 3627.3907562149357

    def test_max_flow_rate_per_compressor(self):
        p_inlet = 20 # bar
        p_outlet = 68 # bar
        flow_rate_kg_d = 2*5.41*24*60**2
        n_compressors = 2
        with raises(ValueError, match=r".* 5\.4 .*"):
            comp = Compressor(p_outlet,flow_rate_kg_d, p_inlet=p_inlet, n_compressors=n_compressors)
    