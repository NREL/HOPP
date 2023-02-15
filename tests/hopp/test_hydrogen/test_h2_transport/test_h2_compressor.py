from hopp.hydrogen.h2_transport.h2_compression import Compressor
from pytest import approx, raises

# test that we get the results we got when the code was received
class TestH2Compressor():
    p_inlet = 20 # bar
    p_outlet = 68 # bar
    flow_rate_kg_d = 9311
    n_compressors = 2

    comp = Compressor(p_outlet,flow_rate_kg_d, p_inlet=p_inlet, n_compressors=n_compressors)
    comp.compressor_power()
    total_capex,total_OM = comp.compressor_costs()

    def test_capex(self):
        assert self.total_capex == 680590.3412708649

    def test_opex(self):
        assert self.total_OM == 200014.00244504173

    def test_max_flow_rate_per_compressor(self):
        p_inlet = 20 # bar
        p_outlet = 68 # bar
        flow_rate_kg_d = 2*5.41*60**2
        n_compressors = 2
        with raises(ValueError, match=r".* 5\.4 .*"):
            comp = Compressor(p_outlet,flow_rate_kg_d, p_inlet=p_inlet, n_compressors=n_compressors)
    