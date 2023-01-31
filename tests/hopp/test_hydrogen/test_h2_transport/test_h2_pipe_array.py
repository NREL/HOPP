from hopp.hydrogen.h2_transport.h2_pipe_array import run_pipe_array
from pytest import approx

# test that we the results we got when the code was recieved
class TestPipeArraySingleSection():
    L = 8                   # Length [km]
    m_dot = 1.5            # Mass flow rate [kg/s] assuming 300 MW -> 1.5 kg/s
    p_inlet = 30            # Inlet pressure [bar]
    p_outlet = 10           # Outlet pressure [bar]
    depth = 80              # depth of pipe [m]
    capex, opex = run_pipe_array([[L]], depth, p_inlet, p_outlet, [[m_dot]])
        
    def test_capex(self):
        assert self.capex == 2334049.398178592
    
    def test_opex(self):
        assert self.opex == 27308.377958689525

# TODO check the values in these test, they are gut checked for being slightly above what would be expected for a single distance, but not well determined

class TestPipeArrayMultiSection():
    L = 8                   # Length [km]
    m_dot = 1.5            # Mass flow rate [kg/s] assuming 300 MW -> 1.5 kg/s
    p_inlet = 30            # Inlet pressure [bar]
    p_outlet = 10           # Outlet pressure [bar]
    depth = 80              # depth of pipe [m]

    capex, opex = run_pipe_array([[L, L]], depth, p_inlet, p_outlet, [[m_dot, m_dot]])
        
    def test_capex(self):
        assert self.capex == 4990951.552596819
    
    def test_opex(self):
        assert self.opex == 58394.13316538278
    
if __name__ == "__main__":
    test_set = TestPipeArraySingleSection()
    test_set = TestPipeArrayMultiSection()
    