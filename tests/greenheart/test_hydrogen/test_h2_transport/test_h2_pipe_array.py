from greenheart.simulation.technologies.hydrogen.h2_transport.h2_pipe_array import run_pipe_array, run_pipe_array_const_diam
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
        assert self.capex == 2226256.16387454
    
    def test_opex(self):
        assert self.opex == 26047.197117332118

# TODO check the values in these test, they are gut checked for being slightly above what would be expected for a single distance, but not well determined

class TestPipeArrayMultiSection():
    L = 8                   # Length [km]
    m_dot = 1.5            # Mass flow rate [kg/s] assuming 300 MW -> 1.5 kg/s
    p_inlet = 30            # Inlet pressure [bar]
    p_outlet = 10           # Outlet pressure [bar]
    depth = 80              # depth of pipe [m]


    def test_capex(self):
        assert self.capex == 5129544.170342567
    
    def test_opex(self):
        assert self.opex == 60015.666793008044
    
    capex, opex = run_pipe_array([[L, L]], depth, p_inlet, p_outlet, [[m_dot, m_dot]])
        

class TestPipeArrayMultiSectionConstDiameter():
    L = 8                   # Length [km]
    m_dot = 1.5            # Mass flow rate [kg/s] assuming 300 MW -> 1.5 kg/s
    p_inlet = 30            # Inlet pressure [bar]
    p_outlet = 10           # Outlet pressure [bar]
    depth = 80              # depth of pipe [m]

    capex, opex = run_pipe_array_const_diam([[L, L], [L, L]], depth, p_inlet, p_outlet, [[m_dot, m_dot], [m_dot, m_dot]])
        
    def test_capex(self):
        assert self.capex == 10360272.637584394
    def test_opex(self):
        assert self.opex == 121215.18985973741
    
if __name__ == "__main__":
    test_set = TestPipeArraySingleSection()
    test_set = TestPipeArrayMultiSection()
    test_set = TestPipeArrayMultiSectionConstDiameter()
    