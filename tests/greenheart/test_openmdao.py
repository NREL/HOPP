from greenheart.openmdao import HOPPComponent, TurbineDistanceComponent, BoundaryDistanceComponent
from pytest import approx, raises

# test that we get the results we got when the code was received
class TestTurbineDistanceComponent():
    
    def test_distance_inside(self):
        
        assert total_capex == 680590.3412708649

    def test_derivative_inside(self):
        
        assert total_capex == 680590.3412708649
    
    def test_distance_outside(self):
        
        assert total_capex == 680590.3412708649

    def test_derivative_outside(self):
        
        assert total_capex == 680590.3412708649
