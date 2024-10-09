from greenheart.simulation.technologies.hydrogen.h2_transport.h2_export_pipe import run_pipe_analysis
from pytest import approx

# test that we the results we got when the code was recieved
class TestExportPipeline():
    L = 8                   # Length [km]
    m_dot = 1.5            # Mass flow rate [kg/s] assuming 300 MW -> 1.5 kg/s
    p_inlet = 30            # Inlet pressure [bar]
    p_outlet = 10           # Outlet pressure [bar]
    depth = 80              # depth of pipe [m]
    costs = run_pipe_analysis(L,m_dot,p_inlet,p_outlet,depth)
        
    def test_grade(self):
        assert self.costs["Grade"][0] == "X42"
    
    def test_od(self):
        assert self.costs["Outer diameter (mm)"][0] == 168.28
    
    def test_od(self):
        assert self.costs["Inner diameter (mm)"][0] == 162.74

    def test_schedule(self):
        assert self.costs["Schedule"][0] == "S 5S"

    def test_thickness(self):
        assert self.costs["Thickness (mm)"][0] == 2.77
    
    def test_volume(self):
        assert self.costs["volume [m3]"][0] == 12.213769866246679

    def test_weight(self):
        assert self.costs["weight [kg]"][0] == 95755.95575137396

    def test_material_cost(self):
        assert self.costs["mat cost [$]"][0] == 210663.1026530227
    
    def test_labor_cost(self):
        assert self.costs["labor cost [$]"][0] == 1199293.4563603932

    def test_misc_cost(self):
        assert self.costs["misc cost [$]"][0] == 429838.3943856504
    
    def test_row_cost(self): #ROW = right of way
        assert self.costs["ROW cost [$]"][0] == 365317.5476681454

    def test_total_cost_output(self):
        assert self.costs["total capital cost [$]"][0] == 2205112.501067212

    def test_total_capital_cost_sum(self):
        total_capital_cost = self.costs["mat cost [$]"][0] \
                    + self.costs["labor cost [$]"][0] \
                    + self.costs["misc cost [$]"][0] \
                    + self.costs["ROW cost [$]"][0]

        assert self.costs["total capital cost [$]"][0] == total_capital_cost

    def test_annual_opex(self):
        assert self.costs["annual operating cost [$]"][0] == 0.0117*self.costs["total capital cost [$]"][0]

class TestExportPipelineRegion():
    L = 8                   # Length [km]
    m_dot = 1.5            # Mass flow rate [kg/s] assuming 300 MW -> 1.5 kg/s
    p_inlet = 30            # Inlet pressure [bar]
    p_outlet = 10           # Outlet pressure [bar]
    depth = 80              # depth of pipe [m]
    region = 'GP'           # great plains region
    costs = run_pipe_analysis(L,m_dot,p_inlet,p_outlet,depth,region=region)

    def test_material_cost(self):
        assert self.costs["mat cost [$]"][0] == 210663.1026530227
    
    def test_labor_cost(self):
        assert self.costs["labor cost [$]"][0] == 408438.062500015

    def test_misc_cost(self):
        assert self.costs["misc cost [$]"][0] == 184458.92890187018
    
    def test_row_cost(self): #ROW = right of way
        assert self.costs["ROW cost [$]"][0] == 52426.57591258784

    def test_total_cost_output(self):
        assert self.costs["total capital cost [$]"][0] == 855986.6699674957

class TestExportPipelineOverrides():
    L = 8                   # Length [km]
    m_dot = 1.5            # Mass flow rate [kg/s] assuming 300 MW -> 1.5 kg/s
    p_inlet = 30            # Inlet pressure [bar]
    p_outlet = 10           # Outlet pressure [bar]
    depth = 80              # depth of pipe [m]
    labor_in_mi = 1000
    misc_in_mi = 2000
    row_in_mi = 3000
    mat_in_mi = 4000
    costs = run_pipe_analysis(
        L,
        m_dot,
        p_inlet,
        p_outlet,
        depth,
        labor_in_mi=labor_in_mi,
        misc_in_mi=misc_in_mi,
        row_in_mi=row_in_mi,
        mat_in_mi=mat_in_mi,
    )

    def test_material_cost(self):
        assert self.costs["mat cost [$]"][0] == 124469.9746153248

    def test_labor_cost(self):
        assert self.costs["labor cost [$]"][0] == 31117.4936538312

    def test_misc_cost(self):
        assert self.costs["misc cost [$]"][0] == 62234.9873076624

    def test_row_cost(self): #ROW = right of way
        assert self.costs["ROW cost [$]"][0] == 93352.4809614936

    def test_total_cost_output(self):
        assert self.costs["total capital cost [$]"][0] == 311174.936538312


if __name__ == "__main__":
    test_set = TestExportPipeline()
