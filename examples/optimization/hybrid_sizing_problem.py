from pathlib import Path

from collections import OrderedDict, namedtuple
from hopp.simulation.technologies.sites import make_circular_site, make_irregular_site, SiteInfo, locations
from hopp.simulation.hybrid_simulation import HybridSimulation
from hopp.tools.optimization.optimization_problem import OptimizationProblem


site = 'irregular'
location = locations[1]
site_data = None

if site == 'circular':
    site_data = make_circular_site(lat=location[0], lon=location[1], elev=location[2])
elif site == 'irregular':
    site_data = make_irregular_site(lat=location[0], lon=location[1], elev=location[2])
else:
    raise Exception("Unknown site '" + site + "'")

g_file = Path(__file__).parent.parent.parent / "resource_files" / "grid" / "pricing-data-2015-IronMtn-002_factors.csv"

site_info = SiteInfo(site_data, grid_resource_file=g_file)

# set up hybrid simulation with all the required parameters
solar_size_mw = 1
battery_capacity_mwh = 1
interconnection_size_mw = 150

technologies = technologies = {'pv': {
                    'system_capacity_kw': solar_size_mw * 1000,
                },
                'wind': {
                    'num_turbines': 25,
                    'turbine_rating_kw': 2000
                },
                'battery': battery_capacity_mwh * 1000,
                'grid': {
                    'interconnect_kw': interconnection_size_mw * 1000}
                }

# Get resource

# Create model
dispatch_options = {'battery_dispatch': 'heuristic',
                    'n_look_ahead_periods': 24}
hybrid_plant = HybridSimulation(technologies,
                                site_info,
                                dispatch_options=dispatch_options)

# Customize the hybrid plant assumptions here...
hybrid_plant.pv.value('inv_eff', 95.0)
hybrid_plant.pv.value('array_type', 0)

# Build a fixed dispatch array
#   length == n_look_ahead_periods
#   normalized (+) discharge (-) charge
fixed_dispatch = [0.0] * 6
fixed_dispatch.extend([-1.0] * 6)
fixed_dispatch.extend([1.0] * 6)
fixed_dispatch.extend([0.0] * 6)
# Set fixed dispatch
hybrid_plant.battery.dispatch.set_fixed_dispatch(fixed_dispatch)


class HybridSizingProblem(OptimizationProblem):
    """
    Optimize the hybrid system sizing design variables
    """
    def __init__(self,
                 simulation: HybridSimulation) -> None:
        """
        design_variables: nametuple of hybrid technologies each with a namedtuple of design variables
        """
        super().__init__(simulation)
        self.candidate_dict = OrderedDict()

    def _set_design_variables_values(self,
                                     design_variables: namedtuple) -> None:
        for tech_key in design_variables._fields:
            tech_model = getattr(self.simulation, tech_key)
            tech_variables = getattr(design_variables, tech_key)
            for key in tech_variables._fields:
                if hasattr(tech_model, key):
                    setattr(tech_model, key, getattr(tech_variables, key))
                else:
                    tech_model.value(key, getattr(tech_variables, key))

    def _set_simulation_to_candidate(self, candidate):
        pass

    def evaluate_objective(self, design_variables: namedtuple):
        self._set_design_variables_values(design_variables)
        self.simulation.simulate(1)
        evaluation = self.simulation.net_present_values.hybrid
        return evaluation


problem = HybridSizingProblem(hybrid_plant)

pv = namedtuple('pv', ['system_capacity_kw', 'tilt'])
pv_vars = pv(50*1e3, 45)

battery = namedtuple('battery', ['system_capacity_kwh', 'system_capacity_kw', 'system_voltage_volts'])
battery_vars = battery(200*1e3, 50*1e3, 500.0)

Variables = namedtuple('Variables', ['pv', 'battery'])

V = Variables(pv_vars, battery_vars)

problem.evaluate_objective(V)



