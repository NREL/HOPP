from pathlib import Path
from typing import Tuple
import numpy as np
from collections import OrderedDict
from hopp.simulation.technologies.sites import make_circular_site, make_irregular_site, SiteInfo, locations
from hopp.simulation.hybrid_simulation import HybridSimulation
from hopp.simulation.technologies.layout.wind_layout import WindBoundaryGridParameters
from hopp.simulation.technologies.layout.pv_layout import PVGridParameters
from hopp.tools.optimization import DataRecorder
from hopp.tools.optimization.optimization_problem import OptimizationProblem
from hopp.tools.optimization.optimization_driver import OptimizationDriver


site = 'irregular'
location = locations[1]
site_data = None

if site == 'circular':
    site_data = make_circular_site(lat=location[0], lon=location[1], elev=location[2])
elif site == 'irregular':
    site_data = make_irregular_site(lat=location[0], lon=location[1], elev=location[2])
else:
    raise Exception("Unknown site '" + site + "'")

g_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "grid" / "pricing-data-2015-IronMtn-002_factors.csv"
site_info = SiteInfo(site_data, grid_resource_file=g_file)

# set up hybrid simulation with all the required parameters
solar_size_mw = 100
wind_size_mw = 100
interconnection_size_mw = 150

technologies = {'pv': {
                    'system_capacity_kw': solar_size_mw * 1000,
                    'layout_params': PVGridParameters(x_position=0.5,
                                                      y_position=0.5,
                                                      aspect_power=0,
                                                      gcr=0.5,
                                                      s_buffer=2,
                                                      x_buffer=2)
                },
                'wind': {
                    'num_turbines': int(wind_size_mw / 2),
                    'turbine_rating_kw': 2000,
                    'layout_mode': 'boundarygrid',
                    'layout_params': WindBoundaryGridParameters(border_spacing=2,
                                                                border_offset=0.5,
                                                                grid_angle=0.5,
                                                                grid_aspect_power=0.5,
                                                                row_phase_offset=0.5)

                },
                'grid': {
                    'interconnect_kw': interconnection_size_mw * 1000
                }}

# Get resource

# Create model
hybrid_plant = HybridSimulation(technologies, site_info)
# hybrid_plant.plot_layout()
# plt.show()
hybrid_plant.simulate(1)

# Setup Optimization Candidate


class HybridLayoutProblem(OptimizationProblem):
    """
    Optimize the layout of the wind and solar plant
    """

    def __init__(self,
                 simulation: HybridSimulation) -> None:
        """
        Optimize layout with fixed number of turbines and solar capacity

        border_spacing: spacing along border = (1 + border_spacing) * min spacing (0, 100)
        border_offset: turbine border spacing offset as ratio of border spacing  (0, 1)
        grid_angle: turbine inner grid rotation (0, pi) [radians]
        grid_aspect_power: grid aspect ratio [cols / rows] = 2^grid_aspect_power
        row_phase_offset: inner grid phase offset (0,1)  (20% suggested)
        solar_x_position: ratio of solar's x coords to site width (0, 1)
        solar_y_position: ratio of solar's y coords to site height (0, 1)
        solar_aspect_power: aspect ratio of solar to site width = 2^solar_aspect_power
        solar_gcr: gcr ratio of solar patch
        solar_s_buffer: south side buffer ratio (0, 1)
        solar_x_buffer: east and west side buffer ratio (0, 1)
        """
        super().__init__()
        self.simulation = simulation

        self.candidate_dict = OrderedDict({
            # "num_turbines": int,
            "border_spacing": {
                "type": float,
                "prior": {
                    "mu": 5, "sigma": 5
                },
                "min": 0, "max": 100
            },
            "border_offset": {
                "type": float,
                "prior": {
                    "mu": 0.5, "sigma": 2
                },
                "min": 0.0, "max": 1.0
            },
            "grid_angle": {
                "type": float,
                "prior": {
                    "mu": np.pi / 2, "sigma": np.pi
                },
                "min": 0.0, "max": np.pi
            },
            "grid_aspect_power": {
                "type": float,
                "prior": {
                    "mu": 0, "sigma": 3
                },
                "min": -4, "max": 4
            },
            "row_phase_offset": {
                "type": float,
                "prior": {
                    "mu": 0.5, "sigma": .5
                },
                "min": 0.0, "max": 1.0
            },
            "solar_x_position": {
                "type": float,
                "prior": {
                    "mu": .5, "sigma": .5
                },
                "min": 0.0, "max": 1.0
            },
            "solar_y_position": {
                "type": float,
                "prior": {
                    "mu": .5, "sigma": .5
                },
                "min": 0.0, "max": 1.0
            },
            "solar_aspect_power": {
                "type": float,
                "prior": {
                    "mu": 0, "sigma": 3
                },
                "min": -4, "max": 4
            },
            "solar_gcr": {
                "type": float,
                "prior": {
                    "mu": .5, "sigma": .5
                },
                "min": 0.1, "max": 0.9
            },
            "solar_s_buffer": {
                "type": float,
                "prior": {
                    "mu": 4, "sigma": 4
                },
                "min": 0.0, "max": 9.0
            },
            "solar_x_buffer": {
                "type": float,
                "prior": {
                    "mu": 4, "sigma": 4
                },
                "min": 0.0, "max": 9.0
            },
        })

    def _set_simulation_to_candidate(self,
                                     candidate: np.ndarray,
                                     ):
        self.check_candidate(candidate) # scaling
        # assign to named parameters
        wind_layout_ind = 0
        wind_layout = WindBoundaryGridParameters(border_spacing=candidate[wind_layout_ind],
                                                 border_offset=candidate[wind_layout_ind + 1],
                                                 grid_angle=candidate[wind_layout_ind + 2],
                                                 grid_aspect_power=candidate[wind_layout_ind + 3],
                                                 row_phase_offset=candidate[wind_layout_ind + 4])
        solar_layout_ind = 5
        solar_layout = PVGridParameters(x_position=candidate[solar_layout_ind],
                                        y_position=candidate[solar_layout_ind + 1],
                                        aspect_power=candidate[solar_layout_ind + 2],
                                        gcr=candidate[solar_layout_ind + 3],
                                        s_buffer=candidate[solar_layout_ind + 4],
                                        x_buffer=candidate[solar_layout_ind + 5])
        self.simulation.layout.set_layout(wind_kw=wind_size_mw * 1e-3, solar_kw=solar_size_mw * 1e-3, wind_params=wind_layout, pv_params=solar_layout)

        return self.simulation.layout.pv.excess_buffer

    def objective(self,
                  candidate: object
                  ) -> Tuple:
        candidate_conforming, penalty_conforming = self.conform_candidate_and_get_penalty(candidate)
        penalty_layout = self._set_simulation_to_candidate(candidate_conforming)
        self.simulation.simulate(1)
        evaluation = self.simulation.net_present_values.hybrid
        # print(candidate, evaluation)
        score = evaluation - penalty_conforming - penalty_layout
        return score, evaluation, candidate_conforming


if __name__ == '__main__':
    # For this example, the generation_size and max_iterations is very small so that the example completes quickly. 
    # For a more realistic optimization run, set these numbers higher, such as 100 generation_size and 10 max_iterations
    max_interations = 5
    optimizer_config = {
        'method':               'CMA-ES',
        'nprocs':               2,
        'generation_size':      10,
        'selection_proportion': .33,
        'prior_scale':          1.0,
        'prior_params':         {
            "grid_angle": {
                "mu": 0.1
                }
            }
        }

    problem = HybridLayoutProblem(hybrid_plant)
    optimizer = OptimizationDriver(problem, recorder=DataRecorder.make_data_recorder("log"), **optimizer_config)

    score, evaluation, best_solution = optimizer.central_solution()
    print(-1, ' ', score, evaluation)

    while optimizer.num_iterations() < max_interations:
        optimizer.step()
        best_score, best_evaluation, best_solution = optimizer.best_solution()
        central_score, central_evaluation, central_solution = optimizer.central_solution()
        print(optimizer.num_iterations(), ' ', optimizer.num_evaluations(), best_score, best_evaluation)

