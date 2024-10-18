from typing import Tuple

import PySAM.Windpower as windpower
from shapely.geometry import Point

from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.technologies.layout.wind_layout_tools import move_turbines_within_boundary

from examples.optimization.layout_opt.parametrized_optimization_problem import ParametrizedOptimizationProblem

from hopp.simulation.technologies.layout.plot_tools import plot_turbines


class WindSimulationVariables:
    """
    Simulation inputs to be optimized for WindOptimizationProblem

    turb_pos_x is a list of all the x-coordinates
    turb_pos_y is a list of all the y-coordinates
    """
    
    def __init__(self,
                 num_turbines: int,
                 turb_pos: [Point]
                 ) -> None:
        self.num_turbines = num_turbines
        if len(turb_pos) != self.num_turbines:
            # raise ValueError("HybridCandidate creation with wrong number of turbines")
            self.num_turbines = int(len(turb_pos) / 2)
        self.turb_pos_x = [pos.x for pos in turb_pos]
        self.turb_pos_y = [pos.y for pos in turb_pos]


class WindOptimizationProblem(ParametrizedOptimizationProblem):
    """
    Simulation of a wind farm with turbines placed within a site, following spacing requirements
    """
    
    def __init__(self,
                 site_info: SiteInfo,
                 num_turbines: int = 20,
                 min_spacing: float = 200.0,  # [m]
                 penalty_scale: float = .1,
                 max_unpenalized_distance: float = 0.0,  # [m]
                 ) -> None:
        """
        Setup wind simulation
        :param site_info: location, site and resource info
        :param num_turbines: number of turbines to place on site
        :param min_spacing: min spacing between turbines
        :param penalty_scale: tuning parameter
        :param max_unpenalized_distance: tuning parameter
        """
        super().__init__(site_info, num_turbines, min_spacing)
        self.candidate_type = lambda t: WindSimulationVariables(num_turbines, t)
        self.penalty_scale: float = penalty_scale
        self.max_unpenalized_distance: float = max_unpenalized_distance
        
        self._scenario = None
        self._setup_simulation()
    
    def _setup_simulation(self
                          ) -> None:
        """
        Wind simulation
            -> PySAM windpower model
        """
        
        def run_wind_model(windmodel: windpower.Windpower):
            windmodel.Farm.system_capacity = \
                max(windmodel.Turbine.wind_turbine_powercurve_powerout) * len(windmodel.Farm.wind_farm_xCoordinates)
            windmodel.execute(0)
            return windmodel.Outputs.annual_energy
        
        self._scenario = dict()
        wind_model = windpower.default("WindPowerSingleOwner")
        wind_model.Resource.wind_resource_data = self.site_info.wind_resource.data
        self.turb_diam = wind_model.Turbine.wind_turbine_rotor_diameter
        wind_model.Farm.wind_farm_wake_model = 2  # use eddy viscosity wake model
        self._scenario['Wind'] = (wind_model, run_wind_model)
    
    def make_conforming_candidate_and_get_penalty(self,
                                                  candidate: WindSimulationVariables
                                                  ) -> Tuple[WindSimulationVariables, float]:
        """
        Penalize turbines out of bounds while moving them within the boundary
                + always generates a feasible solution
                + provides a smooth surface to descend into a good solution
                - requires tuning of penalty
        """
        candidate.turb_pos_x, candidate.turb_pos_y, squared_error = \
            move_turbines_within_boundary(candidate.turb_pos_x, candidate.turb_pos_y,
                                          self.site_info.polygon.boundary, self.site_info.polygon)
        return candidate, squared_error
    
    def objective(self,
                  candidate: WindSimulationVariables
                  ) -> float:
        """
        Annual energy production of turbine layout less penalty of out-of-bound turbines
        :param candidate:
        :return:
        """
        conforming_candidate, squared_error = self.make_conforming_candidate_and_get_penalty(candidate)
        penalty = max(0.0, self.penalty_scale * max(0.0, squared_error - self.max_unpenalized_distance))
        
        wind_model: windpower.Windpower = self._scenario["Wind"][0]
        wind_model.Farm.wind_farm_xCoordinates = conforming_candidate.turb_pos_x
        wind_model.Farm.wind_farm_yCoordinates = conforming_candidate.turb_pos_y
        score = self._scenario["Wind"][1](wind_model) / 1000
        
        return score - penalty  # , score
    
    @staticmethod
    def plot_candidate(candidate: WindSimulationVariables,
                       color=(0, 1, 0),
                       alpha=.5) -> None:
        plot_turbines(candidate.turb_pos_x, candidate.turb_pos_y,
                      color, alpha)
