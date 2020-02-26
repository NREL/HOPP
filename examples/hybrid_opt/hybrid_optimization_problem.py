from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point

from hybrid.site_info import SiteInfo
from examples.optimization_problem import OptimizationProblem

import PySAM.Pvwattsv7 as pvwatts
import PySAM.Windpower as windpower


class HybridOptimizationProblem(OptimizationProblem):
    
    def __init__(self,
                 site_info: SiteInfo,
                 num_turbines: int,
                 system_capacity_mw: float,
                 min_spacing: float = 200.,
                 penalty_scale: float = .1,
                 max_unpenalized_distance: float = 0.0,  # [m]
                 ) -> None:
        super().__init__(site_info, num_turbines, min_spacing)
        self.penalty_scale: float = penalty_scale
        self.max_unpenalized_distance: float = max_unpenalized_distance

        self.scenario = None
        self.solar_gcr_loss_multiplier = None
        self.turb_diam = None
        self.setup_models(self.site_info)

        self.solar_size_kw: float = system_capacity_mw * 1000 - self.num_turbines * \
                                    max(self.scenario['Wind'][0].Turbine.wind_turbine_powercurve_powerout)
        # for optimizing mix of solar and wind
        # self.solar_base_size = None
        # self.solar_base_size_multiplier = None

    def setup_models(self, site_info):
        def run_wind_model(windmodel: windpower.Windpower):
            windmodel.Farm.system_capacity = \
                max(windmodel.Turbine.wind_turbine_powercurve_powerout) * len(windmodel.Farm.wind_farm_xCoordinates)
            windmodel.execute()
            return windmodel.Outputs.annual_energy

        def run_pv_model(pvmodel: pvwatts.Pvwattsv7):
            loss_multipler = pvmodel.AdjustmentFactors.constant
            # aep = pvmodel.SystemDesign.system_capacity * self.solar_base_size_multiplier * gcr_loss_multipler
            aep = pvmodel.SystemDesign.system_capacity * loss_multipler
            return aep

        self.scenario = dict()
        wind_model = windpower.default("WindPowerSingleOwner")
        wind_model.Resource.wind_resource_data = site_info.wind_resource.data
        self.turb_diam = wind_model.Turbine.wind_turbine_rotor_diameter

        solar_model = pvwatts.default("PVWattsSingleOwner")
        solar_model.SolarResource.solar_resource_data = site_info.solar_resource.data
        # annual energy scales linearly with system_capacity
        # self.solar_base_size = solar_model.SystemDesign.system_capacity
        # solar_model.execute()
        # self.solar_base_size_multiplier = solar_model.Outputs.annual_energy / self.solar_base_size

        self.scenario['Wind'] = (wind_model, run_wind_model)
        self.scenario['Solar'] = (solar_model, run_pv_model)
        self.precalculate_gcr_losses()

    def precalculate_gcr_losses(self):
        """
        Memoize the gcr loss relative to the smallest possible gcr of 0.01 in an array indexed such that
        if gcr = x >= 0.01, then apply self.gcr_shading_loss_multiplier[int(x * 100) - 1]
        """
        solar_model = self.scenario['Solar'][0]
        solar_model.SystemDesign.gcr = 0.01
        solar_model.execute()
        smallest_gcr_energy = solar_model.Outputs.annual_energy
        self.solar_gcr_loss_multiplier = []
        for i in range(1, 100, 1):
            solar_model.SystemDesign.gcr = i / 100
            solar_model.execute()
            self.solar_gcr_loss_multiplier.append(solar_model.Outputs.annual_energy / smallest_gcr_energy)

    def make_conforming_candidate_and_get_penalty(self, candidate: np.ndarray) -> Tuple[np.ndarray, float, float]:

        conforming_candidate: np.ndarray = np.array(candidate)
        # conforming_positions = []
        site_polygon = self.site_info.polygon
        boundary = site_polygon.boundary
        valid_region = site_polygon.buffer(1e-8)
        error: float = 0.0
        squared_error: float = 0.0
        
        # move all turbines inside the site boundary
        for i in range(self.num_turbines):
            point = Point(candidate[i], candidate[self.num_turbines + i])
            distance = valid_region.distance(point)
            
            if distance > 0:
                point = boundary.interpolate(boundary.project(point))
                
                error += distance
                squared_error += distance * distance
            
            conforming_candidate[i] = point.x
            conforming_candidate[i + self.num_turbines] = point.y

        return conforming_candidate, error, squared_error
    
    def make_conforming_candidate(self, candidate: np.ndarray) -> np.ndarray:
        return self.make_conforming_candidate_and_get_penalty(candidate)[0]
    
    def objective(self, candidate: np.ndarray) -> float:
        """
        Candidate contains turbine coordinates for first 2 * self.num_turbine entries, then pv loss ratio
        :param candidate:
        :return:
        """

        # move turbines that are outside the boundary slightly inside the boundary
        conforming_candidate, error, squared_error = self.make_conforming_candidate_and_get_penalty(candidate)
        penalty = max(0.0, self.penalty_scale * max(0.0, squared_error - self.max_unpenalized_distance))

        wind_model: windpower.Windpower = self.scenario['Wind'][0]
        wind_model.Farm.wind_farm_xCoordinates = conforming_candidate[0:self.num_turbines]
        wind_model.Farm.wind_farm_yCoordinates = conforming_candidate[self.num_turbines:self.num_turbines*2]
        score = -self.scenario['Wind'][1](wind_model)/1000

        solar_model: pvwatts.Pvwattsv7 = self.scenario['Solar'][0]
        solar_model.AdjustmentFactors.constant = candidate[-1]
        score -= self.scenario['Solar'][1](solar_model)/1000

        return score - penalty  # , score
    
    def plot_candidate(self, candidate, color=(0, 1, 0), alpha=.5) -> None:
        plt.plot(candidate[0:self.num_turbines], candidate[self.num_turbines:self.num_turbines*2], 'o', color=color, alpha=alpha)
