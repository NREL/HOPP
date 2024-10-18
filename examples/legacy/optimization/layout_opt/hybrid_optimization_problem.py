from __future__ import annotations
from math import floor
from pathlib import Path

import PySAM.Pvwattsv8 as pvwatts
import PySAM.Windpower as windpower
import numpy as np
from shapely.geometry import (
    LineString,
    Point,
    )

from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.technologies.layout.flicker_mismatch import module_width, module_height, FlickerMismatch, modules_per_string
from hopp.simulation.technologies.layout.wind_layout_tools import move_turbines_within_boundary
from hopp.utilities.log import opt_logger as logger


from examples.optimization.layout_opt.parametrized_optimization_problem import ParametrizedOptimizationProblem

from hopp.simulation.technologies.layout.pv_layout_tools import (
    calculate_max_hybrid_aep,
    get_flicker_loss_multiplier,
    )
from hopp.simulation.technologies.layout.plot_tools import plot_turbines


class HybridSimulationVariables:
    """
    Simulation inputs to be optimized for HybridOptimizationProblem

    turb_pos_x is a list of all the x-coordinates
    turb_pos_y is a list of all the y-coordinates
    solar_areas is a list of tuples describing each separate solar area
    """
    
    def __init__(self,
                 turb_pos: tuple[Point],
                 solar_areas: tuple[tuple[float, int, list]]
                 ) -> None:
        
        class SolarArea:
            """
            Contains the location and system information of a given solar area

            gcr is the ground-coverage ratio
            num_modules is the number of solar panels in the entire area
            strands is a list of tuples describing each solar north-south row:
                the number of modules, the length, and the shapely LineString
            """
            
            def __init__(self,
                         gcr: float,
                         num_modules: int,
                         strands: list):
                self.gcr = gcr
                self.num_modules = num_modules
                self.strands: list[tuple[int, float, LineString]] = strands  # num modules, length, segment

            def __repr__(self):
                return str({'gcr': self.gcr, 'num_modules': self.num_modules, 'strands': self.strands})

        self.turb_pos_x = [pos.x for pos in turb_pos]
        self.turb_pos_y = [pos.y for pos in turb_pos]
        
        self.solar_areas = []
        if type(solar_areas[0]) is float:
            solar_areas = (solar_areas,)
        for area in solar_areas:
            self.solar_areas.append(SolarArea(area[0], area[1], area[2]))

    def __repr__(self):
        return 'turb_x\n{}\nturb_y\n{}\nsolar\n{}'.format(self.turb_pos_x, self.turb_pos_y, self.solar_areas)


class HybridOptimizationProblem(ParametrizedOptimizationProblem):
    """
    Simulation of a hybrid power plant with wind and solar following spacing requirements
    """
    
    def __init__(self,
                 site_info: SiteInfo,
                 num_turbines: int,
                 solar_capacity: float,
                 min_spacing: float = 200.,
                 penalty_scale: float = .1,
                 max_unpenalized_distance: float = 0.0,  # [m]
                 min_gcr: float = .2,
                 max_gcr: float = .8,
                 min_spacing_between_solar_and_wind: float = 100,  # [m]
                 module_power: float = .321
                 ) -> None:
        """
        Setup turbine flicker data and hybrid simulations
        :param site_info: location, site and resource info
        :param num_turbines: number of turbines to place on site
        :param min_spacing: min spacing between turbines
        :param penalty_scale: tuning parameter
        :param max_unpenalized_distance: tuning parameter
        :param min_gcr: minimum gcr for the solar panels
        :param max_gcr: max gcr
        :param min_spacing_between_solar_and_wind:
        :param module_power: [kw] generation capacity per solar module
        """
        super().__init__(site_info, num_turbines, min_spacing)
        self.candidate_type = HybridSimulationVariables
        self.solar_capacity_kw: float = solar_capacity
        self.penalty_scale: float = penalty_scale
        self.max_unpenalized_distance: float = max_unpenalized_distance
        self.min_gcr: float = min_gcr
        self.max_gcr: float = max_gcr
        self.min_spacing_between_solar_and_wind: float = min_spacing_between_solar_and_wind
        self.module_power: float = module_power
        
        self.turb_diam = 77
        self.module_width: float = module_width
        self.module_height: float = module_height
        self.max_num_modules: int = int(floor(self.solar_capacity_kw / self.module_power))
        self.min_strand_length: int = modules_per_string
        
        self._scenario = None
        self._solar_size_aep_multiplier = None
        self._solar_gcr_loss_multiplier = dict()
        self._flicker_data = self._load_flicker_data()
        self._setup_simulation()
        
        logger.info("Created HybridOptimizationProblem")
    
    def _load_flicker_data(self):
        """
        Load the file containing the flicker heat map of a single turbine for the lat, lon.
        This flicker heat map was generated separately using flicker_mismatch.py for a given lat, lon. It was computed
        with these settings:
            `diam_mult` identifies how many diameters to the left, right and above of the turbine is in the grid
                while 4 diameters to the bottom are inside the grid
            `flicker_diam` identifies the size of the turbine in the flicker model and how to scale the results to =
                turbines of different sizes
            `steps_per_hour` is the timestep interval of shadow calculation
            `angles_per_step` is how many different angles of the blades are calculated per timestep
        :return: tuple:
                    (turbine diameter,
                     tuple of turbine location x, y indices,
                     2-D array containing flicker loss multiplier at x, y coordinates (0-1, 0 is no loss),
                     x_coordinates of grid,
                     y_coordinates of grid)
        """
        flicker_diam = 70  # meters, of the turbine used in flicker modeling
        steps_per_hour = 4
        angles_per_step = 12
        data_path = Path(__file__).parent.parent.parent.parent / "hopp" / "layout" / "flicker_data"
        flicker_path = data_path / "{}_{}_{}_{}_shadow.txt".format(self.site_info.data['lat'],
                                                                   self.site_info.data['lon'],
                                                                   steps_per_hour, angles_per_step)
        try:
            flicker_heatmap = np.loadtxt(flicker_path)
        except OSError:
            raise NotImplementedError("Flicker look up table for project's lat and lon does not exist.")
        
        bounds = FlickerMismatch.get_turb_site(flicker_diam).bounds
        _, heatmap_template = FlickerMismatch._setup_heatmap_template(bounds, module_width, module_height)
        turb_x_ind, turb_y_ind = FlickerMismatch.get_turb_pos_indices(heatmap_template)
        
        return flicker_diam, (turb_x_ind, turb_y_ind), flicker_heatmap, heatmap_template[1], heatmap_template[2]
    
    def _setup_simulation(self
                          ) -> None:
        """
        Wind simulation
            -> PySAM windpower model

        Solar simulation
            -> Surrogate model of PySAM Pvwatts model since the AEP scales linearly and independently
            w.r.t solar capacity and gcr
        """
        
        def run_wind_model(windmodel: windpower.Windpower):
            windmodel.Farm.system_capacity = \
                max(windmodel.Turbine.wind_turbine_powercurve_powerout) * len(windmodel.Farm.wind_farm_xCoordinates)
            windmodel.execute(0)
            return windmodel.Outputs.annual_energy
        
        def run_pv_model(pvmodel: pvwatts.Pvwattsv8):
            cap = pvmodel.SystemDesign.system_capacity
            gcr = pvmodel.SystemDesign.gcr
            est = cap * self._solar_size_aep_multiplier * self.solar_gcr_loss_multiplier(gcr)
            # pvmodel.execute()
            # rl = pvmodel.Outputs.annual_energy
            # err = (rl - est)/rl
            # if err > 0.05:
            #     print("High approx error found with {} kwh and {} gcr of {}".format(cap, gcr, err))
            return est
        
        # create wind model
        self._scenario = dict()
        wind_model = windpower.default("WindPowerSingleOwner")
        wind_model.Resource.wind_resource_data = self.site_info.wind_resource.data
        self.turb_diam = wind_model.Turbine.wind_turbine_rotor_diameter
        wind_model.Farm.wind_farm_wake_model = 2  # use eddy viscosity wake model
        
        self._scenario['Wind'] = (wind_model, run_wind_model)
        
        # create pv model
        solar_model = pvwatts.default("PVWattsSingleOwner")
        solar_model.SolarResource.solar_resource_data = self.site_info.solar_resource.data
        solar_model.SystemDesign.array_type = 2  # single-axis tracking
        solar_model.SystemDesign.tilt = 0
        
        # setup surrogate
        solar_model.execute(0)
        self._solar_size_aep_multiplier = solar_model.Outputs.annual_energy / solar_model.SystemDesign.system_capacity
        
        solar_model.SystemDesign.gcr = 0.01  # lowest possible gcr
        solar_model.SystemDesign.system_capacity = 1
        solar_model.execute(0)
        if solar_model.Outputs.annual_energy > 0:
            self._solar_gcr_loss_multiplier['unit'] = solar_model.Outputs.annual_energy
        else:
            raise RuntimeError("Solar GCR Loss Multiplier: Setup failed due to 0 for unit value")
        
        self._scenario['Solar'] = (solar_model, run_pv_model)
        
        # estimate max AEP
        self.upper_bounds = calculate_max_hybrid_aep(self.site_info, self.num_turbines, self.solar_capacity_kw)
        
        logger.info("Setup Wind and Solar models. Max AEP is {} for wind, {} solar, {} total".format(
            self.upper_bounds['wind'], self.upper_bounds['solar'], self.upper_bounds['total']
            ))
    
    def solar_gcr_loss_multiplier(self, gcr):
        """
        Memoize the gcr loss relative to the smallest possible gcr of 0.01 in an array indexed such that
        if gcr = x >= 0.01, then apply self.gcr_shading_loss_multiplier[int(x * 100) - 1]
        """
        if gcr < 0 or gcr > 1:
            raise ValueError("gcr must be [0, 1]")
        gcr = max(0.01, gcr)
        gcr_str = str(round(gcr, 2))
        if gcr_str in self._solar_gcr_loss_multiplier.keys():
            return self._solar_gcr_loss_multiplier[gcr_str]
        
        solar_model: pvwatts.Pvwattsv8 = self._scenario['Solar'][0]
        old_cap, old_gcr = solar_model.SystemDesign.system_capacity, solar_model.SystemDesign.gcr
        solar_model.SystemDesign.system_capacity = 1
        solar_model.SystemDesign.gcr = gcr
        solar_model.execute(0)
        self._solar_gcr_loss_multiplier[gcr_str] = \
            solar_model.Outputs.annual_energy / self._solar_gcr_loss_multiplier['unit']
        solar_model.SystemDesign.system_capacity = old_cap
        solar_model.SystemDesign.gcr = old_gcr
        return self._solar_gcr_loss_multiplier[gcr_str]
    
    def make_conforming_candidate_and_get_penalty(self,
                                                  candidate: HybridSimulationVariables
                                                  ) -> Tuple[HybridSimulationVariables, float]:
        """
        Penalize turbines out of bounds while moving them within the boundary
                + always generates a feasible solution
                + provides a smooth surface to descend into a good solution
                - requires tuning of penalty
        """
        candidate.turb_pos_x, candidate.turb_pos_y, squared_error = \
            move_turbines_within_boundary(candidate.turb_pos_x, candidate.turb_pos_y,
                                          self.site_info.polygon.boundary, self.site_info.polygon)
        
        logger.info("Made conforming candidate {}".format(vars(candidate)))
        return candidate, squared_error
    
    def compute_objective(self,
                          candidate: HybridSimulationVariables
                          ):
        """
        Annual energy production of wind and solar with the given layout
        """
        
        conforming_candidate, squared_error = self.make_conforming_candidate_and_get_penalty(candidate)
        penalty = max(0.0, self.penalty_scale * max(0.0, squared_error - self.max_unpenalized_distance))
        
        # wind
        wind_model: windpower.Windpower = self._scenario['Wind'][0]
        wind_model.Farm.wind_farm_xCoordinates = conforming_candidate.turb_pos_x
        wind_model.Farm.wind_farm_yCoordinates = conforming_candidate.turb_pos_y
        wind_score = self._scenario['Wind'][1](wind_model) / 1000
        
        # get solar capacity after flicker losses
        net_solar_capacities = []
        flicker_losses = 1
        for area in conforming_candidate.solar_areas:
            solar_capacity = self.module_power * area.num_modules
            # gcr_loss = self.solar_gcr_loss_multiplier(area.gcr)
            # solar_capacity *= gcr_loss
            flicker_loss = get_flicker_loss_multiplier(self._flicker_data,
                                                       conforming_candidate.turb_pos_x,
                                                       conforming_candidate.turb_pos_y,
                                                       self.turb_diam,
                                                       (self.module_width, self.module_height),
                                                       primary_strands=area.strands)
            solar_capacity *= flicker_loss
            flicker_losses *= flicker_loss
            net_solar_capacities.append(solar_capacity)
        
        total_solar_capacity = sum(net_solar_capacities)
        if total_solar_capacity == 0:
            return 0
        
        # solar capacity after gcr losses
        avg_gcr = np.dot(np.array(net_solar_capacities) / total_solar_capacity,
                         np.array([area.gcr for area in conforming_candidate.solar_areas]))
        
        solar_model: pvwatts.Pvwattsv8 = self._scenario['Solar'][0]
        solar_model.SystemDesign.gcr = avg_gcr
        solar_model.SystemDesign.system_capacity = float(total_solar_capacity)
        solar_score = self._scenario['Solar'][1](solar_model) / 1000
        score = wind_score + solar_score
        
        # report losses
        gcr_losses = (1 - self.solar_gcr_loss_multiplier(avg_gcr)) * 100
        logger.info("Evaluative objective with score {} = {} w + {} s. "
                    "Wake losses {}%, gcr losses {}%, flicker losses {}%".format(score - penalty,
                                                                                 wind_score,
                                                                                 solar_score,
                                                                                 wind_model.Outputs.wake_losses,
                                                                                 gcr_losses,
                                                                                 (1 - flicker_losses) * 100))
        return score - penalty, score, wind_score, solar_score, wind_model.Outputs.wake_losses, gcr_losses, (
                    1 - flicker_losses) * 100

    def objective(self,
                  candidate: HybridSimulationVariables
                  ) -> float:
        return self.compute_objective(candidate)[0]
    
    @staticmethod
    def plot_candidate(candidate: HybridSimulationVariables,
                       color='k',
                       alpha=.5
                       ) -> None:
        plot_turbines(candidate.turb_pos_x, candidate.turb_pos_y,
                      color, alpha)
