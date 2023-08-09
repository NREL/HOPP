from .bos_model import BOSCostPerMW, BOSCalculator
from .bos_lookup import BOSLookup
# from .hybrid_bosse import HybridBOSSE
from hopp.log import bos_logger as logger
import numpy as np

class CostCalculator():
    """
    CostCalculator class contains tools to determine BOS component costs and Installed costs for a single technology
    or hybrid plant
    """
    def __init__(self,
                 bos_cost_source,
                 scenario,
                 interconnection_size,
                 wind_installed_cost_mw,
                 pv_installed_cost_mw,
                 storage_installed_cost_mw,
                 storage_installed_cost_mwh,
                 wind_bos_cost_mw=0,
                 pv_bos_cost_mw=0,
                 storage_bos_cost_mw=0,
                 storage_bos_cost_mwh=0,
                 modify_costs=False,
                 cost_reductions=[]):

        """
        :param bos_cost_source: Defines the type of bos analysis used. Options are 'JSONLookup', 'Cost/MW',
                                    'HybridBOSSE', 'HybridBOSSE manual'
        :param scenario: 'greenfield' or 'solar addition'
        :param interconnection_size: Size (MW) of interconnection
        :param wind_installed_cost_mw: $USD cost/mw for installed wind
        :param pv_installed_cost_mw: $USD cost/mw for installed solar
        :param storage_installed_cost_mw: $USD cost/mw for installed storage
        :param storage_installed_cost_mwh: $USD cost/mwh for installed storage
        :param wind_bos_cost_mw: $USD cost/mw for for wind BOS
        :param pv_bos_cost_mw: $USD cost/mw for for solar BOS
        :param storage_bos_cost_mw: $USD cost/mw for for storage BOS
        :param storage_bos_cost_mwh: $USD cost/mw for for storage BOS
        :param modify_costs: (boolean) Flag to determine whether returned costs will be modified using supplied
                                modifiers
        :param cost_reductions: Dictionary specifying CAPEX reduction fraction
        """
        self.descriptor = 'BOS function'

        if scenario == 'greenfield':
            self.scenario = scenario
        elif scenario == 'solar addition':
            raise NotImplementedError
        else:
            raise ValueError("CostCalculator scenario must be 'greenfield' or 'solar addition'")

        self.interconnection_size = interconnection_size
        self.model = BOSCalculator()

        if bos_cost_source.lower() == "boslookup":
            self.model = BOSLookup()
        elif bos_cost_source.lower() == "costpermw":
            self.model = BOSCostPerMW()
        elif bos_cost_source.lower() == "hybridbosse":
            raise NotImplementedError

        self.bos_cost_source = bos_cost_source
        self.wind_installed_cost_mw = wind_installed_cost_mw
        self.pv_installed_cost_mw = pv_installed_cost_mw
        self.storage_installed_cost_mw = storage_installed_cost_mw
        self.storage_installed_cost_mwh = storage_installed_cost_mwh
        self.wind_bos_cost_mw = wind_bos_cost_mw
        self.pv_bos_cost_mw = pv_bos_cost_mw
        self.storage_bos_cost_mw = storage_bos_cost_mw
        self.storage_bos_cost_mwh = storage_bos_cost_mwh
        self.modify_costs = modify_costs
        self.cost_reductions = cost_reductions

    def calculate_installed_costs(self, wind_size, pv_size, storage_size_mw=0., storage_size_mwh=0.):
        """
        Calculates installed costs for wind, solar, and hybrid based on installed cost/mw and size of plant
        :return: installed cost of wind, solar and hybrid components of plant

        """
        total_installed_cost = 0.
        wind_installed_cost = self.wind_installed_cost_mw * wind_size
        solar_installed_cost = self.pv_installed_cost_mw * pv_size
        if storage_size_mwh > 0 and storage_size_mw > 0:
            storage_hours = storage_size_mwh / storage_size_mw
            storage_installed_cost = storage_size_mw * (self.storage_installed_cost_mw +
                                            (self.storage_installed_cost_mwh * storage_hours))
        else:
            storage_installed_cost = 0
        total_installed_cost += wind_installed_cost
        total_installed_cost += solar_installed_cost
        total_installed_cost += storage_installed_cost
        return wind_installed_cost, solar_installed_cost, storage_installed_cost, total_installed_cost

    def calculate_total_costs(self, wind_mw, pv_mw, storage_mw=0., storage_mwh=0.):
        """
        Calculates total installed cost of plant (BOS Cost + Installed Cost).
        Modifies the capex or opex costs as specified in cost_reductions if modify_costs is True
        :return: Total installed cost of plant (BOS Cost + Installed Cost)
        """

        wind_installed_cost, solar_installed_cost, storage_installed_cost, total_installed_cost = \
            self.calculate_installed_costs(wind_mw, pv_mw, storage_mw, storage_mwh)

        if self.bos_cost_source.lower() == 'costpermw':
            wind_bos_cost, solar_bos_cost, storage_bos_cost, total_bos_cost, _ = \
                self.model.calculate_bos_costs(wind_mw, pv_mw, storage_mw, storage_mwh, self.wind_bos_cost_mw,
                                               self.pv_bos_cost_mw, self.storage_bos_cost_mw, self.storage_bos_cost_mwh,
                                               self.interconnection_size, self.scenario)
        else:
            wind_bos_cost, solar_bos_cost, total_bos_cost, min_distance = \
                self.model.calculate_bos_costs(wind_mw, pv_mw, self.interconnection_size)
            storage_bos_cost = 0.

        total_wind_cost = wind_installed_cost + wind_bos_cost
        total_solar_cost = solar_installed_cost + solar_bos_cost
        total_storage_cost = storage_installed_cost + storage_bos_cost
        total_project_cost = total_installed_cost + total_bos_cost

        if self.modify_costs:
            logger.info('Modifying costs using selected multipliers')
            logger.info("Total Project Cost Before Modifiers: {}".format(total_project_cost))
            if wind_mw > 0 and pv_mw > 0:
                total_project_cost = ((1 - self.cost_reductions['solar_capex_reduction_hybrid']) *
                                      solar_installed_cost) + \
                                 ((1 - self.cost_reductions[
                                     'solar_bos_reduction_hybrid']) * solar_bos_cost) + \
                                 ((1 - self.cost_reductions['wind_capex_reduction_hybrid']) *
                                  wind_installed_cost) + \
                                 ((1 - self.cost_reductions[
                                     'wind_bos_reduction_hybrid']) * wind_bos_cost)
            elif pv_mw > 0:
                total_project_cost = ((1 - self.cost_reductions['solar_capex_reduction']) *
                                      solar_installed_cost) + \
                                 ((1 - self.cost_reductions['solar_bos_reduction']) * solar_bos_cost)
            elif wind_mw > 0:
                total_project_cost = ((1 - self.cost_reductions['wind_capex_reduction']) *
                                      wind_installed_cost) + \
                                 ((1 - self.cost_reductions['wind_bos_reduction']) * wind_bos_cost)

            logger.info("Total Project Cost After Modifiers: {}".format(total_project_cost))
        # else:
            # logger.info('Not modifying costs')
            # Not modifying wind or solar costs

        logger.info("Total Project Cost (Installed Cost + BOS Cost): {}".format(total_project_cost))
        return total_solar_cost, total_wind_cost, total_storage_cost, total_project_cost


def create_cost_calculator(interconnection_mw: float,
                           bos_cost_source: str = "CostPerMW",
                           scenario: str = "greenfield",
                           atb_costs: bool = False,
                           atb_year: float = 2020,
                           atb_scenario: str = "Moderate",
                           wind_installed_cost_mw: float = 1454000,
                           solar_installed_cost_mw: float = 960000,
                           storage_installed_cost_mw: float = 291000,
                           storage_installed_cost_mwh: float = 335000,
                           wind_bos_cost_mw: float = 0,
                           solar_bos_cost_mw: float = 0,
                           storage_bos_cost_mw: float = 0,
                           storage_bos_cost_mwh: float = 0,
                           modify_costs: bool = False,
                           cost_reductions=dict()) -> CostCalculator:

    if modify_costs:
        cost_reductions['solar_capex_reduction'] = 0
        cost_reductions['wind_capex_reduction'] = 0
        cost_reductions['wind_bos_reduction'] = 0
        cost_reductions['solar_bos_reduction'] = 0
        cost_reductions['wind_capex_reduction_hybrid'] = 0.1
        cost_reductions['solar_capex_reduction_hybrid'] = 0.1
        cost_reductions['wind_bos_reduction_hybrid'] = 0.1
        cost_reductions['solar_bos_reduction_hybrid'] = 0.1

    if atb_costs:
        from .atb_lookup import ATBLookup
        atblookup = ATBLookup()
        wind_installed_cost_mw, solar_installed_cost_mw, storage_installed_cost_mw, storage_installed_cost_mwh = \
            atblookup.calculate_atb_costs(atb_year, atb_scenario)

    return CostCalculator(bos_cost_source, scenario, interconnection_mw, wind_installed_cost_mw,
                          solar_installed_cost_mw, storage_installed_cost_mw, storage_installed_cost_mwh,
                          wind_bos_cost_mw, solar_bos_cost_mw, storage_bos_cost_mw, storage_bos_cost_mwh,
                          modify_costs, cost_reductions)
