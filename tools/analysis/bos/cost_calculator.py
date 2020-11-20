from .bos_model import BOSCostPerMW, BOSCalculator
from .bos_lookup import BOSLookup
# from .hybrid_bosse import HybridBOSSE
from hybrid.log import bos_logger as logger


class CostCalculator():
    """
    CostCalculator class contains tools to determine BOS component costs and Installed costs for a single technology or hybrid plant
    """
    def __init__(self,
                 bos_cost_source,
                 scenario,
                 interconnection_size,
                 wind_installed_cost_mw,
                 solar_installed_cost_mw,
                 modify_costs,
                 cost_reductions):

        """
        :param bos_cost_source: Defines the type of bos analysis used. Options are 'JSONLookup', 'Cost/MW',
         'HybridBOSSE', 'HybridBOSSE manual'
        :param scenario: 'greenfield' or 'solar addition'
        :param interconnection_size: Size (MW) of interconnection
        :param wind_installed_cost_mw: $USD cost/mw for installed wind
        :param solar_installed_cost_mw: $USD cost/mw for installed solar
        :param modify_costs: (boolean) Flag to determine whether returned costs will be modified using supplied
        modifiers
        :param cost_reductions: Dictionary specifiying CAPEX reduction fraction
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

        self.wind_installed_cost_mw = wind_installed_cost_mw
        self.solar_installed_cost_mw = solar_installed_cost_mw
        self.modify_costs = modify_costs
        self.cost_reductions = cost_reductions

    def calculate_installed_costs(self, wind_size, solar_size):
        """
        Calculates installed costs for wind, solar, and hybrid based on installed cost/mw and size of plant
        :return: installed cost of wind, solar and hybrid components of plant

        """
        total_installed_cost = 0
        wind_installed_cost = self.wind_installed_cost_mw * wind_size
        solar_installed_cost = self.solar_installed_cost_mw * solar_size
        total_installed_cost += wind_installed_cost
        total_installed_cost += solar_installed_cost
        return wind_installed_cost, solar_installed_cost, total_installed_cost

    def calculate_total_costs(self, wind_mw, solar_mw):
        """
        Calculates total installed cost of plant (BOS Cost + Installed Cost).
        Modifies the capex or opex costs as specified in cost_reductions if modify_costs is True
        :return: Total installed cost of plant (BOS Cost + Installed Cost)
        """
        logger.info("Determining total costs for Wind size: {}MW and Solar size: {}MW and Interconnection size: {}MW"
                    .format(wind_mw, solar_mw, self.interconnection_size))

        logger.info("Using {}$/MW for installed Wind cost and {}$/MW for installed Solar cost"
                    .format(self.wind_installed_cost_mw, self.solar_installed_cost_mw))

        logger.info("Using '{}' to determine BOS costs".format(self.model.name))

        wind_installed_cost, solar_installed_cost, total_installed_cost = self.calculate_installed_costs(wind_mw,
                                                                                                         solar_mw)
        wind_bos_cost, solar_bos_cost, total_bos_cost, _ = self.model.calculate_bos_costs(wind_mw,
                                                                                          solar_mw,
                                                                                          self.interconnection_size,
                                                                                          self.scenario)
        total_wind_cost = wind_installed_cost + wind_bos_cost
        total_solar_cost = solar_installed_cost + solar_bos_cost
        total_project_cost = total_installed_cost + total_bos_cost

        if self.modify_costs:
            logger.info('Modifying costs using selected multipliers')
            logger.info("Total Project Cost Before Modifiers: {}".format(total_project_cost))
            if wind_mw > 0 and solar_mw > 0:
                total_project_cost = ((1 - self.cost_reductions['solar_capex_reduction_hybrid']) *
                                      solar_installed_cost) + \
                                 ((1 - self.cost_reductions[
                                     'solar_bos_reduction_hybrid']) * solar_bos_cost) + \
                                 ((1 - self.cost_reductions['wind_capex_reduction_hybrid']) *
                                  wind_installed_cost) + \
                                 ((1 - self.cost_reductions[
                                     'wind_bos_reduction_hybrid']) * wind_bos_cost)
            elif solar_mw > 0:
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
        return total_solar_cost, total_wind_cost, total_project_cost


def create_cost_calculator(interconnection_mw: float,
                           bos_cost_source: str = "BOSLookup",
                           scenario: str = "greenfield",
                           wind_installed_cost_mw: float = 1454000,
                           solar_installed_cost_mw: float = 960000,
                           modify_costs: bool = False
                           ):
    cost_reductions = dict()
    cost_reductions['solar_capex_reduction'] = 0
    cost_reductions['wind_capex_reduction'] = 0
    cost_reductions['wind_bos_reduction'] = 0
    cost_reductions['solar_bos_reduction'] = 0
    cost_reductions['wind_capex_reduction_hybrid'] = 0.1
    cost_reductions['solar_capex_reduction_hybrid'] = 0.1
    cost_reductions['wind_bos_reduction_hybrid'] = 0.1
    cost_reductions['solar_bos_reduction_hybrid'] = 0.1

    return CostCalculator(bos_cost_source, scenario, interconnection_mw, wind_installed_cost_mw,
                          solar_installed_cost_mw, modify_costs, cost_reductions)
