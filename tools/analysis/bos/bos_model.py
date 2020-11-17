class BOSCalculator:
    def __init__(self):
        self.name = "None"

    def _calculate_greenfield(self, wind_mw, solar_mw, interconnection_mw: float = 0):
        """

        :param interconnection_mw:
        :param wind_mw:
        :param solar_mw:
        :return: wind_bos_cost, solar_bos_cost, total_bos_cost
        """
        raise NotImplementedError

    def _calculate_solar_addition(self, wind_mw, solar_mw, interconnection_mw: float = 0):
        """

        :param interconnection_mw:
        :param wind_mw:
        :param solar_mw:
        :return: wind_bos_cost, solar_bos_cost, total_bos_cost
        """
        raise NotImplementedError

    def calculate_bos_costs(self, wind_mw, solar_mw, interconnection_mw, scenario_info):
        raise NotImplementedError


class BOSCostPerMW(BOSCalculator):
    def __init__(self):
        super().__init__()
        self.name = "CostPerMW"

    @staticmethod
    def _calculate(wind_mw, solar_mw, fixed_wind, fixed_solar, fixed_hybrid, solar_per_mw, wind_per_mw):
        total_bos_cost = 0
        solar_bos_cost = (solar_per_mw * solar_mw) + fixed_solar
        total_bos_cost += solar_bos_cost
        wind_bos_cost = (wind_per_mw * wind_mw) + fixed_wind
        total_bos_cost += wind_bos_cost
        if wind_mw and solar_mw:
            total_bos_cost += fixed_hybrid

        return wind_bos_cost, solar_bos_cost, total_bos_cost

    def _calculate_greenfield(self, wind_mw: float, solar_mw: float, interconnection_mw: float = 0):
        fixed_bos_cost_wind = 15000000
        fixed_bos_cost_solar = 5000000
        fixed_bos_cost_hybrid = 1000000
        solar_bos_cost_per_mw = 99
        wind_bos_cost_per_mw = 88
        return BOSCostPerMW._calculate(wind_mw, solar_mw, fixed_bos_cost_wind, fixed_bos_cost_solar,
                                       fixed_bos_cost_hybrid, solar_bos_cost_per_mw, wind_bos_cost_per_mw)

    def _calculate_solar_addition(self, wind_mw: float, solar_mw: float, interconnection_mw: float = 0):
        fixed_bos_cost_wind = 15000000
        fixed_bos_cost_solar = 0
        fixed_bos_cost_hybrid = 0
        solar_bos_cost_per_mw = 99
        wind_bos_cost_per_mw = 88
        return BOSCostPerMW._calculate(wind_mw, solar_mw, fixed_bos_cost_wind, fixed_bos_cost_solar,
                                       fixed_bos_cost_hybrid, solar_bos_cost_per_mw, wind_bos_cost_per_mw)
