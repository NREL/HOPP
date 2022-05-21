import pytest
from tools.analysis import CostCalculator, BOSLookup, create_cost_calculator


class TestCostCalculator:
    def test_calculate_installed_costs_per_mw(self):
        """
        Test if calculate_installed_costs returns the correct value for cost/mw model
        """
        cost_calculator = create_cost_calculator(100, 'CostPerMW', 'greenfield',
                                                 wind_installed_cost_mw=1454000, solar_installed_cost_mw=960000,
                                                 storage_installed_cost_mw=1455000, storage_installed_cost_mwh=0,
                                                 wind_bos_cost_mw=0, solar_bos_cost_mw=0, storage_bos_cost_mw=0,
                                                 storage_bos_cost_mwh=0, modify_costs=False)
        assert cost_calculator.calculate_installed_costs(1, 1, 1, 2) == (1454000, 960000, 1455000, 3869000)
        assert cost_calculator.model.calculate_bos_costs(wind_mw=1, solar_mw=1, storage_mw=1, storage_mwh=1,
                              wind_bos_cost_mw=550000, solar_bos_cost_mw=550000,
                              storage_bos_cost_mw=200000, storage_bos_cost_mwh=300000,
                              interconnection_mw=100, scenario_info='greenfield') == (550000, 550000, 500000, 1600000, 0)

    def test_calculate_installed_costs_per_mw_atb(self):
        """
        Test if calculate_installed_costs and calculate_bos_costs returns the correct value for cost/mw model with ATB
        """
        cost_calculator = create_cost_calculator(100, 'CostPerMW', 'greenfield', atb_costs=True, atb_year=2020,
                                                 atb_scenario='Moderate',
                                                 wind_installed_cost_mw=1454000, solar_installed_cost_mw=960000,
                                                 storage_installed_cost_mw=1455000, storage_installed_cost_mwh=0,
                                                 wind_bos_cost_mw=0, solar_bos_cost_mw=0, storage_bos_cost_mw=0,
                                                 storage_bos_cost_mwh=0, modify_costs=False)
        assert cost_calculator.calculate_installed_costs(1, 1, 1, 2) == (1678595.2959999999, 1353542.8569999998,
                                                                         857314.6919, 3889452.8449)
        assert cost_calculator.model.calculate_bos_costs(wind_mw=1, solar_mw=1, storage_mw=1, storage_mwh=1,
                              wind_bos_cost_mw=550000, solar_bos_cost_mw=550000,
                              storage_bos_cost_mw=200000, storage_bos_cost_mwh=300000,
                              interconnection_mw=100, scenario_info='greenfield') == (550000, 550000, 500000, 1600000, 0)

    def test_calculate_installed_costs(self):
        """
        Test if calculate_installed_costs runs
        """
        cost_calculator = create_cost_calculator(100, 'BOSLookup', 'greenfield', wind_installed_cost_mw=1454000,
                                                 solar_installed_cost_mw=960000, storage_installed_cost_mw=1455000)
        assert cost_calculator.calculate_installed_costs(1, 1, 1, 1) == (1454000, 960000, 1790000, 4204000)

    def test_bos_calculate_bos_costs_lookup(self):
        bos_calc = BOSLookup()

        interconnection_mw = 10
        wind_mw = 10
        solar_mw = 0
        wind_bos_cost, solar_bos_cost, total_bos_cost, min_distance = bos_calc.calculate_bos_costs(wind_mw,
                                                                                                   solar_mw,
                                                                                                   interconnection_mw)
        assert wind_bos_cost == pytest.approx(33937975)
        assert solar_bos_cost == 0
        assert total_bos_cost == pytest.approx(33937975)
        assert min_distance < 1e-7

    def test_bos_calculate_bos_costs_lookup_range(self):
        bos_calc = BOSLookup()

        found = []
        not_found = []

        interconnection_range = range(10, 510, 10)
        for interconnection_mw in interconnection_range:
            hybrid_capacities = [interconnection_mw * p / 100 for p in range(100, 210, 10)]
            for hybrid_cap in hybrid_capacities:
                solar_mws = [hybrid_cap * p / 100 for p in range(0, 110, 10)]
                wind_mws = [hybrid_cap - s for s in solar_mws]
                for solar_mw, wind_mw in zip(solar_mws, wind_mws):
                    wind_bos, solar_bos, total_bos, min_dist = bos_calc.calculate_bos_costs(wind_mw,
                                                                                            solar_mw,
                                                                                            interconnection_mw)
                    if min_dist > 1e-7:
                        not_found.append([wind_mw, solar_mw, interconnection_mw])
                    else:
                        found.append([wind_mw, solar_mw, interconnection_mw])

        assert(len(not_found) == 0)

    def test_bos_calculate_bos_costs_interpolate(self):
        bos_calc = BOSLookup()

        interconnection_mw = 100
        wind_mw = 95
        solar_mw = 15
        wind_bos_cost, solar_bos_cost, total_bos_cost, min_distance = bos_calc.calculate_bos_costs(wind_mw,
                                                                                                   solar_mw,
                                                                                                   interconnection_mw)

        low_wind_bos, low_solar_bos_cost, _, low_distance = bos_calc.calculate_bos_costs(wind_mw - 5,
                                                                                         solar_mw,
                                                                                         interconnection_mw)

        high_wind_bos, high_solar_bos_cost, _, high_distance = bos_calc.calculate_bos_costs(wind_mw + 5,
                                                                                            solar_mw,
                                                                                            interconnection_mw)

        assert wind_bos_cost > low_wind_bos
        assert wind_bos_cost < high_wind_bos
        assert solar_bos_cost == low_solar_bos_cost == high_solar_bos_cost
        assert total_bos_cost == pytest.approx(75356239)
        assert min_distance != 0

    def test_bos_calculate_bos_costs_extrapolate_error(self):
        # lookup cannot be extrapolated
        bos_calc = BOSLookup()

        interconnection_mw = 550
        wind_mw = 295
        solar_mw = 295
        try:
            bos_calc.calculate_bos_costs(wind_mw, solar_mw, interconnection_mw)
            assert False
        except ValueError:
            assert True


