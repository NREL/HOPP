import pytest
from tools.analysis import CostCalculator, BOSLookup, create_cost_calculator


class TestCostCalculator:
    def test_calculate_installed_costs(self):
        """
        Test if calculate_installed_costs runs
        """
        cost_calculator = create_cost_calculator(100, 'BOSLookup', 'greenfield', 1454000, 960000, False)
        assert cost_calculator.calculate_installed_costs(1, 1) == (1454000, 960000, 2414000)

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
        assert total_bos_cost == pytest.approx(79600641)
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

    def test_run_hybridbosse(self):
        try:
            cost_calculator = create_cost_calculator(100, 'hybridbosse', 'hybrid', 1454000, 960000, False)
            answer = cost_calculator.calculate_bos_costs(10, 10)
        except ValueError:
            assert True

