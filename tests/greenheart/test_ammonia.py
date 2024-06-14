from pytest import approx

from greenheart.simulation.technologies.ammonia import ammonia

grid_prices_dict = {
    "2035": 89.42320514456621,
    "2036": 89.97947569251141,
    "2037": 90.53574624045662,
    "2038": 91.09201678840184,
    "2039": 91.64828733634704,
    "2040": 92.20455788429224,
    "2041": 89.87291235917809,
    "2042": 87.54126683406393,
    "2043": 85.20962130894978,
    "2044": 82.87797578383562,
    "2045": 80.54633025872147,
    "2046": 81.38632144593608,
    "2047": 82.22631263315068,
    "2048": 83.0663038203653,
    "2049": 83.90629500757991,
    "2050": 84.74628619479452,
    "2051": 84.74628619479452,
    "2052": 84.74628619479452,
    "2053": 84.74628619479452,
    "2054": 84.74628619479452,
    "2055": 84.74628619479452,
    "2056": 84.74628619479452,
    "2057": 84.74628619479452,
    "2058": 84.74628619479452,
    "2059": 84.74628619479452,
    "2060": 84.74628619479452,
    "2061": 84.74628619479452,
    "2062": 84.74628619479452,
    "2063": 84.74628619479452,
    "2064": 84.74628619479452,
}

financial_assumptions = {
    "total income tax rate": 0.2574,
    "capital gains tax rate": 0.15,
    "leverage after tax nominal discount rate": 0.10893,
    "debt equity ratio of initial financing": 0.624788,
    "debt interest rate": 0.050049,
}


def test_run_ammonia_model():
    capacity = 100.0
    capacity_factor = 0.9

    ammonia_production_kgpy = ammonia.run_ammonia_model(capacity, capacity_factor)

    assert ammonia_production_kgpy == 90


def test_ammonia_cost_model(subtests):
    config = ammonia.AmmoniaCostModelConfig(
        plant_capacity_kgpy=362560672.27155423,
        plant_capacity_factor=0.9,
        feedstocks=ammonia.Feedstocks(
            electricity_cost=89.42320514456621,
            hydrogen_cost=4.2986685034417045,
            cooling_water_cost=0.00291,
            iron_based_catalyst_cost=23.19977341,
            oxygen_cost=0,
        ),
    )

    res: ammonia.AmmoniaCostModelOutputs = ammonia.run_ammonia_cost_model(config)

    with subtests.test("Total CapEx"):
        assert res.capex_total == approx(74839480.74961768)
    with subtests.test("Fixed OpEx"):
        assert res.total_fixed_operating_cost == approx(10569658.376900101)
    with subtests.test("Variable costs"):
        assert res.variable_cost_in_startup_year == approx(4259805.969069265)
    with subtests.test("Land costs"):
        assert res.land_cost == approx(2160733.0556864925)


def test_ammonia_finance_model():
    cost_config = ammonia.AmmoniaCostModelConfig(
        plant_capacity_kgpy=362560672.27155423,
        plant_capacity_factor=0.9,
        feedstocks=ammonia.Feedstocks(
            electricity_cost=89.42320514456621,
            hydrogen_cost=4.2986685034417045,
            cooling_water_cost=0.00291,
            iron_based_catalyst_cost=23.19977341,
            oxygen_cost=0,
        ),
    )

    costs: ammonia.AmmoniaCostModelOutputs = ammonia.run_ammonia_cost_model(cost_config)

    plant_capacity_kgpy = 362560672.27155423
    plant_capacity_factor = 0.9

    config = ammonia.AmmoniaFinanceModelConfig(
        plant_life=30,
        plant_capacity_kgpy=plant_capacity_kgpy,
        plant_capacity_factor=plant_capacity_factor,
        feedstocks=cost_config.feedstocks,
        grid_prices=grid_prices_dict,
        financial_assumptions=financial_assumptions,
        costs=costs,
    )

    lcoa_expected = 0.9322866176899477

    res: ammonia.AmmoniaFinanceModelOutputs = ammonia.run_ammonia_finance_model(config)

    assert res.sol.get("price") == lcoa_expected


def test_ammonia_size_h2_input(subtests):
    config = ammonia.AmmoniaCapacityModelConfig(
        hydrogen_amount_kgpy=73288888.8888889,
        input_capacity_factor_estimate=0.9,
        feedstocks=ammonia.Feedstocks(
            electricity_cost=89.42320514456621,
            hydrogen_cost=4.2986685034417045,
            cooling_water_cost=0.00291,
            iron_based_catalyst_cost=23.19977341,
            oxygen_cost=0,
        ),
    )

    res: ammonia.AmmoniaCapacityModelOutputs = ammonia.run_size_ammonia_plant_capacity(
        config
    )

    with subtests.test("Ammonia plant size"):
        assert res.ammonia_plant_capacity_kgpy == approx(334339658.8730839)
    with subtests.test("hydrogen input"):
        assert res.hydrogen_amount_kgpy == approx(73288888.8888889)


def test_ammonia_size_NH3_input(subtests):
    config = ammonia.AmmoniaCapacityModelConfig(
        desired_ammonia_kgpy=334339658.8730839,
        input_capacity_factor_estimate=0.9,
        feedstocks=ammonia.Feedstocks(
            electricity_cost=89.42320514456621,
            hydrogen_cost=4.2986685034417045,
            cooling_water_cost=0.00291,
            iron_based_catalyst_cost=23.19977341,
            oxygen_cost=0,
        ),
    )

    res: ammonia.AmmoniaCapacityModelOutputs = ammonia.run_size_ammonia_plant_capacity(
        config
    )

    with subtests.test("Ammonia plant size"):
        assert res.ammonia_plant_capacity_kgpy == approx(371488509.8589821)
    with subtests.test("hydrogen input"):
        assert res.hydrogen_amount_kgpy == approx(73288888.8888889)


def test_ammonia_full_model(subtests):
    config = {
        "ammonia": {
            "capacity": {
                "hydrogen_amount_kgpy": 73288888.8888889,
                "input_capacity_factor_estimate": 0.9,
            },
            "costs": {
                "feedstocks": {
                    "electricity_cost": 89.42320514456621,
                    "hydrogen_cost": 4.2986685034417045,
                    "cooling_water_cost": 0.00291,
                    "iron_based_catalyst_cost": 23.19977341,
                    "oxygen_cost": 0,
                },
            },
            "finances": {
                "plant_life": 30,
                "grid_prices": grid_prices_dict,
                "financial_assumptions": financial_assumptions,
            },
        }
    }

    res = ammonia.run_ammonia_full_model(config)

    assert len(res) == 3

    with subtests.test("Ammonia plant size"):
        assert res[0].ammonia_plant_capacity_kgpy == approx(334339658.8730839)

    with subtests.test("capex"):
        assert res[1].capex_total == approx(71189795.03176591)

    with subtests.test("LCOA"):
        assert res[2].sol.get("price") == approx(0.934951845207612)
