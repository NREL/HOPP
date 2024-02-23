import os
from typing import Dict

from pytest import approx, fixture
import pandas as pd

from greenheart.simulation.technologies import steel

from tests import TEST_ROOT_DIR

def get_ng_prices_dict(year: int, site: str) -> Dict[int, float]:
    ng_prices = pd.read_csv(
        TEST_ROOT_DIR / "greenheart" / "inputs" / "ngprices_base.csv", 
        index_col = 0,
        header = 0, 
    )
    ng_prices = ng_prices * 1000 # convert to GJ
    ng_prices_dict = {}

    price = ng_prices.loc[year, site]
    ng_prices_dict[year] = price

    return ng_prices_dict


@fixture
def ng_prices_dict():
    return get_ng_prices_dict(2035, "MN")


def test_run_steel_model():
    capacity = 100.
    capacity_factor = .9

    steel_production_mtpy = steel.run_steel_model(capacity, capacity_factor)

    assert steel_production_mtpy == 90.


def test_steel_cost_model(subtests, ng_prices_dict):
    config = steel.SteelCostModelConfig(
        operational_year=2035,
        plant_capacity_mtpy=1084408.2137715619,
        lcoh=4.186771317772853,
        feedstocks=steel.Feedstocks(
            natural_gas_prices=ng_prices_dict,
            oxygen_market_price=0
        ),
        o2_heat_integration=False
    )

    res: steel.SteelCostModelOutputs = steel.run_steel_cost_model(config)

    with subtests.test("CapEx"):
        assert res.total_plant_cost == approx(451513562.41513157)
    with subtests.test("Fixed OpEx"):
        assert res.total_fixed_operating_cost == approx(99119892.8431614)
    with subtests.test("Installation"):
        assert res.installation_cost == approx(179525207.856775)


def test_steel_finance_model(ng_prices_dict):
    # Parameter -> Hydrogen/Steel/Ammonia
    financial_assumptions = {
        "total income tax rate": 0.2574,
        "capital gains tax rate": 0.15,
        "leverage after tax nominal discount rate": 0.10893,
        "debt equity ratio of initial financing": 0.624788,
        "debt interest rate": 0.050049,
    }

    cost_config = steel.SteelCostModelConfig(
        operational_year=2035,
        plant_capacity_mtpy=1084408.2137715619,
        lcoh=4.186771317772853,
        feedstocks=steel.Feedstocks(
            natural_gas_prices=ng_prices_dict,
            oxygen_market_price=0
        ),
        o2_heat_integration=False
    )

    costs: steel.SteelCostModelOutputs = steel.run_steel_cost_model(cost_config)

    plant_capacity_factor=.9
    steel_production_mtpy = steel.run_steel_model(
        cost_config.plant_capacity_mtpy, plant_capacity_factor
    )

    config = steel.SteelFinanceModelConfig(
        plant_life=30,
        plant_capacity_mtpy=cost_config.plant_capacity_mtpy,
        plant_capacity_factor=plant_capacity_factor,
        steel_production_mtpy=steel_production_mtpy,
        lcoh=cost_config.lcoh,
        feedstocks=cost_config.feedstocks,
        grid_prices={
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
            "2064": 84.74628619479452
        },
        financial_assumptions=financial_assumptions,
        costs=costs
    )

    lcos_expected = 961.2866791076059
    
    res: steel.SteelFinanceModelOutputs = steel.run_steel_finance_model(config)

    assert res.sol.get('price') == lcos_expected