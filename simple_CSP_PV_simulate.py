from pathlib import Path
import json
import pprint
import pandas as pd
import numpy as np
import humpday
import pyDOE2 as pyDOE

from hybrid.sites import SiteInfo
from hybrid.hybrid_simulation import HybridSimulation
from alt_dev.optimization_problem_alt import HybridSizingProblem
from alt_dev.optimization_driver_alt import OptimizationDriver

def init_hybrid_plant():
    """
    Initialize hybrid simulation object using specific project inputs
    :return: HybridSimulation as defined for this problem
    """

    site_data = {
        "lat": 32.69,
        "lon": 10.90,
        "elev": 115,
        "year": 2019,
        "tz": 0,
        "no_wind": True
        }

    solar_file = str(Path(__file__).parents[0]) + "/resource_files/solar/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv"
    prices_file = str(Path(__file__).parents[0]) + "/resource_files/grid/caiso_rice_ironmtn_2015.csv"

    # If normalized pricing is used, then PPA price must be adjusted after HybridSimulation is initialized
    site = SiteInfo(site_data, 
                    solar_resource_file=solar_file, 
                    grid_resource_file=prices_file)

    interconnection_size_kw = 60 * 1000                         # tower, PV and battery
    technologies = {
                    'tower': {
                        'cycle_capacity_kw': 50 * 1000,
                        'solar_multiple': 2.0,
                        'tes_hours': 12.0,
                        'optimize_field_before_sim': False
                        },
                    'pv': {
                        'system_capacity_kw': 70 * 1000
                        },
                    'battery': {
                        'system_capacity_kwh': 200 * 1000,
                        'system_capacity_kw': 50 * 1000
                        },
                    'grid': interconnection_size_kw
                   }


    # Create model
    hybrid_plant = HybridSimulation(technologies, 
                                    site,
                                    interconnect_kw=interconnection_size_kw,
                                    dispatch_options={
                                        'is_test_start_year': True,
                                        'is_test_end_year': True,
                                        'solver': 'cbc'
                                        }
                                    )

    # financial & depreciation parameters
    fin_params_file = 'financial_parameters.json'
    with open(fin_params_file) as f:
        fin_info = json.load(f)

    hybrid_plant.assign(fin_info["FinancialParameters"])
    hybrid_plant.assign(fin_info["TaxCreditIncentives"])
    hybrid_plant.assign(fin_info["Revenue"])
    hybrid_plant.assign(fin_info["Depreciation"])
    hybrid_plant.assign(fin_info["PaymentIncentives"])

    if (hybrid_plant.tower):
        hybrid_plant.tower.value('helio_width', 7.0)
        hybrid_plant.tower.value('helio_height', 7.0)

    if (hybrid_plant.pv):
        hybrid_plant.pv.dc_degradation = [0] * 25

    # hybrid_plant.ppa_price = (0.12,)  # $/kWh

    return hybrid_plant

def init_problem():
    """
    Initialize design problem and design variables
    :return: HybridSizingProblem
    """
    design_variables = dict(
        # tower =   {'cycle_capacity_kw':  {'bounds':(50*1e3, 125*1e3)},
        #            'solar_multiple':     {'bounds':(1.5,     3.5)},
        #            'tes_hours':          {'bounds':(6,       16)}
        #           },
        pv =      {'system_capacity_kw': {'bounds':(25*1e3,  200*1e3)},
                   'tilt':               {'bounds':(15,      60)}
                  },
    )

    # fixed_variables = {'tower': {'cycle_capacity_kw': 125*1e3}}

    # Problem definition
    problem = HybridSizingProblem(init_hybrid_plant, design_variables) #, fixed_variables)

    return problem

def max_hybrid_energy(result):
    return -result['annual_energies']['hybrid']

def min_pv_lcoe(result):
    return result['lcoe_real']['pv']

if __name__ == '__main__':

    # Driver config
    # cache_file = 'test_csp_pv.df.gz'
    # driver_config = dict(n_proc=4, eval_limit=100, cache_file=cache_file, cache_dir='test_lcoe')
    # driver = OptimizationDriver(init_problem, **driver_config)
    # n_dim = 2

    ### Sampling Example

    ## Parametric sweep
    # levels = np.array([3, 2])
    # design = pyDOE.fullfact(levels)
    # levels[levels == 1] = 2
    # ff_scaled = design / (levels - 1)

    ## Latin Hypercube
    # lhs_scaled = pyDOE.lhs(n_dim, criterion='center', samples=12)

    ## Execute Candidates
    # num_evals = driver.sample(ff_scaled, design_name='test_s', cache_file=cache_file)
    # num_evals = driver.parallel_sample(lhs_scaled, design_name='test_p', cache_file=cache_file)

    ### Optimization Example

    ## Show humpday optimizers
    # for i, f in enumerate(humpday.OPTIMIZERS):
    #     print(i, f.__name__)

    ## Select optimization algorithms, common configuration
    # optimizers = [humpday.OPTIMIZERS[0], humpday.OPTIMIZERS[1]]  # humpday.OPTIMIZERS[53]]
    # opt_config = dict(n_dim=n_dim, n_trials=100, with_count=True)

    ## Execute optimizer(s)
    # best_energy, best_energy_candidate = driver.optimize(optimizers[:1], opt_config, max_hybrid_energy, cache_file=cache_file)
    # best_lcoe, best_lcoe_candidate = driver.parallel_optimize(optimizers, opt_config, min_pv_lcoe, cache_file=cache_file)

    ## Print cache information
    # print(driver.cache_info)


    hybrid_plant = init_hybrid_plant()

    hybrid_plant.simulate()

    print("PPA price: {}".format(hybrid_plant.ppa_price[0]))

    if (hybrid_plant.tower):
        print("Tower CSP:")
        print("\tEnergy: {:.2f}".format(hybrid_plant.annual_energies.tower))
        print("\tCapacity Factor: {:.2f}".format(hybrid_plant.capacity_factors.tower))
        print("\tInstalled Cost: {:.2f}".format(hybrid_plant.tower.total_installed_cost))
        print("\tNPV: {:.2f}".format(hybrid_plant.net_present_values.tower))
        print("\tLCOE (nominal): {:.2f}".format(hybrid_plant.lcoe_nom.tower))
        print("\tLCOE (real): {:.2f}".format(hybrid_plant.lcoe_real.tower))
        print("\tBenefit Cost Ratio: {:.2f}".format(hybrid_plant.benefit_cost_ratios.tower))

    if (hybrid_plant.trough):
        print("Trough CSP:")
        print("\tEnergy: {:.2f}".format(hybrid_plant.annual_energies.trough))
        print("\tCapacity Factor: {:.2f}".format(hybrid_plant.capacity_factors.trough))
        print("\tInstalled Cost: {:.2f}".format(hybrid_plant.trough.total_installed_cost))
        print("\tNPV: {:.2f}".format(hybrid_plant.net_present_values.trough))
        print("\tLCOE (nominal): {:.2f}".format(hybrid_plant.lcoe_nom.trough))
        print("\tLCOE (real): {:.2f}".format(hybrid_plant.lcoe_real.trough))
        print("\tBenefit Cost Ratio: {:.2f}".format(hybrid_plant.benefit_cost_ratios.trough))

    if (hybrid_plant.pv):
        print("PV plant:")
        print("\tEnergy: {:.2f}".format(hybrid_plant.annual_energies.pv))
        print("\tCapacity Factor: {:.2f}".format(hybrid_plant.capacity_factors.pv))
        print("\tInstalled Cost: {:.2f}".format(hybrid_plant.pv.total_installed_cost))
        print("\tNPV: {:.2f}".format(hybrid_plant.net_present_values.pv))
        print("\tLCOE (nominal): {:.2f}".format(hybrid_plant.lcoe_nom.pv))
        print("\tLCOE (real): {:.2f}".format(hybrid_plant.lcoe_real.pv))
        print("\tBenefit Cost Ratio: {:.2f}".format(hybrid_plant.benefit_cost_ratios.pv))

    if (hybrid_plant.battery):
        print("Battery:")
        print("\tEnergy: {:.2f}".format(hybrid_plant.annual_energies.battery))
        print("\tInstalled Cost: {:.2f}".format(hybrid_plant.battery.total_installed_cost))
        print("\tNPV: {:.2f}".format(hybrid_plant.net_present_values.battery))
        print("\tLCOE (nominal): {:.2f}".format(hybrid_plant.lcoe_nom.battery))
        print("\tLCOE (real): {:.2f}".format(hybrid_plant.lcoe_real.battery))
        print("\tBenefit Cost Ratio: {:.2f}".format(hybrid_plant.benefit_cost_ratios.battery))

    print("Hybrid System:")
    print("\tEnergy: {:.2f}".format(hybrid_plant.annual_energies.hybrid))
    print("\tCapacity Factor: {:.2f}".format(hybrid_plant.capacity_factors.hybrid))
    print("\tInstalled Cost: {:.2f}".format(hybrid_plant.grid.total_installed_cost))
    print("\tNPV: {:.2f}".format(hybrid_plant.net_present_values.hybrid))
    print("\tLCOE (nominal): {:.2f}".format(hybrid_plant.lcoe_nom.hybrid))
    print("\tLCOE (real): {:.2f}".format(hybrid_plant.lcoe_real.hybrid))
    print("\tBenefit Cost Ratio: {:.2f}".format(hybrid_plant.benefit_cost_ratios.hybrid))

    tower_dict = hybrid_plant.tower.outputs.ssc_time_series
    tower_dict.update(hybrid_plant.tower.outputs.dispatch)

    # Print outputs to file
    # df = pd.DataFrame(tower_dict)
    # df.to_csv("tower_data_multipliers.csv")
    # outputs = hybrid_plant.hybrid_outputs(filename='check.csv')

    pass


# outputs = ("annual_energies", "capacity_factors", "lcoe_real", "lcoe_nom", "internal_rate_of_returns", "capacity_payments", "total_revenues", "net_present_values",
#                "benefit_cost_ratios", "energy_values", "energy_purchases_values", "energy_sales_values",
#                "federal_depreciation_totals", "federal_taxes", "om_expenses", "cost_installed", "insurance_expenses", "debt_payment", "")

# print("Outputs:")
# res = dict()
# for val in outputs:
#     try:
#         res[val] = str(getattr(hybrid_plant, val))
#     except:
#         pass

# pprint.pprint(res)
