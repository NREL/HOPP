from pathlib import Path
import csv
import json
import pprint
import pandas as pd
import numpy as np
import humpday
import pyDOE2 as pyDOE
import time

from hybrid.sites import SiteInfo
from hybrid.hybrid_simulation import HybridSimulation
from alt_dev.optimization_problem_alt import HybridSizingProblem
from alt_dev.optimization_driver_alt import OptimizationDriver

def print_table_metric(hybrid: HybridSimulation, metric: str, display_name: str=None):
    sep = " \t| "
    
    def sept(value):
        if value == 0:
            return " \t\t| "
        else:
            return " \t| "
    
    def value_line(value):
        line = "{:.2f}".format(value)
        sep = sept(value)
        return line + sep
    
    if display_name is None:
        line = metric + sep
    else:
        line = display_name + sep
        
    line += value_line(hybrid.grid.value(metric))
        
    if (hybrid_plant.tower):
        line += value_line(hybrid.tower.value(metric))
    if (hybrid_plant.trough):
        line += value_line(hybrid.trough.value(metric))
    if (hybrid_plant.pv):
        line += value_line(hybrid.pv.value(metric))
    if (hybrid_plant.battery):
        line += value_line(hybrid.battery.value(metric))
    print(line)


def init_hybrid_plant():
    """
    Initialize hybrid simulation object using specific project inputs
    :return: HybridSimulation as defined for this problem
    """
    is_test = True  # Turns off full year dispatch and optimize tower and receiver
    
    techs_in_sim = ['tower',
                    'pv',
                    'battery',
                    'grid']

    site_data = {
        "lat": 34.85,
        "lon": -116.9,
        "elev": 641,
        "year": 2012,
        "tz": -8,
        "no_wind": True
        }

    solar_file = "../HOPP analysis/weather/daggett_CA/34.85_-116.90_psmv3_2012_with_leap_day_relabeled.csv"
    price_year = 2030
    # NOTE: prices with carbon cost
    prices_file = "../HOPP analysis/Cambium data/MidCase BA10 (southern CA)/cambium_midcase_BA10_{year}_price.csv".format(year=price_year)
    cap_hrs_file = "../HOPP analysis/Capacity_payments/100_high_net_load/cambium_midcase_BA10_{year}_cap_hours.csv".format(year=price_year)

    with open(cap_hrs_file) as f:
        csvreader = csv.reader(f)
        cap_hrs = []
        for row in csvreader:
            cap_hrs.append(row[0] == 'True')

    # If normalized pricing is used, then PPA price must be adjusted after HybridSimulation is initialized
    site = SiteInfo(site_data, 
                    solar_resource_file=solar_file, 
                    grid_resource_file=prices_file,
                    capacity_hours=cap_hrs)

    # 2030 SETO cost targets
    with open("2030_SETO_targets.json") as f:
        cost_info = json.load(f)

    tower_rec_cost = cost_info['csp_costs']['tower_rec_cost_per_kwt']
    cost_info['csp_costs'].pop('tower_rec_cost_per_kwt')

    technologies = {'tower': {
                        'cycle_capacity_kw': 100 * 1000,
                        'solar_multiple': 2.0,
                        'tes_hours': 14.0,
                        'optimize_field_before_sim': not is_test,
                        'scale_input_params': True,
                        },
                    'trough': {
                        'cycle_capacity_kw': 100 * 1000,
                        'solar_multiple': 2.0,
                        'tes_hours': 14.0
                    },
                    'pv': {
                        'system_capacity_kw': 120 * 1000
                        },
                    'battery': {
                        'system_capacity_kwh': 200 * 1000,
                        'system_capacity_kw': 100 * 1000
                        },
                    'grid': 100 * 1000}

    # Create model
    hybrid_plant = HybridSimulation({key: technologies[key] for key in techs_in_sim}, 
                                    site,
                                    interconnect_kw=technologies['grid'],
                                    dispatch_options={
                                        'is_test_start_year': is_test,
                                        'is_test_end_year': is_test,
                                        'solver': 'xpress_persistent',
                                        'grid_charging': False,
                                        'pv_charging_only': True
                                        },
                                    simulation_options={
                                        'storage_capacity_credit': False,
                                    },
                                    cost_info=cost_info['cost_info']
                                    )

    # csp costs
    if hybrid_plant.tower:
        hybrid_plant.tower.ssc.set(cost_info['csp_costs'])
    hybrid_plant.assign(cost_info["SystemCosts"])

    # financial & depreciation parameters
    fin_params_file = 'financial_parameters_SAM.json'   # Capacity payment amount is set here
    with open(fin_params_file) as f:
        fin_info = json.load(f)

    hybrid_plant.assign(fin_info["FinancialParameters"])
    hybrid_plant.assign(fin_info["TaxCreditIncentives"])
    hybrid_plant.assign(fin_info["Revenue"])
    hybrid_plant.assign(fin_info["Depreciation"])
    hybrid_plant.assign(fin_info["PaymentIncentives"])

    if hybrid_plant.pv:
        hybrid_plant.pv.dc_degradation = [0.5] * 25
        hybrid_plant.pv.value('array_type', 2)  # 1-axis tracking
        hybrid_plant.pv.value('tilt', 0)        # Tilt for 1-axis

    # This is required if normalized prices are provided
    # hybrid_plant.ppa_price = (0.12,)  # $/kWh

    return hybrid_plant

def init_problem():
    """
    Initialize design problem and design variables
    :return: HybridSizingProblem
    """
    design_variables = dict(
        tower =   {'cycle_capacity_kw':  {'bounds': (50*1e3,  99*1e3)},
                   'solar_multiple':     {'bounds': (1.0,     3.5)},
                   'tes_hours':          {'bounds': (5,       16)},
                   #'dni_des':            {'bounds': (750,     1000)}
                   },
        pv =      {'system_capacity_kw':  {'bounds': (25*1e3,  400*1e3)},
                   'dc_ac_ratio':         {'bounds': (1.0,     1.6)},
                   #'tilt':                {'bounds': (15,      60)}
                   },
        battery =  {'system_capacity_kwh': {'bounds': (50.0*1e3, 5*50.0*1e3)},
                    'system_capacity_kw':  {'bounds': (1*1e3,  50.0*1e3)},
                   }
    )

    #fixed_variables = {'tower': {'cycle_capacity_kw': 110*1e3,
    #                             'dni_des': 950}}

    out_options = dict(dispatch_factors=True,  # add dispatch factors to objective output
                       generation_profile=True,  # add technology generation profile to output
                       financial_model=False,  # add financial model dictionary to output
                       shrink_output=False,  # keep only the first year of output
                       )

    # Problem definition
    problem = HybridSizingProblem(init_hybrid_plant, 
                                  design_variables, 
                                  #fixed_variables = fixed_variables, 
                                  output_options=out_options,)

    return problem

def max_hybrid_energy(result):
    return -result['annual_energies']['hybrid']

def min_pv_lcoe(result):
    return result['lcoe_real']['pv']

def max_hybrid_npv(result):
    return result['net_present_values']['pv']

if __name__ == '__main__':

    test_init_hybrid_plant = True
    sample_design = False
    save_lhs = True
    read_lhs = False
    reconnect_cache = False
    set_battery_power_based_on_cycle = True

    if sample_design:
        case_str = 'lhs_cm_pvBat_2030Ctargets_carbonCost_upWF_350'
        # Driver config
        driver_config = dict(n_proc=12, eval_limit=1000, cache_dir=case_str+'_cp_cs', reconnect_cache=reconnect_cache)
        driver = OptimizationDriver(init_problem, **driver_config)

        ### Sampling Example

        ## Parametric sweep
        # levels = np.array([1, 1, 1, 1, 1, 1, 2])
        # design = pyDOE.fullfact(levels)
        # levels[levels == 1] = 2
        # ff_scaled = design / (levels - 1)
        #
        ## Latin Hypercube
        n_dim = 7-1   # without battery power
        lhs_scaled = pyDOE.lhs(n_dim, criterion='cm', samples=200)
        if set_battery_power_based_on_cycle:
            # adding battery power assuming that cycle is first and battery power is last
            lhs_scaled = np.insert(lhs_scaled, lhs_scaled.shape[1], list(1 - lhs_scaled[:, 0]), axis=1)

        if save_lhs:
            with open(case_str + '.csv', 'w', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerows(lhs_scaled)

        if read_lhs:
            with open(case_str + '.csv', 'r') as f:
                csv_reader = csv.reader(f)
                rows = []
                for row in csv_reader:
                    row = [float(x) for x in row]
                    rows.append(row)

            lhs_scaled = rows
            
            # lhs_scaled_sb = lhs_scaled[840:1000]
        

        ## Execute Candidates
        # num_evals = driver.sample(ff_scaled, design_name='cp_test')
        num_evals = driver.parallel_sample(lhs_scaled, design_name=case_str)

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
        print(driver.cache_info)

    # Test the initial simulation function
    if test_init_hybrid_plant:
        project_life = 25

        hybrid_plant = init_hybrid_plant()

        start_time = time.time()
        hybrid_plant.simulate(project_life)
        end_time = time.time()
        print("      %6.2f seconds required to simulate system" % (end_time - start_time))

        print("PPA price: {}".format(hybrid_plant.ppa_price[0]))

        if hybrid_plant.tower:
            print("Tower CSP:")
            print("\tEnergy (year 1) [kWh]: {:.2f}".format(hybrid_plant.annual_energies.tower))
            print("\tCapacity Factor: {:.2f}".format(hybrid_plant.capacity_factors.tower))
            print("\tInstalled Cost: {:.2f}".format(hybrid_plant.tower.total_installed_cost))
            print("\tNPV: {:.2f}".format(hybrid_plant.net_present_values.tower))
            print("\tLCOE (nominal): {:.2f}".format(hybrid_plant.lcoe_nom.tower))
            print("\tLCOE (real): {:.2f}".format(hybrid_plant.lcoe_real.tower))
            print("\tIRR : {:.2f}".format(hybrid_plant.internal_rate_of_returns.tower))
            print("\tBenefit Cost Ratio: {:.2f}".format(hybrid_plant.benefit_cost_ratios.tower))
            print("\tCapacity credit [%]: {:.2f}".format(hybrid_plant.capacity_credit_percent.tower))
            print("\tCapacity payment (year 1): {:.2f}".format(hybrid_plant.capacity_payments.tower[1]))

        if hybrid_plant.trough:
            print("Trough CSP:")
            print("\tEnergy (year 1) [kWh]: {:.2f}".format(hybrid_plant.annual_energies.trough))
            print("\tCapacity Factor: {:.2f}".format(hybrid_plant.capacity_factors.trough))
            print("\tInstalled Cost: {:.2f}".format(hybrid_plant.trough.total_installed_cost))
            print("\tNPV: {:.2f}".format(hybrid_plant.net_present_values.trough))
            print("\tLCOE (nominal): {:.2f}".format(hybrid_plant.lcoe_nom.trough))
            print("\tLCOE (real): {:.2f}".format(hybrid_plant.lcoe_real.trough))
            print("\tIRR : {:.2f}".format(hybrid_plant.internal_rate_of_returns.trough))
            print("\tBenefit Cost Ratio: {:.2f}".format(hybrid_plant.benefit_cost_ratios.trough))
            print("\tCapacity credit [%]: {:.2f}".format(hybrid_plant.capacity_credit_percent.trough))
            print("\tCapacity payment (year 1): {:.2f}".format(hybrid_plant.capacity_payments.trough[1]))

        if hybrid_plant.pv:
            print("PV plant:")
            print("\tEnergy (year 1) [kWh]: {:.2f}".format(hybrid_plant.annual_energies.pv))
            print("\tCapacity Factor: {:.2f}".format(hybrid_plant.capacity_factors.pv))
            print("\tInstalled Cost: {:.2f}".format(hybrid_plant.pv.total_installed_cost))
            print("\tNPV: {:.2f}".format(hybrid_plant.net_present_values.pv))
            print("\tLCOE (nominal): {:.2f}".format(hybrid_plant.lcoe_nom.pv))
            print("\tLCOE (real): {:.2f}".format(hybrid_plant.lcoe_real.pv))
            print("\tIRR : {:.2f}".format(hybrid_plant.internal_rate_of_returns.pv))
            print("\tBenefit Cost Ratio: {:.2f}".format(hybrid_plant.benefit_cost_ratios.pv))
            print("\tCapacity credit [%]: {:.2f}".format(hybrid_plant.capacity_credit_percent.pv))
            print("\tCapacity payment (year 1): {:.2f}".format(hybrid_plant.capacity_payments.pv[1]))

        if hybrid_plant.battery:
            print("Battery:")
            print("\tEnergy (year 1) [kWh]: {:.2f}".format(hybrid_plant.annual_energies.battery))
            print("\tInstalled Cost: {:.2f}".format(hybrid_plant.battery.total_installed_cost))
            print("\tNPV: {:.2f}".format(hybrid_plant.net_present_values.battery))
            print("\tLCOE (nominal): {:.2f}".format(hybrid_plant.lcoe_nom.battery))
            print("\tLCOE (real): {:.2f}".format(hybrid_plant.lcoe_real.battery))
            print("\tIRR : {:.2f}".format(hybrid_plant.internal_rate_of_returns.battery))
            print("\tBenefit Cost Ratio: {:.2f}".format(hybrid_plant.benefit_cost_ratios.battery))
            print("\tCapacity credit [%]: {:.2f}".format(hybrid_plant.capacity_credit_percent.battery))
            print("\tCapacity payment (year 1): {:.2f}".format(hybrid_plant.capacity_payments.battery[1]))

        print("Hybrid System:")
        print("\tEnergy (year 1) [kWh]: {:.2f}".format(hybrid_plant.annual_energies.hybrid))
        print("\tCapacity Factor: {:.2f}".format(hybrid_plant.capacity_factors.hybrid))
        print("\tInstalled Cost: {:.2f}".format(hybrid_plant.grid.total_installed_cost))
        print("\tNPV: {:.2f}".format(hybrid_plant.net_present_values.hybrid))
        print("\tLCOE (nominal): {:.2f}".format(hybrid_plant.lcoe_nom.hybrid))
        print("\tLCOE (real): {:.2f}".format(hybrid_plant.lcoe_real.hybrid))
        print("\tIRR : {:.2f}".format(hybrid_plant.internal_rate_of_returns.hybrid))
        print("\tBenefit Cost Ratio: {:.2f}".format(hybrid_plant.benefit_cost_ratios.hybrid))
        print("\tCapacity credit [%]: {:.2f}".format(hybrid_plant.capacity_credit_percent.hybrid))
        print("\tCapacity payment (year 1): {:.2f}".format(hybrid_plant.capacity_payments.hybrid[1]))
        print("\tCurtailment percentage: {:.2f}".format(hybrid_plant.grid.curtailment_percent))

        # BCR Breakdown
        print("\n ======= Benefit Cost Ratio Breakdown ======= \n")
        header = " Term \t\t\t| Hybrid \t| "
        
        if hybrid_plant.tower:
            header += "Tower \t | "
        if hybrid_plant.trough:
            header += "Trough \t | "
        if hybrid_plant.pv:
            header += "PV \t\t | "
        if hybrid_plant.battery:
            header += "Battery \t | "
        print(header)
        
        BCR_terms = {"npv_ppa_revenue": "PPA revenue [$]",
                     "npv_capacity_revenue": "Capacity revenue [$]", 
                     "npv_curtailment_revenue": "Curtail revenue [$]",
                     "npv_fed_pbi_income": "Federal PBI income [$]", 
                     "npv_oth_pbi_income": "Other PBI income [$]", 
                     "npv_salvage_value": "Salvage value [$]", 
                     "npv_sta_pbi_income": "State PBI income [$]",
                     "npv_uti_pbi_income": "Utility PBI income [$]",
                     "npv_annual_costs": "annual costs [$]"}
        
        for term in BCR_terms.keys():
            print_table_metric(hybrid_plant, term, BCR_terms[term])
        
        
        test = hybrid_plant.hybrid_simulation_outputs()

        # solve_times = list(hybrid_plant.dispatch_builder.problem_state.solve_time)
        # with open('xpress_persistent_2.csv', 'w', newline='') as f:
        #     csv_writer = csv.writer(f)
        #     for st in solve_times:
        #         csv_writer.writerow([st])


    # if cache_analysis:
    #     df = pd.read_pickle('test_cp_cs/_dataframe/2021-12-20_17.49.02/study_results.df.gz')



    # tower_dict = hybrid_plant.tower.outputs.ssc_time_series
    # tower_dict.update(hybrid_plant.tower.outputs.dispatch)

    # Print outputs to file
    # df = pd.DataFrame(tower_dict)
    # df.to_csv("tower_data_multipliers.csv")
    # outputs = hybrid_plant.hybrid_outputs(filename='check.csv')

    pass


# outputs = ("annual_energies", "capacity_factors", "lcoe_real", "lcoe_nom", "internal_rate_of_returns", "capacity_payments", "total_revenues", "net_present_values",
#                "benefit_cost_ratios", "energy_values", "energy_purchases_values", "energy_sales_values",
#                "federal_depreciation_totals", "federal_taxes", "cost_installed", "insurance_expenses", "debt_payment", "")

# print("Outputs:")
# res = dict()
# for val in outputs:
#     try:
#         res[val] = str(getattr(hybrid_plant, val))
#     except:
#         pass

# pprint.pprint(res)
