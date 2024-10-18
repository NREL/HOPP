import csv
import json
import functools

from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.hybrid_simulation import HybridSimulation
from alt_dev.optimization_problem_alt import HybridSizingProblem


class DesignProblem:

    def __init__(self, techs_in_sim: list, design_variables: dict = {}, is_test: bool = False):
        """
        Initializes the design problem

        :param techs_in_sim: List of technologies to include in the simulation
        :param design_variables: Nested dictionary containing technologies, variable names, and bounds
        :param is_test: if True, runs dispatch for the first and last 5 days of the year
            and turns off tower and receiver optimization

        """
        self.techs_in_sim = techs_in_sim
        self.is_test = is_test

        if design_variables:
            variables = design_variables
        else:
            csp_vars = {'cycle_capacity_kw': {'bounds': (50 * 1e3, 200 * 1e3)},
                        'solar_multiple': {'bounds': (1.0, 4.0)},
                        'tes_hours': {'bounds': (4, 18)}}

            variables = {'tower': csp_vars,
                         'trough': csp_vars,
                         'pv': {'system_capacity_kw':  {'bounds': (50*1e3,  400*1e3)}},
                         'battery': {'system_capacity_kwh': {'bounds': (50*1e3, 15*200*1e3)},
                                     'system_capacity_kw':  {'bounds': (50*1e3,  200*1e3)}}}

        # Set design variables based on technologies in simulation
        self.design_vars = {key: variables[key] for key in self.techs_in_sim}

        self.out_options = {"dispatch_factors": True,       # add dispatch factors to objective output
                            "generation_profile": True,     # add technology generation profile to output
                            "financial_model": False,       # add financial model dictionary to output
                            "shrink_output": False}         # keep only the first year of output

    def create_problem(self):
        """
        Creates a hybrid sizing problem based on hybrid simulation callable, design variables, and output options

        :returns: HybridSizingProblem
        """
        # hybrid simulation callable initialization for driver
        self.callable_init = functools.partial(init_hybrid_plant, techs_in_sim=self.techs_in_sim, is_test=self.is_test)
        return HybridSizingProblem(self.callable_init, self.design_vars, output_options=self.out_options)

    def get_problem_dimen(self):
        n_dim = 0
        for key in self.design_vars.keys():
            n_dim += len(self.design_vars[key].keys())
        return n_dim

def get_example_path_root():
    return "./examples/CSP_PV_Battery_Analysis/"

def init_hybrid_plant(techs_in_sim: list, is_test: bool = False, ud_techs: dict = {}):
    """
    Initialize hybrid simulation object using specific project inputs
    :param techs_in_sim: List of technologies to include in the simulation
    :param is_test: if True, runs dispatch for the first and last 5 days of the year
        and turns off tower and receiver optimization
    :param ud_techs: Dictionary containing technology initialization parameters required by HybridSimulation

    :return: HybridSimulation as defined for this problem
    """
    schedule_scale = 100  # MWe
    grid_interconnect_mw = 100  # MWe
    example_root = get_example_path_root()

    # Set plant location
    site_data = {
        "lat": 34.8653,
        "lon": -116.7830,
        "elev": 561,
        "tz": 1,
    }
    solar_file = example_root + "02_weather_data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv"
    prices_file = example_root + "03_cost_load_price_data/constant_norm_prices.csv"
    desired_schedule_file = example_root + "03_cost_load_price_data/desired_schedule_normalized.csv"
    # Reading in desired schedule
    with open(desired_schedule_file) as f:
        csvreader = csv.reader(f)
        desired_schedule = []
        for row in csvreader:
            desired_schedule.append(float(row[0])*schedule_scale)

    # If normalized pricing is used, then PPA price must be adjusted after HybridSimulation is initialized
    site = SiteInfo(site_data, 
                    solar_resource_file=solar_file, 
                    grid_resource_file=prices_file,
                    desired_schedule=desired_schedule,
                    wind=False
                    )

    # Load in system costs
    with open(example_root + "03_cost_load_price_data/system_costs_SAM.json") as f:
        cost_info = json.load(f)

    # Initializing technologies
    if ud_techs:
        technologies = ud_techs
    else:
        technologies = {'tower': {
                            'cycle_capacity_kw': 200 * 1000, #100
                            'solar_multiple': 4.0,  #2.0
                            'tes_hours': 20.0,  #14
                            'optimize_field_before_sim': not is_test,
                            'scale_input_params': True,
                            },
                        'trough': {
                            'cycle_capacity_kw': 200 * 1000,
                            'solar_multiple': 6.0,
                            'tes_hours': 28.0
                        },
                        'pv': {
                            'system_capacity_kw': 120 * 1000
                            },
                        'battery': {
                            'system_capacity_kwh': 200 * 1000,
                            'system_capacity_kw': 100 * 1000
                            },
                        'grid': {
                            'interconnect_kw': grid_interconnect_mw * 1000
                            }
                        }

    # Create hybrid simulation class based on the technologies needed in the simulation
    sim_techs = {key: technologies[key] for key in techs_in_sim}
    sim_techs['grid'] = technologies['grid']

    hybrid_plant = HybridSimulation(sim_techs,
                                    site,
                                    dispatch_options={
                                        'is_test_start_year': is_test,
                                        'is_test_end_year': is_test,
                                        'solver': 'cbc',
                                        'grid_charging': False,
                                        'pv_charging_only': True
                                        },
                                    cost_info=cost_info['cost_info']
                                    )

    csp_dispatch_obj_costs = {'cost_per_field_generation': 0.5,
                              'cost_per_field_start_rel': 0.0,
                              'cost_per_cycle_generation': 2.0,
                              'cost_per_cycle_start_rel': 0.0,
                              'cost_per_change_thermal_input': 0.5}

    # Set CSP costs
    if hybrid_plant.tower:
        hybrid_plant.tower.ssc.set(cost_info['tower_costs'])
        hybrid_plant.tower.dispatch.objective_cost_terms = csp_dispatch_obj_costs
    if hybrid_plant.trough:
        hybrid_plant.trough.ssc.set(cost_info['trough_costs'])
        hybrid_plant.trough.dispatch.objective_cost_terms = csp_dispatch_obj_costs

    # Set O&M costs for all technologies
    for tech in ['tower', 'trough', 'pv', 'battery']:
        if not tech in techs_in_sim:
            cost_info["SystemCosts"].pop(tech)

    hybrid_plant.assign(cost_info["SystemCosts"])

    # Set financial parameters for singleowner model
    with open(example_root + '03_cost_load_price_data/financial_parameters_SAM.json') as f:
        fin_info = json.load(f)

    hybrid_plant.assign(fin_info["FinancialParameters"])
    hybrid_plant.assign(fin_info["TaxCreditIncentives"])
    hybrid_plant.assign(fin_info["Revenue"])
    hybrid_plant.assign(fin_info["Depreciation"])
    hybrid_plant.assign(fin_info["PaymentIncentives"])

    # Set specific technology assumptions here
    if hybrid_plant.pv:
        hybrid_plant.pv.dc_degradation = [0.5] * 25
        hybrid_plant.pv.value('array_type', 2)  # 1-axis tracking
        hybrid_plant.pv.value('tilt', 0)        # Tilt for 1-axis

    # This is required if normalized prices are provided
    hybrid_plant.ppa_price = (0.10,)  # $/kWh

    return hybrid_plant