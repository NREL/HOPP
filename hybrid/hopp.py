import math
import multiprocessing

from hybrid.reopt import run_reopt
from hybrid.scenario import run_default_scenario
from hybrid.systems_behavior import get_system_behavior_fx
from hybrid.wind.wind_opt import wind_opt
from parameters.parameter_data import get_input_output_data
from hybrid.optimize_MP import run_optimizer, process_outputs, init


if __name__ == '__main__':

    optimal = []
    def optimal_result(result):
        """
        Evaluate the result against a stored optimal

        Parameters:
        -----------
        result : list
            list of parameters to evaluate
        """
        global optimal
        if not optimal:
            optimal = result
        else:
            if result[0] > optimal[0]:
                optimal = result


def run_hopp(technologies,
             defaults,
             site,
             run_reopt_optimization=True,
             reopt_constraints=None,
             run_wind_layout_opt=True,
             run_system_optimization=True):
    """
    Run the Hybrid Operations and Performance Platform (HOPP),

    Run HOPP by specifying which technologies
    to evaluate, which default values to use, and which components of the system to optimize.

    Parameters
    ----------
    technologies : list
      list of technologies to run, e.g ['Wind', 'Solar', 'Generic']
    defaults : dict
      dictionary of technology defaults, available from defaults.default_data.get_default
    site: dict
      dictionary of site related information, e.g {'lat': 39.0, 'lon': -104.34, ...}
    run_reopt_optimization: bool (optional)
      boolean (true/false) whether to run the reopt optimization for the given default case
    reopt_constraints: dict (optional)
      nested dictionary of constraints to pass to reopt.
    run_wind_layout_opt: bool (optional)
       boolean (true/false) whether to run wind turbine layout optimization
    run_system_optimization: bool (optional)
       boolean (true/false) whether to run optimization of additional system components

    Returns
    -------
    outputs: dict
        A dictionary of optimal outputs
    """

    # Step 0: Define which technologies to evaluate, available technologies are: 'Solar', 'Wind', 'Geothermal'
    # Generic system should always be added to end, which combined the output of other technologies to evaluate
    # total system output and financial performance
    system_behaviors = get_system_behavior_fx(technologies)  # defines which models get run in each system

    # Step 1: Run the resource allocation to get the optimal mix
    # Reopt results are returned as a nested dictionary, also output to results/reopt_results.json for reference
    # If force download is false, REopt will not run if results/reopt_results.json exists
    # REopt will update the defaults data structure with the REopt results for apriori optimization sizes
    if run_reopt_optimization:
        system_behaviors, defaults = run_reopt(lat=site['lat'],
                                      lon=site['lon'],
                                      defaults=defaults,
                                      reopt_constraints=reopt_constraints,
                                      systems=system_behaviors,
                                      force_download=False,
                                      update_scenario=True)

    input_data, output_data = get_input_output_data(system_behaviors)

    # ---- Step 2: Optimal Design --------
    # Optimize Wind Layout (Jen approach)
    # Optimize Solar Layout (Darice approach, or unify with Jen's approach)
    # Consider interactions (figure out shading, current constraint is geometry)
    # - Interaction: geometry constraint (access roads)
    # - Interaction: combined wiring and power electronics
    # - Interaction: general balance-of-station relationships
    # - Interaction: wind shading solar
    # - Interaction: solar heating ground, creating convective currents
    # - Interaction: storage footprint

    # This runs the current default in the unoptimized configuration
    scenario, outputs = run_default_scenario(defaults=defaults,
                                             input_info=input_data,
                                             output_info=output_data,
                                             run_systems=system_behaviors,
                                             print_status=False)

    # Get the optimal wind farm layout and update the scenario
    if run_wind_layout_opt and 'Wind' in scenario.systems:
        turbine_x, turbine_y = wind_opt(scenario, site, plot_bool=False)
        scenario.systems['Wind']['Windpower'].Farm.wind_farm_xCoordinates = turbine_x
        scenario.systems['Wind']['Windpower'].Farm.wind_farm_yCoordinates = turbine_y

    # Reset system behavior to default
    scenario.system_behavior = get_system_behavior_fx(scenario.system_names)

    # This does an optimization of some solar and wind design parameters
    if run_system_optimization:

        print("\n\nOptimizing remaining parameters:\n=========================\n")
        # Optimize the technologies over the range of selected inputs to vary within parameter_data

        #########################################
        # Optimization settings
        #########################################

        # using gridded sampling to test multiprocessing
        tolerance = 1e-4
        n_processes = math.ceil(multiprocessing.cpu_count() / 2)
        n_intervals_per_process_per_axis = 2  # should be an even number for clean division of space

        n_dim = scenario.vec_length
        n_intervals_per_axis = n_processes * n_intervals_per_process_per_axis
        all_outputs = multiprocessing.Array('d', n_intervals_per_axis ** n_dim)

        pool = multiprocessing.Pool(processes=n_processes, initializer=init, initargs=(all_outputs,))
        for i in range(n_processes):
            bounds = (i / n_processes, (i + 1) / n_processes)
            results = pool.apply_async(run_optimizer, args=(defaults, input_data, output_data, system_behaviors, bounds,
                                                            n_intervals_per_axis, tolerance, n_dim, i),
                                       callback=optimal_result)
        pool.close()
        pool.join()

        # print('opt', optimal[0])

        # dim of plot_config_by_index = n_dim
        plot_config_by_index = [0]

        # what is this?
        # if len(plot_config_by_index) != n_dim:
        #    raise ValueError

        outputs = process_outputs(plot_config_by_index, scenario, n_processes, n_intervals_per_axis, n_dim, tolerance, optimal,
                            all_outputs)


    return outputs