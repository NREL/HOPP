import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import math

import hybrid.wind.func_tools as func_tools
import hybrid.wind.opt_tools as opt_tools

from hybrid.scenario import *


def wind_opt(scenario, site, plot_bool, print_status=False):

    # generate x, y locations for new turbine locations
    # ===================================================
    # site boundaries of rhte flatirons campus
    # ===================================================

    print("\n\nOptimizing wind turbine layout:\n=========================\n")

    # get out the vertices of the flatirons campus
    verts_true = site['site_boundaries']['verts']
    verts = site['site_boundaries']['verts_simple']

    x_verts = np.zeros(len(verts_true))
    y_verts = np.zeros(len(verts_true))
    for i in range(len(verts_true)):
        x_verts[i] = verts_true[i][0]
        y_verts[i] = verts_true[i][1]

    # ===================================================
    # setup the Scenario to only run the Wind tech system
    # ===================================================
    def run_wind_model(systems):
        windmodel = systems['Wind']['Windpower']
        windmodel.Farm.system_capacity = max(windmodel.Turbine.wind_turbine_powercurve_powerout) \
                                         * len(windmodel.Farm.wind_farm_xCoordinates)
        windmodel.execute()

    def skip_model(_):
        pass

    original_system_behavior = scenario.system_behavior

    wind_only_system_behavior = {}
    for k, v in original_system_behavior.items():
        if k == 'Wind':
            wind_only_system_behavior['Wind'] = run_wind_model
        else:
            wind_only_system_behavior[k] = skip_model

    scenario.system_behavior = wind_only_system_behavior

    # ===================================================
    # get the random initial turbine locations
    # ===================================================
    wind_size_kw = scenario.systems["Wind"]["Windpower"].Farm.system_capacity
    turbine_output_kw = max(scenario.systems["Wind"]["Windpower"].Turbine.wind_turbine_powercurve_powerout)
    n_turbines = math.ceil(wind_size_kw / turbine_output_kw)


    D = scenario.systems['Wind']['Windpower'].Turbine.wind_turbine_rotor_diameter
    min_spacing = 2 * D
    x, y = func_tools.findTurbines(verts_true, n_turbines, min_spacing, D)
    x0 = np.concatenate([x, y])
    if print_status:
        print(x0)

    # run the optimization
    x_opt, power_opt = opt_tools.layout_opt(x0, scenario, verts)

    if plot_bool:
        plt.figure()
        # func_tools.plot_site(verts,'ko-','Simplified')
        func_tools.plot_site(verts_true, 'r--', 'True')
        plt.legend()
        plt.tick_params(which='both', labelsize=15)
        plt.xlabel('x (m)', fontsize=15)
        plt.ylabel('y (m)', fontsize=15)
        # plt.close()

        plt.grid()
        plt.tick_params(which='both', labelsize=15)
        plt.xlabel('x (m)', fontsize=15)
        plt.ylabel('y (m)', fontsize=15)
        # plt.close()

        # plt.figure()
        # func_tools.plot_site(verts,'ko-','Simplified')
        # func_tools.plot_site(verts_true,'r--','True')
        plt.plot(x, y, 'go', label='Initial Layout')
        plt.plot(x_opt[0:n_turbines], x_opt[n_turbines:], 'go', alpha=0.5, label='Opt Layout')
        plt.legend()
        plt.xlabel('x (m)', fontsize=15)
        plt.savefig(os.path.join('wind_layout.png'), dpi=300, bbox_inches='tight')

    # reset system behavior
    scenario.system_behavior = original_system_behavior

    return x_opt[0:n_turbines].tolist(), x_opt[n_turbines:].tolist()