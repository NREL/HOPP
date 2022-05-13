
import csv
import json
from hybrid.sites import SiteInfo, flatirons_site
from hybrid.hybrid_simulation import HybridSimulation

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams['font.size'] = 7.75
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

# TODO: Make this callable from other scripts and use

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
    is_test = False  # Turns off full year dispatch and optimize tower and receiver
    
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

    root = "C:/Users/WHamilt2/Documents/Projects/HOPP/CSP_PV_battery_dispatch_plots/"
    solar_file = root + "34.865371_-116.783023_psmv3_60_tmy.csv"
    prices_file = root + "constant_nom_prices.csv"
    schedule_scale = 100  # MWe
    desired_schedule_file = root + "sample_load_profile_normalized.csv"
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
                    desired_schedule=desired_schedule
                    )

    technologies = {'tower': {
                        'cycle_capacity_kw':  100 * 1000, #100 * 1000,
                        'solar_multiple': 3.0, #2.0,
                        'tes_hours': 16.0, #16.0,
                        'optimize_field_before_sim': not is_test,
                        'scale_input_params': True,
                        },
                    'trough': {
                        'cycle_capacity_kw': 100 * 1000,
                        'solar_multiple': 4.0,
                        'tes_hours': 20.0
                    },
                    'pv': {
                        'system_capacity_kw': 50 * 1000
                        },
                    'battery': {
                        'system_capacity_kwh': 300 * 1000,
                        'system_capacity_kw': 100 * 1000
                        },
                    'grid': 150 * 1000}

    # Create model
    hybrid_plant = HybridSimulation({key: technologies[key] for key in techs_in_sim}, 
                                    site,
                                    interconnect_kw=technologies['grid'],
                                    dispatch_options={
                                        'is_test_start_year': is_test,
                                        'is_test_end_year': is_test,
                                        'solver': 'xpress',
                                        'grid_charging': False,
                                        'pv_charging_only': True,
                                        },
                                    simulation_options={
                                        'storage_capacity_credit': False,
                                    }
                                    )

    # Defaults:
    # {'cost_per_field_generation': 0.5,
    #  'cost_per_field_start_rel': 1.5,
    #  'cost_per_cycle_generation': 2.0,
    #  'cost_per_cycle_start_rel': 40.0,
    #  'cost_per_change_thermal_input': 0.5}

    csp_dispatch_obj_costs = dict()
    csp_dispatch_obj_costs = {
                              'cost_per_field_generation': 0.0, #0.5,
    #                           'cost_per_field_start_rel': 0.0,
    #                           'cost_per_cycle_generation': 2.0,
                              'cost_per_cycle_start_rel': 0.0,
                              'cost_per_change_thermal_input': 0.5}

    # Set CSP costs
    if hybrid_plant.tower:
        hybrid_plant.tower.dispatch.objective_cost_terms.update(csp_dispatch_obj_costs)
        hybrid_plant.tower.value('cycle_max_frac', 1.0)
    if hybrid_plant.trough:
        hybrid_plant.trough.dispatch.objective_cost_terms.update(csp_dispatch_obj_costs)
        hybrid_plant.trough.value('cycle_max_frac', 1.0)

    # if hybrid_plant.battery:
    #     hybrid_plant.battery.dispatch.lifecycle_cost_per_kWh_cycle = 0.0265 / 100
    #     # hybrid_plant.battery.dispatch.lifecycle_cost_per_kWh_cycle = 1e-6

    if hybrid_plant.pv:
        hybrid_plant.pv.dc_degradation = [0.5] * 25
        hybrid_plant.pv.value('array_type', 2)  # 1-axis tracking
        hybrid_plant.pv.value('tilt', 0)        # Tilt for 1-axis

    # This is required if normalized prices are provided
    hybrid_plant.ppa_price = (0.12,)  # $/kWh

    return hybrid_plant


if __name__ == '__main__':
    # TODO: Update name and location if saving
    plot_dispatch_profiles = True
    save_figures = True
    save_location_root = "C:/Users/WHamilt2/Documents/Projects/HOPP/CSP_PV_battery_dispatch_plots/50MWdc_PV/dispatch_load_constant_prices_zero_cycle_start_zero_field_generation/"
    plotname = 'tower100_sm3_tes16_pv50_batt100_hr3_grid100'
    if plot_dispatch_profiles:
        plotname += '_dispatch'

    # Test the initial simulation function
    project_life = 25

    hybrid_plant = init_hybrid_plant()

    hybrid_plant.simulate(project_life)

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
    if hybrid_plant.site.follow_desired_schedule:
        print("\tMissed load [MWh]: {:.2f}".format(sum(hybrid_plant.grid.missed_load[0:8760]) / 1.e3))
        print("\tMissed load percentage: {:.2f}".format(hybrid_plant.grid.missed_load_percentage * 100.0))
        print("\tSchedule curtailed [MWh]: {:.2f}".format(sum(hybrid_plant.grid.schedule_curtailed[0:8760]) / 1.e3))
        print("\tSchedule curtailed percentage: {:.2f}".format(hybrid_plant.grid.schedule_curtailed_percentage * 100.0))

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

    # Plotting dispatch results
    # Set up labels, colors, alpha
    setup = {'tower': ['Tower (MWe)', 'slategrey', 1.0],
             'trough': ['Trough (MWe)', 'slategrey', 1.0],
             'tower_soc': ['TES charge', 'slategrey', 1.0],
             'trough_soc': ['TES charge', 'slategrey', 1.0],

             'pv': ['PV (MWe)', 'navy', 0.75],
             'pv_to_battery': ['PV to battery (MWe)', 'steelblue', 0.75],
             'pv_to_grid': ['PV (MWe)', 'navy', 0.75],

             'battery': ['Battery (MWe)', 'seagreen', 1.0],
             'battery_dispatch': ['Battery (MWe)', 'seagreen', 1.0],
             'battery_soc': ['Battery charge', 'seagreen', 1.0],

             'hybrid': ['Hybrid (MWe)', 'rebeccapurple', 0.75],
             'curtailed': ['Curtailed (MWe)', 'maroon', 0.75],

             'price': ['Price ($/MWhe)', '0.3', 0.8],
             'dni': ['DNI (W/m$^2$)', 'darkgoldenrod', 0.8],
             }

    # Creating results dictionary
    D = dict()
    D['price'] = (np.array(hybrid_plant.grid.value("dispatch_factors_ts"))
                 * hybrid_plant.grid.value("ppa_price_input")[0] * 1e3)
    D['dni'] = np.array(hybrid_plant.site.solar_resource.data['dn'])

    storage_techs = []
    for tech in ['tower', 'trough', 'pv', 'battery', 'grid']:
        if tech in hybrid_plant.power_sources.keys():
            if plot_dispatch_profiles:
                if tech in ['tower', 'trough']:
                    generation = getattr(hybrid_plant, tech).outputs.dispatch['cycle_generation']
                    load = getattr(hybrid_plant, tech).outputs.dispatch['system_load']
                    gen = [g - l for g, l in zip(generation, load)]
                    D[tech] = np.array([x for x in gen])
                elif tech in ['battery']:
                    gen = getattr(hybrid_plant, tech).Outputs.dispatch_P
                    D[tech] = np.array([x for x in gen])
                else:
                    gen = getattr(hybrid_plant, tech).generation_profile
                    if len(gen) > 8760:
                        gen = gen[0:8760]
                    D[tech] = np.array([x / 1e3 for x in gen])  # Convert to MW
            else:
                gen = getattr(hybrid_plant, tech).generation_profile
                if len(gen) > 8760:
                    gen = gen[0:8760]
                D[tech] = np.array([x/1e3 for x in gen])  # Convert to MW

            if tech in ['tower', 'trough']:
                charge_state = getattr(hybrid_plant, tech).outputs.ssc_time_series['e_ch_tes']
                max_charge = getattr(hybrid_plant, tech).tes_capacity
                D[tech+'_soc'] = np.array([x/max_charge for x in charge_state])
                storage_techs.append(tech)
            elif tech in ['battery']:
                D[tech+'_soc'] = np.array([x/100 for x in getattr(hybrid_plant, tech).Outputs.SOC])  # Convert to fraction
                storage_techs.append(tech)

    # --- Calculate curtailed energy
    D['hybrid'] = D['grid']
    total_gen = sum([D[k] for k in D.keys() if
                     k in ['tower', 'trough', 'pv', 'battery']])  # Total generation from individual technologies
    D['curtailed'] = total_gen - D['hybrid']
    D['pv_to_battery'] = -np.minimum(D['battery'], 0.0) if 'battery' in D.keys() else np.zeros_like(D['hybrid'])
    D['battery_dispatch'] = np.maximum(D['battery'], 0.0) if 'battery' in D.keys() else np.zeros_like(D['hybrid'])
    # Note: Some minor curtailment from csp is possible b/c of mismatch b/w dispatch and actual solution
    D['curtailed_from_pv'] = np.minimum(D['curtailed'], D['pv'])
    D['pv_to_grid'] = D['pv'] - D['pv_to_battery'] - D['curtailed_from_pv']

    # --- Get limits for plots
    for k in setup.keys():
        if k in D.keys():
            lims = [0,1] if k in ['tower_soc', 'trough_soc', 'battery_soc'] else [D[k].min(), D[k].max()]
            setup[k].append([min(1.02 * lims[0], 0), lims[1] + 0.05 * (lims[1] - lims[0])])

    def setup_subplots(nrow, ncol, wsub=2.0, hsub=2.0, wspace=0.2, hspace=0.2, left=0.45, right=0.15, bot=0.45,
                       top=0.15, hr=None):
        width = wsub * ncol + (ncol - 1) * wspace + left + right
        height = hsub * nrow + (nrow - 1) * hspace + top + bot
        fig = plt.figure(figsize=(width, height))
        gs = gridspec.GridSpec(nrow, ncol, bottom=bot / height, top=1.0 - (top / height), left=left / width,
                               right=1.0 - (right / width),
                               wspace=wspace / wsub, hspace=hspace / hsub, height_ratios=hr)
        ax = [fig.add_subplot(gs[row, col]) for row in range(nrow) for col in range(ncol)]
        return [fig, ax]


    def create_plot_data(data, stepped=True):  # Create data to plot with stepped lines
        return data if not stepped else np.repeat(data, 2)


    def create_plot_times(nday, stepped=True):
        nperhour = int(len(D['hybrid']) / 8760)  # Number of points per hour
        step = 1. / nperhour  # time step in hr
        times = np.linspace(0, 24 * nday, nperhour * nday * 24, endpoint=False)
        if stepped:
            times = np.append(np.append(times[0], np.repeat(times[1:], 2)), times[-1] + step)
        return times


    def stacked_plot(start_day, nday, stepped=True, savename=None):
        nrow = 2 + len(storage_techs)
        hstack = 1.75
        hsub = 0.6
        htot = hstack + (nrow - 1) * hsub
        havg = htot / nrow
        hr = [hsub / htot for j in range(nrow - 1)] + [hstack / htot]
        fig, axs = setup_subplots(nrow, 1, wsub=1.25 * nday, hsub=havg, hspace=0.15, left=0.55, right=0.55, top=0.1,
                                  hr=hr)
        times = create_plot_times(nday, stepped)
        nperhour = int(len(D['hybrid']) / 8760)
        inds = np.arange(start_day * 24 * nperhour, (start_day + nday) * 24 * nperhour)

        # Plot price/dni, state of charge
        keys = [['price', 'dni']] + [t + '_soc' for t in storage_techs]
        for j in range(len(keys)):
            n = 1 if isinstance(keys[j], str) else len(keys[j])
            for i in range(n):
                k = keys[j] if n == 1 else keys[j][i]
                ax = axs[j] if i == 0 else axs[j].twinx()
                ax.fill_between(times, create_plot_data(D[k][inds], stepped), np.zeros_like(times), color=setup[k][1],
                                alpha=setup[k][2], label=k)
                ax.plot(times, create_plot_data(D[k][inds], stepped), lw=1.0, color=setup[k][1])
                ax.set_ylabel(setup[k][0])
                ax.set_ylim(setup[k][3])

        # Stacked plot of generation
        p = nrow - 1
        order = ['tower', 'trough', 'pv_to_grid', 'pv_to_battery', 'battery_dispatch', 'curtailed']
        if not hybrid_plant.tower:
            order.remove('tower')
        if not hybrid_plant.trough:
            order.remove('trough')
        if not hybrid_plant.battery:
            order.remove('pv_to_battery')
            order.remove('battery_dispatch')

        yprev = np.zeros_like(times)
        for j in range(len(order)):
            k = order[j]
            y = yprev + create_plot_data(D[order[j]][inds], stepped)
            axs[p].fill_between(times, y, yprev, color=setup[k][1], alpha=setup[k][2], lw=0.1, label=setup[k][0])
            yprev = np.maximum(y, 0.0)
        axs[p].set_ylabel('Generation (MWe)')
        ylims = axs[p].get_ylim()
        ylims = [ylims[0], ylims[1] + 0.15 * (ylims[1] - ylims[0])]
        axs[p].set_ylim(ylims)
        axs[p].legend(ncol=5, loc='upper left')

        if hybrid_plant.site.follow_desired_schedule:
            y = create_plot_data(hybrid_plant.site.desired_schedule[int(min(inds)):int(max(inds))+1], stepped)
            axs[p].plot(times, y, color='black', label='Desired Schedule')

        for p in range(len(axs)):
            axs[p].set_xlim(0, nday * 24)
            axs[p].set_xticks(np.arange(0, nday * 24, 6))
            if p == len(axs) - 1:
                axs[p].set_xlabel('Time (hr)')
            else:
                axs[p].set_xticklabels([])
            ylim = axs[p].get_ylim()
            for d in range(nday - 1):
                axs[p].plot([(d + 1) * 24, (d + 1) * 24], [ylim[0], ylim[1]], lw=0.5, color='0.2', ls='--')
        if savename is not None:
            plt.savefig(savename, dpi=400)
        else:
            plt.show()
        return

    # -----------------------------------------------------------------------------
    nday = 5
    # 
    for d in range(0,361,5):
    # for d in [0, 360]:
        stacked_plot(d, nday, savename=save_location_root + plotname + '_day' + str(d) if save_figures else None)