import matplotlib.pyplot as plt
from hybrid.hybrid_simulation import HybridSimulation

# TODO: This code is for reference only,
#  once plotting functionality exists in main framework this can be deleted


def plot_battery_output(hybrid: HybridSimulation,
                        start_day: int = 0,
                        n_days: int = 5,
                        plot_filename: str = None,
                        font_size: int = 14):

    if not hasattr(hybrid, 'dispatch_builder'):
        raise AttributeError("Simulation with dispatch must be called before plotting battery output.")

    start = start_day * hybrid.site.n_periods_per_day
    end = start + n_days * hybrid.site.n_periods_per_day
    time = range(start, end)
    time_slice = slice(start, end)

    fig, axs = plt.subplots(5, 1, figsize=(15, 10))
    p = 0
    control_attr = 'P'
    if not hybrid.dispatch_builder.options.battery_dispatch == 'simple':
        control_attr = 'I'

    axs[p].plot(time, getattr(hybrid.battery.Outputs, 'dispatch_'+control_attr)[time_slice], 'k', label='Control')
    response = [x/1000. for x in getattr(hybrid.battery.Outputs, control_attr)[time_slice]]
    axs[p].plot(time, response, 'k--', label='Response')
    axs[p].fill_between(time, response, getattr(hybrid.battery.Outputs, 'dispatch_'+control_attr)[time_slice],
                        color='red', alpha=0.5)

    axs[p].set_xlim([start, end])
    axs[p].xaxis.set_ticks(list(range(start, end, hybrid.site.n_periods_per_day)))
    axs[p].grid()
    axs[p].set_ylabel('Control\n& Response', multialignment='center', fontsize=font_size)
    axs[p].tick_params(which='both', labelsize=font_size)
    axs[p].legend(fontsize=font_size - 2, loc='upper left')
    p += 1

    control_error = [r/1000. - c for r, c in zip(getattr(hybrid.battery.Outputs, control_attr)[time_slice],
                                               getattr(hybrid.battery.Outputs, 'dispatch_'+control_attr)[time_slice])]
    axs[p].tick_params(which='both', labelsize=font_size)
    axs[p].fill_between(time, control_error, color='red', alpha=0.5)
    axs[p].plot(time, control_error, 'k')

    axs[p].set_xlim([start, end])
    axs[p].xaxis.set_ticks(list(range(start, end, hybrid.site.n_periods_per_day)))
    axs[p].grid()
    axs[p].set_ylabel('Response\n- Control', multialignment='center', fontsize=font_size)
    p += 1

    axs[p].tick_params(which='both', labelsize=font_size)
    axs[p].plot(time, hybrid.battery.Outputs.SOC[time_slice], 'r', label="Stateful")
    axs[p].fill_between(time, hybrid.battery.Outputs.SOC[time_slice], color='red', alpha=0.5)
    axs[p].plot(time, hybrid.battery.Outputs.dispatch_SOC[time_slice], 'b.', label="Dispatch")

    axs[p].set_xlim([start, end])
    axs[p].xaxis.set_ticks(list(range(start, end, hybrid.site.n_periods_per_day)))
    axs[p].grid()
    axs[p].set_ylabel('Battery\nSOC [%]', multialignment='center', fontsize=font_size)
    axs[p].legend(fontsize=font_size-2, loc='upper left')
    p += 1

    soc_error = [a - d for a, d in zip(hybrid.battery.Outputs.SOC[time_slice],
                                       hybrid.battery.Outputs.dispatch_SOC[time_slice])]
    axs[p].tick_params(which='both', labelsize=font_size)
    axs[p].fill_between(time, soc_error, color='red', alpha=0.5)
    axs[p].plot(time, soc_error, 'k')

    axs[p].set_xlim([start, end])
    axs[p].xaxis.set_ticks(list(range(start, end, hybrid.site.n_periods_per_day)))
    axs[p].grid()
    axs[p].set_ylabel('SOC error\n(act. - disp.)', multialignment='center', fontsize=font_size)
    p += 1

    axs[p].tick_params(which='both', labelsize=font_size)
    axs[p].plot(time, hybrid.battery.Outputs.T_batt[time_slice], 'r')

    axs[p].set_xlim([start, end])
    axs[p].xaxis.set_ticks(list(range(start, end, hybrid.site.n_periods_per_day)))
    axs[p].grid()
    axs[p].set_ylabel('Battery\nTemperature', multialignment='center', fontsize=font_size)

    plt.tight_layout()

    if plot_filename is not None:
        plt.savefig(plot_filename)
        plt.close()


def plot_battery_dispatch_error(hybrid: HybridSimulation,
                                plot_filename: str = None,
                                font_size: int = 14):

    if not hasattr(hybrid, 'dispatch_builder'):
        raise AttributeError("Simulation with dispatch must be called before plotting dispatch error.")

    n_rows = 3
    # First sub-plot SOC
    if hybrid.dispatch_builder.options.battery_dispatch == 'simple':
        n_cols = 2
        fig_width = 10
    else:
        n_cols = 3
        fig_width = 10

    plt.figure(figsize=(15, fig_width))
    sub_plot = 1
    plt.subplot(n_rows, n_cols, sub_plot)
    plt.plot([0, 100.0], [0, 100.0], 'r--')
    plt.scatter(hybrid.battery.Outputs.dispatch_SOC, hybrid.battery.Outputs.SOC, alpha=0.2)
    plt.tick_params(which='both', labelsize=font_size)
    plt.ylabel('SOC\n(state model) [%]', multialignment='center', fontsize=font_size)
    plt.xlabel('SOC (dispatch model) [%]', fontsize=font_size)
    sub_plot += 1

    dispatch_P_MW =  hybrid.battery.Outputs.dispatch_P
    P_MW = [x / 1000. for x in hybrid.battery.Outputs.P]
    plt.subplot(n_rows, n_cols, sub_plot)
    maxpoint = max(max(dispatch_P_MW), max(P_MW))
    minpoint = min(min(dispatch_P_MW), min(P_MW))
    maxpoint *= 1.025
    minpoint *= 1.025
    plt.plot([minpoint, maxpoint], [0, 0], 'k--')
    plt.plot([0, 0], [minpoint, maxpoint], 'k--')
    plt.text(minpoint * 0.40, minpoint, "Charging", fontsize=font_size)
    plt.text(maxpoint * 0.01, maxpoint*0.85, "Discharging", fontsize=font_size)

    plt.plot([minpoint, maxpoint], [minpoint, maxpoint], 'r--')
    plt.scatter(dispatch_P_MW, P_MW, alpha=0.2)
    plt.tick_params(which='both', labelsize=font_size)
    plt.ylabel('Power\n(state model) [MW]', multialignment='center', fontsize=font_size)
    plt.xlabel('Power (dispatch model) [MW]', fontsize=font_size)
    sub_plot += 1

    if not hybrid.dispatch_builder.options.battery_dispatch == 'simple':
        plt.subplot(n_rows, n_cols, sub_plot)
        dispatch_I_kA = [x / 1000. for x in hybrid.battery.Outputs.dispatch_I]
        I_kA = [x / 1000. for x in hybrid.battery.Outputs.I]
        maxpoint = max(max(dispatch_I_kA), max(I_kA))
        minpoint = min(min(dispatch_I_kA), min(I_kA))
        maxpoint *= 1.025
        minpoint *= 1.025
        plt.plot([minpoint, maxpoint], [0, 0], 'k--')
        plt.plot([0, 0], [minpoint, maxpoint], 'k--')
        plt.text(minpoint * 0.40, minpoint, "Charging", fontsize=font_size)
        plt.text(maxpoint * 0.01, maxpoint * 0.85, "Discharging", fontsize=font_size)

        plt.plot([minpoint, maxpoint], [minpoint, maxpoint], 'r--')
        plt.scatter(dispatch_I_kA, I_kA)
        plt.tick_params(which='both', labelsize=font_size)
        plt.ylabel('Current\n(state model) [kA]', multialignment='center', fontsize=font_size)
        plt.xlabel('Current (dispatch model) [kA]', fontsize=font_size)
        sub_plot += 1

    soc_error = [state - dispatch for (state, dispatch) in zip(hybrid.battery.Outputs.SOC, hybrid.battery.Outputs.dispatch_SOC)]
    plt.subplot(n_rows, n_cols, sub_plot)
    plt.hist(soc_error, alpha=0.5)
    plt.tick_params(which='both', labelsize=font_size)
    plt.ylabel('Number\nof Occurrences', multialignment='center', fontsize=font_size)
    plt.xlabel('SOC Error (state) - (dispatch) [%]', fontsize=font_size)
    sub_plot += 1

    plt.subplot(n_rows, n_cols, sub_plot)
    dispatch_P_discharge = [(p > 0) * p for p in dispatch_P_MW]
    dispatch_P_charge = [-(p < 0) * p for p in dispatch_P_MW]
    P_discharge = [(p > 0) * p for p in P_MW]
    P_charge = [-(p < 0) * p for p in P_MW]

    cP_err = [state - dispatch for (state, dispatch) in zip(P_charge, dispatch_P_charge)]
    dcP_err = [state - dispatch for (state, dispatch) in zip(P_discharge, dispatch_P_discharge)]
    min_err = min(cP_err + dcP_err)
    max_err = max(cP_err + dcP_err)
    bins = [x for x in range(int(min_err - 1), int(max_err + 1))]

    plt.hist(cP_err, bins, alpha=0.5, label='Charging')
    plt.hist(dcP_err, bins, alpha=0.5, label='Discharging')

    plt.tick_params(which='both', labelsize=font_size)
    plt.ylabel('Number\nof Occurrences', multialignment='center', fontsize=font_size)
    plt.xlabel('Power Error (state) - (dispatch) [MW]', fontsize=font_size)
    plt.legend(fontsize=font_size-2)
    sub_plot += 1

    if not hybrid.dispatch_builder.options.battery_dispatch == 'simple':
        plt.subplot(n_rows, n_cols, sub_plot)
        dispatch_I_discharge = [(i > 0) * i for i in dispatch_I_kA]
        dispatch_I_charge = [-(i < 0) * i for i in dispatch_I_kA]
        I_discharge = [(i > 0) * i for i in I_kA]
        I_charge = [-(i < 0) * i for i in I_kA]
        cI_err = [state - dispatch for (state, dispatch) in zip(I_charge, dispatch_I_charge)]
        dcI_err = [state - dispatch for (state, dispatch) in zip(I_discharge, dispatch_I_discharge)]
        min_err = min(cI_err + dcI_err)
        max_err = max(cI_err + dcI_err)
        bins = [x for x in range(int(min_err - 1), int(max_err + 1))]

        plt.hist(cI_err, bins, alpha=0.5, label='Charging')
        plt.hist(dcI_err, bins, alpha=0.5, label='Discharging')

        plt.tick_params(which='both', labelsize=font_size)
        plt.ylabel('Number\nof Occurrences', multialignment='center', fontsize=font_size)
        plt.xlabel('Current Error (state) - (dispatch) [kA]', fontsize=font_size)
        plt.legend(fontsize=font_size-2)
        sub_plot += 1

    plt.subplot(n_rows, n_cols, sub_plot)
    plt.scatter(hybrid.battery.Outputs.SOC, soc_error, alpha=0.5)
    plt.plot([0, 100], [0, 0], 'k--')
    plt.tick_params(which='both', labelsize=font_size)
    plt.ylabel('SOC Error\n(state) - (dispatch) [%]', multialignment='center', fontsize=font_size)
    plt.xlabel('SOC (state model) [%]', fontsize=font_size)
    sub_plot += 1

    plt.subplot(n_rows, n_cols, sub_plot)
    plt.scatter(hybrid.battery.Outputs.SOC, cP_err, alpha=0.5, label='Charging')
    plt.scatter(hybrid.battery.Outputs.SOC, dcP_err, alpha=0.5, label='Discharging')
    plt.tick_params(which='both', labelsize=font_size)
    plt.ylabel('Power Error\n(state) - (dispatch) [MW]', multialignment='center', fontsize=font_size)
    plt.xlabel('SOC (state model) [%]', fontsize=font_size)
    plt.legend(fontsize=font_size-2)
    sub_plot += 1

    if not hybrid.dispatch_builder.options.battery_dispatch == 'simple':
        plt.subplot(n_rows, n_cols, sub_plot)
        plt.scatter(hybrid.battery.Outputs.SOC, cI_err, alpha=0.5, label='Charging')
        plt.scatter(hybrid.battery.Outputs.SOC, dcI_err, alpha=0.5, label='Discharging')
        plt.tick_params(which='both', labelsize=font_size)
        plt.ylabel('Current Error\n(state) - (dispatch) [MW]', multialignment='center', fontsize=font_size)
        plt.xlabel('SOC (state model) [%]', fontsize=font_size)
        plt.legend(fontsize=font_size-2)
        sub_plot += 1

    plt.tight_layout()

    if plot_filename is not None:
        plt.savefig(plot_filename)
        plt.close()


def plot_generation_profile(hybrid: HybridSimulation,
                            start_day: int = 0,
                            n_days: int = 5,
                            plot_filename: str = None,
                            font_size: int = 14,
                            power_scale: float = 1/1000,
                            solar_color='r',
                            wind_color='b',
                            discharge_color='b',
                            charge_color='r',
                            gen_color='g',
                            price_color='r'
                            ):

    if not hasattr(hybrid, 'dispatch_builder'):
        raise AttributeError("Simulation with dispatch must be called before plotting generation profile.")

    start = start_day * hybrid.site.n_periods_per_day
    end = start + n_days * hybrid.site.n_periods_per_day
    time = range(start, end)
    time_slice = slice(start, end)

    plt.figure(figsize=(15, 15))

    # First sub-plot (resources)
    gen = [p * power_scale for p in list(hybrid.grid.generation_profile[time_slice])]
    original_gen = [0]*len(gen)
    plt.subplot(3, 1, 1)
    if hybrid.pv:
        solar = hybrid.pv.generation_profile[time_slice]
        original_gen = [og + (s * power_scale) for og, s in zip(original_gen, solar)]
        plt.plot(time, [x * power_scale for x in solar], color=solar_color, label='PV Generation')
    if hybrid.wind:
        wind = hybrid.wind.generation_profile[time_slice]
        original_gen = [og + (w * power_scale) for og, w in zip(original_gen, wind)]
        plt.plot(time, [x * power_scale for x in wind], color=wind_color, label='Wind Farm Generation')
    # plt.plot(time, [x * power_scale for x in ts_wind][st:et], 'b--', label='Wind Farm Resource')
    # plt.plot(time, [x * power_scale for x in ts_solar][st:et], 'r--', label='PV Resource')

    plt.xlim([start, end])
    ax = plt.gca()
    ax.xaxis.set_ticks(list(range(start, end, hybrid.site.n_periods_per_day)))
    plt.grid()
    plt.tick_params(which='both', labelsize=font_size)
    plt.ylabel('Power (MW)', fontsize=font_size)
    plt.title('Generation Resources', fontsize=font_size)
    plt.legend(fontsize=font_size-2, loc='upper left')

    # Battery action
    plt.subplot(3, 1, 2)
    plt.tick_params(which='both', labelsize=font_size)
    discharge = [(p > 0) * p * power_scale for p in hybrid.battery.Outputs.P[time_slice]]
    charge = [(p < 0) * p * power_scale for p in hybrid.battery.Outputs.P[time_slice]]
    plt.bar(time, discharge, width=0.9, color=discharge_color, edgecolor='white', label='Battery Discharge')
    plt.bar(time, charge, width=0.9, color=charge_color, edgecolor='white', label='Battery Charge')
    plt.xlim([start, end])
    ax = plt.gca()
    ax.xaxis.set_ticks(list(range(start, end, hybrid.site.n_periods_per_day)))
    plt.grid()
    ax1 = plt.gca()
    ax1.legend(fontsize=font_size-2, loc='upper left')
    ax1.set_ylabel('Power (MW)', fontsize=font_size)

    ax2 = ax1.twinx()
    ax2.plot(time, hybrid.battery.Outputs.SOC[time_slice], 'k', label='State-of-Charge')
    ax2.plot(time, hybrid.battery.Outputs.dispatch_SOC[time_slice], '.', label='Dispatch')
    ax2.set_ylabel('State-of-Charge (-)', fontsize=font_size)
    ax2.legend(fontsize=font_size-2, loc='upper right')
    plt.title('Battery Power Flow', fontsize=font_size)

    # Net action
    plt.subplot(3, 1, 3)
    plt.tick_params(which='both', labelsize=font_size)
    plt.plot(time, original_gen, 'k--', label='Original Generation')
    plt.plot(time, gen, color=gen_color, label='Optimized Dispatch')
    plt.xlim([start, end])
    ax = plt.gca()
    ax.xaxis.set_ticks(list(range(start, end, hybrid.site.n_periods_per_day)))
    plt.grid()
    ax1 = plt.gca()
    ax1.legend(fontsize=font_size-2, loc='upper left')
    ax1.set_ylabel('Power (MW)', fontsize=font_size)

    ax2 = ax1.twinx()

    price = [p * hybrid.ppa_price[0] for p in hybrid.site.elec_prices.data[time_slice]]
    ax2.plot(time, price, color=price_color, label='Price')
    ax2.set_ylabel('Grid Price ($/kWh)', fontsize=font_size)
    ax2.legend(fontsize=font_size-2, loc='upper right')
    plt.xlabel('Time (hours)', fontsize=font_size)
    plt.title('Net Generation', fontsize=font_size)

    plt.tight_layout()

    if plot_filename is not None:
        plt.savefig(plot_filename)
        plt.close()
    else:
        plt.show()

def plot_battery_generation(hybrid: HybridSimulation,
                            start_day: int = 0,
                            n_days: int = 5,
                            plot_filename: str = None,
                            font_size: int = 14,
                            power_scale: float = 1/1000,
                            discharge_color='b',
                            charge_color='r',
                            gen_color='g',
                            price_color='r'
                            ):

    if not hasattr(hybrid, 'dispatch_builder'):
        raise AttributeError("Simulation with dispatch must be called before plotting generation profile.")

    start = start_day * hybrid.site.n_periods_per_day
    end = start + n_days * hybrid.site.n_periods_per_day
    time = range(start, end)
    time_slice = slice(start, end)

    plt.figure(figsize=(15, 10))
    # Battery action
    plt.subplot(2, 1, 1)
    plt.tick_params(which='both', labelsize=font_size)
    discharge = [(p > 0) * p * power_scale for p in hybrid.battery.Outputs.P[time_slice]]
    charge = [(p < 0) * p * power_scale for p in hybrid.battery.Outputs.P[time_slice]]
    plt.bar(time, discharge, width=0.9, color=discharge_color, edgecolor='white', label='Battery Discharge')
    plt.bar(time, charge, width=0.9, color=charge_color, edgecolor='white', label='Battery Charge')
    plt.xlim([start, end])
    ax = plt.gca()
    ax.xaxis.set_ticks(list(range(start, end, hybrid.site.n_periods_per_day)))
    plt.grid()
    ax1 = plt.gca()
    ax1.legend(fontsize=font_size-2, loc='upper left')
    ax1.set_ylabel('Power (MW)', fontsize=font_size)

    ax2 = ax1.twinx()
    ax2.plot(time, hybrid.battery.Outputs.SOC[time_slice], 'k', label='State-of-Charge')
    ax2.plot(time, hybrid.battery.Outputs.dispatch_SOC[time_slice], '.', label='Dispatch')
    ax2.set_ylabel('State-of-Charge (-)', fontsize=font_size)
    ax2.legend(fontsize=font_size-2, loc='upper right')
    plt.title('Battery Power Flow', fontsize=font_size)

    # Net action
    plt.subplot(2, 1, 2)
    plt.tick_params(which='both', labelsize=font_size)
    original_gen = [b * power_scale for b in list(hybrid.battery.generation_profile[time_slice])]
    gen = [p * power_scale for p in list(hybrid.grid.generation_profile[time_slice])]
    plt.plot(time, original_gen, 'k--', label='Battery Output')
    plt.plot(time, gen, color=gen_color, label='Grid Output')
    plt.xlim([start, end])
    ax = plt.gca()
    ax.xaxis.set_ticks(list(range(start, end, hybrid.site.n_periods_per_day)))
    plt.grid()
    ax1 = plt.gca()
    ax1.legend(fontsize=font_size-2, loc='upper left')
    ax1.set_ylabel('Power (MW)', fontsize=font_size)

    ax2 = ax1.twinx()

    price = [p * hybrid.ppa_price[0] for p in hybrid.site.elec_prices.data[time_slice]]
    ax2.plot(time, price, color=price_color, label='Price')
    ax2.set_ylabel('Grid Price ($/kWh)', fontsize=font_size)
    ax2.legend(fontsize=font_size-2, loc='upper right')
    plt.xlabel('Time (hours)', fontsize=font_size)
    plt.title('Net Generation', fontsize=font_size)

    plt.tight_layout()

    if plot_filename is not None:
        plt.savefig(plot_filename)
        plt.close()
    else:
        plt.show()
