import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plot
import pandas as pd

def print_output_vals(outputs):

    if 'Solar' in outputs:
        print('Annual energy PV' + ' ' + str(outputs['Solar']['annual_energy']/1000) + ' MWh')
        print('Average PV generation' + ' ' + str(round(np.average(outputs['Solar']['gen']) / 1000, 2)) + ' MW')
    if 'Wind' in outputs:
        print('Annual energy Wind' + ' ' + str(outputs['Wind']['annual_energy']/1000) + ' MWh')
        print('Average Wind generation' + ' ' + str(round(np.average(outputs['Wind']['gen']) / 1000, 2)) + ' MW')
        print('Average Wind speed' + ' ' + str(round(np.average(outputs['Wind']['wind_speed']), 2)) + ' m/s')
        print('Average Wind direction' + ' ' + str(round(np.average(outputs['Wind']['wind_direction']), 2)) + ' degrees')

    print('Annual energy Hybrid' + ' ' + str(outputs['Generic']['annual_energy']/1000) + ' MWh')
    print('Average total generation' + ' ' + str(round(np.average(outputs['Generic']['gen'])/1000, 2)) + ' MW')
    print('Hybrid Capacity Factor' + ' ' + str(outputs['Generic']['capacity_factor']))
    print('Hybrid PPA Price Yr 1: $' + ' ' + str(outputs['Generic']['ppa_price'] * 10) + '/MWh')
    print('Hybrid IRR' + ' ' + str(round(outputs['Generic']['analysis_period_irr'], 2)) + '%')
    print('Hybrid NPV' + ' $' + str(round(outputs['Generic']['project_return_aftertax_npv']/1000000, 2)) + ' (millions)')

def save_output_array_vals(outputs, filename=None):

    df_dict = dict()
    for tech in outputs.keys():
        key = tech + ' generation (kW)'
        df_dict[key] = outputs[tech]['gen']

    df = pd.DataFrame(df_dict)

    if filename is not None:
        df.to_csv(filename)

    return df

def plot_2d_slice(plot_config_by_index, all_samples, n_intervals_per_axis, all_outputs, scenario, title):
    # inputs to plot as x and y axis. mark x-axis as -1 and y-axis as -2. otherwise which tick value to set a constant

    x_ind = plot_config_by_index.index(-1)
    y_ind = plot_config_by_index.index(-2)
    constant = [i for i in range(len(plot_config_by_index)) if plot_config_by_index[i] == 0]

    ticks = np.unique(all_samples).tolist()

    results = np.zeros((n_intervals_per_axis, n_intervals_per_axis))

    cnt = 0
    for sample in all_samples:
        keep = True
        for ind in constant:
            if sample[ind] != ticks[plot_config_by_index[ind]]:
                keep = False
        if keep:
            x_pos = ticks.index(sample[x_ind])
            y_pos = ticks.index(sample[y_ind])
            results[x_pos][y_pos] = all_outputs[cnt]
        cnt += 1

    X, Y = np.meshgrid(ticks, ticks)

    x_ind = plot_config_by_index.index(-1)
    y_ind = plot_config_by_index.index(-2)
    x_axis = scenario.parameter_map[x_ind][3] + " (normalized)"
    y_axis = scenario.parameter_map[y_ind][3] + " (normalized)"

    caption = "\n"
    for i in constant:
        caption += scenario.parameter_map[i][3] + " (normalized) at " + str(ticks[plot_config_by_index[i]]) + ". "

    print(results)

    fig = plot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, results)
    ax.set(xlabel=x_axis, ylabel=y_axis)
    ax.set_title(title)
    plot.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=8)

