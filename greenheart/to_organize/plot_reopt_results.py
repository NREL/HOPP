from greenheart.to_organize.plot_power_to_load import plot_power_to_load
from greenheart.to_organize.plot_shortfall_curtailment import plot_shortfall_curtailment
from greenheart.to_organize.plot_battery import plot_battery
import numpy as np
import matplotlib.pyplot as plt

def plot_reopt_results(REoptResultsDF, site_name, atb_year, critical_load_factor,
                               useful_life, tower_height,
                               wind_size_mw, solar_size_mw, storage_size_mw, storage_size_mwh, lcoe,
                               feedstock_cost_h2_via_net_cap_cost_lifetime_h2_hopp, feedstock_cost_h2_levelized_hopp,
                               hybrid_installed_cost, h2a_costs,
                               total_unit_cost_of_hydrogen,
                               output_dir,
                               monthly_separation=False, reopt_was_run=False):
    """
    Plots REopt results (wind, solar, battery power) at 24hr averaged periods over the course of a year
    @param REoptResultsDataframe: Dataframe of REOpt results
    @param monthly_separation: (Boolean) determines whether plots are grouped by months or not (alternative is by hour)
    @param reopt_was_run: (Boolean) indicates whether reopt was run or whether dummy data was loaded
    """
    REoptResultsDF.index = REoptResultsDF.Date

    # Determine either monthly or hourly means and 95% CIs
    if monthly_separation:
        # Group by months
        df_mean = REoptResultsDF.groupby(by=[REoptResultsDF.index.month, REoptResultsDF.index.hour]).mean()
        df_std = REoptResultsDF.groupby(by=[REoptResultsDF.index.month, REoptResultsDF.index.hour]).std()
        df_n = REoptResultsDF.groupby(by=[REoptResultsDF.index.month, REoptResultsDF.index.hour]).count()
        z = 1.96
        df_ci = z * df_std / df_n.applymap(np.sqrt)
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        month_map = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }

        mapped = [f"{month_map[m]}-{h}" for m, h in df_mean.index.values]
        y = range(df_mean.index.values.shape[0])

        xticks_major = [x * 24 for x in range(1, 13)]
        xticks_minor = list(range(0, 24 * 12, 6))
        xlabels_major = [month_map[m / 24].ljust(13) for m in xticks_major]
        xlabels_minor = ["", "06", "12", "18"] + ["06", "12", "18", "24"] * 11
    else:
        # Group by hours
        df_mean = REoptResultsDF.groupby(by=[REoptResultsDF.index.hour]).mean()
        df_std = REoptResultsDF.groupby(by=[REoptResultsDF.index.hour]).std()
        df_n = REoptResultsDF.groupby(by=[REoptResultsDF.index.hour]).count()
        z = 1.96
        df_ci = z * df_std / df_n.applymap(np.sqrt)
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        y = range(df_mean.index.values.shape[0])

        xticks_major = [x for x in range(1, 24)]
        xticks_minor = list(range(0, 24))
        xlabels_major = [hr for hr in xticks_major]
        xlabels_minor = xticks_minor

    # Plot 1 - Energy supplied to load
    if not reopt_was_run:
        reopt_not_run_warning = 'WARNING: REOPT WAS NOT RUN. DUMMY DATA LOADED'
    else:
        reopt_not_run_warning = ''

    titletext = '{} \n PV and Wind Power at {} plant | ATB Year {} \n Critical Load Factor (0-1): {:,.2f} \n' \
                ' Wind Size (MW): {:,.2f} | Solar Size (MW): {:,.2f} \n Storage Size (MW): {:,.2f} | Storage Size MWh: {:,.2f} \n' \
                ' HOPP LCOE: {:,.2f}c | H2 Levelized Electrical Cost ($/kg): ${:,.2f} | Hybrid Installed Cost: ${:,.2f} \n' \
                ' H2A Levelized Plant Costs ($/kg): ${:,.2f} | Total Levelized H2 Cost ($/kg): ${:,.2f} \n' \
                ' Total Energy Provided (MWh): {:,.2f} Total Energy Shortfall: {:,.2f} Total Energy Curtailed (MWh) {:,.2f}' \
        .format(reopt_not_run_warning, site_name, atb_year, critical_load_factor, wind_size_mw, solar_size_mw, storage_size_mw, storage_size_mwh,
                lcoe, feedstock_cost_h2_levelized_hopp, hybrid_installed_cost, h2a_costs, total_unit_cost_of_hydrogen,
                np.sum(REoptResultsDF['combined_pv_wind_storage_power_production']),
                np.sum(REoptResultsDF['energy_shortfall']), np.sum(REoptResultsDF['combined_pv_wind_curtailment']))
    filename = "wind_pv_{}_production_atb{}_uselife{}_critlo{}_hh{}.png".format(site_name,
                                                                                     atb_year, useful_life,
                                                                                     critical_load_factor, tower_height)
    save_location = output_dir + filename
    ylim = [0, 10000]
    plot_power_to_load(titletext, df_mean, df_ci, y, ylim,
                       colors, xticks_major, xticks_minor, xlabels_major, xlabels_minor,
                       save_location=save_location)

    # Plot 2 - SHORTFALL & CURTAILMENT
    filename = "wind_pv_{}_shortfall_curtailment_atb{}_uselife{}_critlo{}_hh{}.png".format(site_name,
                                                                                                atb_year, useful_life,
                                                                                                critical_load_factor,
                                                                                                tower_height)
    save_location = output_dir + filename
    ylim = [0, 10000]
    plot_shortfall_curtailment(titletext, df_mean, df_ci, y, ylim,
                               colors, xticks_major, xticks_minor, xlabels_major, xlabels_minor,
                               save_location=save_location)

    # Plot 3 - Battery Only
    filename = "battery_{}_atb{}_uselife{}_critlo{}_hh{}.png".format(site_name,
                                                                          atb_year,
                                                                          useful_life,
                                                                          critical_load_factor,
                                                                          tower_height)
    save_location = output_dir + filename
    ylim = [0, 10000]
    plot_battery(titletext, df_mean, df_ci, y, ylim,
                 colors, xticks_major, xticks_minor, xlabels_major, xlabels_minor,
                 save_location=save_location)