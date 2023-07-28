import matplotlib.pyplot as plt

def plot_battery(titletext, df_mean, df_ci, y, ylim, colors, xticks_major, xticks_minor,
                       xlabels_major, xlabels_minor,  save_location):

    fig2, ax2 = plt.subplots(figsize=(15, 10))
    ax3 = ax2.twinx()
    for i in y:
        if i % 24 == 0:
            ax3.plot(y[i: i + 24], df_mean.battery_soc_pct[i: i + 24], marker="o", label="$Battery SOC$", c=colors[1])
            ax3.fill_between(
                y[i: i + 24],
                (df_mean.battery_soc_pct - df_ci.battery_soc_pct)[i: i + 24],
                (df_mean.battery_soc_pct + df_ci.battery_soc_pct)[i: i + 24],
                alpha=0.3, color=colors[1], label="$Battery SOC$ 95% CI"
            )

            ax2.plot(y[i: i + 24], df_mean.wind_power_to_battery[i: i + 24], marker="o", label="$Wind to Battery$", c=colors[0])
            ax2.fill_between(
                y[i: i + 24],
                (df_mean.wind_power_to_battery - df_ci.wind_power_to_battery)[i: i + 24],
                (df_mean.wind_power_to_battery + df_ci.wind_power_to_battery)[i: i + 24],
                alpha=0.3, color=colors[0], label="$Wind to Battery$ 95% CI"
            )
            ax2.plot(y[i: i + 24], df_mean.pv_power_to_battery[i: i + 24], marker="o", label="$Solar to Battery$", c='red')
            ax2.fill_between(
                y[i: i + 24],
                (df_mean.pv_power_to_battery - df_ci.pv_power_to_battery)[i: i + 24],
                (df_mean.pv_power_to_battery + df_ci.pv_power_to_battery)[i: i + 24],
                alpha=0.3, color='red', label="$Solar to Battery$ 95% CI"
            )
            ax2.plot(y[i: i + 24], df_mean.storage_power_to_load[i: i + 24], marker="o", label="$Battery to Load$", c='green')
            ax2.fill_between(
                y[i: i + 24],
                (df_mean.storage_power_to_load - df_ci.storage_power_to_load)[i: i + 24],
                (df_mean.storage_power_to_load + df_ci.storage_power_to_load)[i: i + 24],
                alpha=0.3, color='green', label="$Battery to Load$ 95% CI"
            )
            ax2.plot(y[i: i + 24], df_mean.storage_power_to_load[i: i + 24], marker="o", label="$Battery to Grid$", c='cyan')
            ax2.fill_between(
                y[i: i + 24],
                (df_mean.storage_power_to_load - df_ci.storage_power_to_load)[i: i + 24],
                (df_mean.storage_power_to_load + df_ci.storage_power_to_load)[i: i + 24],
                alpha=0.3, color='cyan', label="$Battery to Grid$ 95% CI"
            )

    ax3.set_ylabel("Battery SOC")
    ax3.set_ylim(0, 1)
    ax2.set_ylabel("Power (kW)")
    ax2.set_xlim(0, 24)
    ax2.set_xticks(xticks_major)
    for t in ax2.get_xticklabels():
        t.set_y(-0.05)
    ax2.set_xticks(xticks_minor, minor=True)
    ax2.set_xticklabels(xlabels_major, ha="right")
    ax2.set_xticklabels(xlabels_minor, minor=True)
    ax2.set_xlabel("Hour of day")
    ax2.set_title(titletext)

    handles, labels = ax2.get_legend_handles_labels()
    labels_set = ["$Wind to Battery$", "$Wind to Battery$ 95% CI",
                   "$Solar to Battery$", "$Solar to Battery$ 95% CI",
                   "$Battery to Load$", "$Battery to Load$ 95% CI",
                   "$Battery to Grid$", "$Battery to Grid$ 95% CI"]

    ax2.grid(alpha=0.7)
    ax2.grid(alpha=0.2, which="minor")

    ix_filter = [labels.index(el) for el in labels_set]
    handles = [handles[ix] for ix in ix_filter]
    labels = [labels[ix] for ix in ix_filter]
    ax2.legend(handles, labels, ncol=3, loc="lower left")

    handles2, labels2 = ax3.get_legend_handles_labels()
    labels_set2 = ["$Battery SOC$", "$Battery SOC$ 95% CI",]
    ix_filter2 = [labels2.index(el) for el in labels_set2]
    handles2 = [handles2[ix] for ix in ix_filter2]
    labels2 = [labels2[ix] for ix in ix_filter2]

    ax2.legend(handles, labels, ncol=3, loc="lower left")

    fig2.tight_layout()
    # fig2.show()
    fig2.savefig(save_location, dpi=240,
                    bbox_to_inches="tight")
    plt.close('all')