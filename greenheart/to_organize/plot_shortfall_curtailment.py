import matplotlib.pyplot as plt

def plot_shortfall_curtailment(titletext, df_mean, df_ci, y, ylim, colors, xticks_major, xticks_minor,
                       xlabels_major, xlabels_minor,  save_location):

    fig2, ax2 = plt.subplots(figsize=(15, 10))
    for i in y:
        if i % 24 == 0:
            ax2.plot(y[i: i + 24], df_mean.wind_power_curtailed[i: i + 24], marker="o", label="$Wind Curtailed$", c=colors[0])
            ax2.fill_between(
                y[i: i + 24],
                (df_mean.wind_power_curtailed - df_ci.wind_power_curtailed)[i: i + 24],
                (df_mean.wind_power_curtailed + df_ci.wind_power_curtailed)[i: i + 24],
                alpha=0.3, color=colors[0], label="$Wind Curtailed$ 95% CI"
            )

            ax2.plot(y[i: i + 24], df_mean.pv_power_curtailed[i: i + 24], marker="o", label="$Solar Curtailed$", c=colors[1])
            ax2.fill_between(
                y[i: i + 24],
                (df_mean.pv_power_curtailed - df_ci.pv_power_curtailed)[i: i + 24],
                (df_mean.pv_power_curtailed + df_ci.pv_power_curtailed)[i: i + 24],
                alpha=0.3, color=colors[1], label="$Solar Curtailed$ 95% CI"
            )
            ax2.plot(y[i: i + 24], df_mean.energy_shortfall[i: i + 24], marker="o", label="$Energy Shortfall$", c='red')
            ax2.fill_between(
                y[i: i + 24],
                (df_mean.energy_shortfall - df_ci.energy_shortfall)[i: i + 24],
                (df_mean.energy_shortfall + df_ci.energy_shortfall)[i: i + 24],
                alpha=0.3, color='red', label="$Energy Shortfall$ 95% CI"
            )


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

    handles2, labels2 = ax2.get_legend_handles_labels()
    labels_set2 = ["$Wind Curtailed$", "$Wind Curtailed$ 95% CI",
                   "$Solar Curtailed$", "$Solar Curtailed$ 95% CI",
                   "$Energy Shortfall$", "$Energy Shortfall$ 95% CI"]

    ax2.grid(alpha=0.7)
    ax2.grid(alpha=0.2, which="minor")

    ix_filter2 = [labels2.index(el) for el in labels_set2]
    handles2 = [handles2[ix] for ix in ix_filter2]
    labels2 = [labels2[ix] for ix in ix_filter2]
    ax2.legend(handles2, labels2, ncol=3, loc="lower left")

    fig2.tight_layout()
    # fig2.show()
    fig2.savefig(save_location, dpi=240,
                    bbox_to_inches="tight")
    plt.close('all')