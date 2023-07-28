import matplotlib.pyplot as plt


def plot_power_to_load(titletext, df_mean, df_ci, y, ylim, colors, xticks_major, xticks_minor,
                       xlabels_major, xlabels_minor,  save_location):

    fig, ax = plt.subplots(figsize=(15, 10))

    for i in y:
        if i % 24 == 0:
            ax.plot(y[i: i + 24], df_mean.pv_power_production[i: i + 24], marker="o", label="$PV Power$",
                    c=colors[1])
            ax.fill_between(
                y[i: i + 24],
                (df_mean.pv_power_production - df_ci.pv_power_production)[i: i + 24],
                (df_mean.pv_power_production + df_ci.pv_power_production)[i: i + 24],
                alpha=0.3, color=colors[1], label="$PV Power$ 95% CI"
            )

            ax.plot(y[i: i + 24], df_mean.wind_power_production[i: i + 24], marker="o",
                    label="$Wind Power$",
                    c=colors[0])
            ax.fill_between(
                y[i: i + 24],
                (df_mean.wind_power_production - df_ci.wind_power_production)[i: i + 24],
                (df_mean.wind_power_production + df_ci.wind_power_production)[i: i + 24],
                alpha=0.3, color=colors[0], label="$Wind Power$ 95% CI"
            )
            ax.plot(y[i: i + 24], df_mean.combined_pv_wind_power_production[i: i + 24], marker="o",
                    label="$Wind + PV Combined$", c=colors[3])
            ax.fill_between(
                y[i: i + 24],
                (df_mean.combined_pv_wind_power_production - df_ci.combined_pv_wind_power_production)[i: i + 24],
                (df_mean.combined_pv_wind_power_production + df_ci.combined_pv_wind_power_production)[i: i + 24],
                alpha=0.3, color=colors[3], label="$Wind + PV Combined$ 95% CI"
            )

            ax.plot(y[i: i + 24], df_mean.combined_pv_wind_storage_power_production[i: i + 24],
                    marker="o", label="$Wind + PV + Storage Combined$", c=colors[2])
            ax.fill_between(
                y[i: i + 24],
                (df_mean.combined_pv_wind_storage_power_production - df_ci.combined_pv_wind_storage_power_production)[
                i: i + 24],
                (df_mean.combined_pv_wind_storage_power_production + df_ci.combined_pv_wind_storage_power_production)[
                i: i + 24],
                alpha=0.3, color=colors[2], label="$Wind + PV + Storage Combined$ 95% CI"
            )

            ax.plot(y[i: i + 24], df_mean.storage_power_to_load[i: i + 24],
                    marker="o", label="$Storage Power$", c=colors[4])
            ax.fill_between(
                y[i: i + 24],
                (df_mean.storage_power_to_load - df_ci.storage_power_to_load)[i: i + 24],
                (df_mean.storage_power_to_load + df_ci.storage_power_to_load)[i: i + 24],
                alpha=0.3, color=colors[4], label="$Storage Power$ 95% CI"
            )

    ax.set_ylabel("Power (kW)")
    ax.set_xlim(0, 24)
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xticks(xticks_major)
    for t in ax.get_xticklabels():
        t.set_y(-0.05)
    ax.set_xticks(xticks_minor, minor=True)
    ax.set_xticklabels(xlabels_major, ha="right")
    ax.set_xticklabels(xlabels_minor, minor=True)
    ax.set_xlabel("Hour of day")

    ax.set_title(titletext)
    plt.grid(alpha=0.7)
    plt.grid(alpha=0.2, which="minor")

    handles, labels = ax.get_legend_handles_labels()
    labels_set = ["$PV Power$", "$PV Power$ 95% CI",
                  "$Wind Power$", "$Wind Power$ 95% CI",
                  "$Wind + PV Combined$", "$Wind + PV Combined$ 95% CI",
                  "$Wind + PV + Storage Combined$", "$Wind + PV + Storage Combined$ 95% CI",
                  "$Storage Power$", "$Storage Power$ 95% CI"]

    ax.grid(alpha=0.7)
    ax.grid(alpha=0.2, which="minor")

    ix_filter = [labels.index(el) for el in labels_set]
    handles = [handles[ix] for ix in ix_filter]
    labels = [labels[ix] for ix in ix_filter]
    ax.legend(handles, labels, ncol=5, loc="lower left")

    plt.tight_layout()

    # plt.show()
    plt.savefig(save_location, dpi=240, bbox_to_inches="tight")

    plt.close('all')