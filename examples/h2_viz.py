import pandas as pd
import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(15, 7))

y = range(df_mean.index.values.shape[0])

for i in y:
    if i % 24 == 0:
        ax.plot(y[i: i+24], df_mean.Uwindspeed[i: i+24], marker="o", label="$ws_u$", c=colors[0])
        ax.fill_between(
            y[i: i+24],
            (df_mean.Uwindspeed - df_ci.Uwindspeed)[i: i+24],
            (df_mean.Uwindspeed + df_ci.Uwindspeed)[i: i+24],
            alpha=0.3, color=colors[0], label="$ws_u$ 95% CI"
        )

        ax.plot(y[i: i+24], df_mean.Vwindspeed[i: i+24], marker="o", label="$ws_v$", c=colors[1])
        ax.fill_between(
            y[i: i+24],
            (df_mean.Vwindspeed - df_ci.Vwindspeed)[i: i+24],
            (df_mean.Vwindspeed + df_ci.Vwindspeed)[i: i+24],
            alpha=0.3, color=colors[1], label="$ws_v$ 95% CI"
        )

        ax.plot(y[i: i+24], df_mean.windspeed[i: i+24], marker="o", label="$ws$", c=colors[2])
        ax.fill_between(
            y[i: i+24],
            (df_mean.windspeed - df_ci.windspeed)[i: i+24],
            (df_mean.windspeed + df_ci.windspeed)[i: i+24],
            alpha=0.3, color=colors[2], label="$ws$ 95% CI"
        )

xticks_major = [x * 24 for x in range(1, 13)]
xticks_minor = list(range(0, 24 * 12, 6))
xlabels_major = [month_map[m / 24].ljust(13) for m in xticks_major]
xlabels_minor = ["", "06", "12", "18"] + ["06", "12", "18"] * 11

ax.set_ylim(0, 16)
ax.set_ylabel("Windspeed (m/s)")

ax.set_xlim(0, 24)
ax.set_xticks(xticks_major)
for t in ax.get_xticklabels():
    t.set_y(-0.05)
ax.set_xticks(xticks_minor, minor=True)
ax.set_xticklabels(xlabels_major, ha="right")
ax.set_xticklabels(xlabels_minor, minor=True)
ax.set_xlabel("Hour of day")