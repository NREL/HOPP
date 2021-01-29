#!/usr/bin/env python
# coding: utf-8

# In[1]:


import netCDF4
import numpy as np
import pandas as pd
import datetime as datetime
import matplotlib.pyplot as plt
from netCDF4 import Dataset

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")


# In[2]:


lat, lon = 53.249, 1.39
start, stop = '2015/03/26', '2017/09/30'
# Source: https://en.wikipedia.org/wiki/Dudgeon_Offshore_Wind_Farm#cite_note-14


# In[3]:


# Print out Dudgeon at 10m wind data for comparison
df_10m = pd.read_csv("dudgeon_weather.csv")
df_10m.describe()


# In[ ]:





# In[4]:


fn_100m = 'data/wind_100m.nc'


# In[5]:


data = Dataset(fn_100m, mode='r')
for key, value in data.variables.items():
    print(key, value.long_name, f'{value[:].mean():.2f} ± {value[:].std():.2f}')


# In[6]:


dt = netCDF4.num2date(data.variables['time'][:], data.variables['time'].units, data.variables['time'].calendar)
dt = [datetime.datetime.strftime(x, '%m/%d/%y %H:%M') for x in dt]
df = pd.DataFrame(
    np.hstack((
        data.variables['u100'][:, 0, 0].data.reshape(-1, 1),
        data.variables['v100'][:, 0, 0].data.reshape(-1, 1)
    )),
    index=dt, columns=['Uwindspeed', 'Vwindspeed'])
df.index.name = 'datetime'
df["windspeed"] = np.sqrt((df.Uwindspeed ** 2) + (df.Vwindspeed ** 2))
df.to_csv('dudgeon_wind100m.csv', index_label='datetime')
df.head()


# In[7]:


df[df.Uwindspeed > 0]["Uwindspeed"].describe()


# In[8]:


df[df.Uwindspeed < 0]["Uwindspeed"].describe()


# In[9]:


df[df.Vwindspeed > 0]["Vwindspeed"].describe()


# In[10]:


df[df.Vwindspeed < 0]["Vwindspeed"].describe()


# In[11]:


df.windspeed.describe()


# In[ ]:





# In[12]:


df.index = pd.to_datetime(df.index)
df.index


# In[13]:


df_mean = df.abs().groupby(by=df.index.hour).mean()
df_std = df.abs().groupby(by=df.index.hour).mean()
df_n = df.abs().groupby(by=df.index.hour).count()

z = 1.96
df_ci = z * df_std / df_n.applymap(np.sqrt)


# In[14]:


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


# In[16]:


fig, ax = plt.subplots(figsize=(10, 5))

y = df_mean.index.values

ax.plot(y, df_mean.Uwindspeed, marker="o", label="$ws_u$", c=colors[0])
ax.fill_between(
    y, 
    df_mean.Uwindspeed - df_ci.Uwindspeed, 
    df_mean.Uwindspeed + df_ci.Uwindspeed, 
    alpha=0.3, color=colors[0], label="$ws_u$ 95% CI"
)

ax.plot(y, df_mean.Vwindspeed, marker="o", label="$ws_v$", c=colors[1])
ax.fill_between(
    y, 
    df_mean.Vwindspeed - df_ci.Vwindspeed, 
    df_mean.Vwindspeed + df_ci.Vwindspeed, 
    alpha=0.3, color=colors[1], label="$ws_v$ 95% CI"
)

ax.plot(y, df_mean.windspeed, marker="o", label="$ws$", c=colors[2])
ax.fill_between(
    y, 
    df_mean.windspeed - df_ci.windspeed, 
    df_mean.windspeed + df_ci.windspeed, 
    alpha=0.3, color=colors[2], label="$ws$ 95% CI"
)

ax.set_ylim(0, 12)
ax.set_ylabel("Windspeed (m/s)")

ax.set_xlim(0, 23)
ax.set_xticks(list(range(0, 24, 2)))
ax.set_xlabel("Hour of day")

ax.set_title("100m Windspeed Distribution at Dudgeon Windfarm\nbetween 2015/03/26 and 2017/09/30")

handles, labels = ax.get_legend_handles_labels()
labels_set = ["$ws_u$", "$ws_u$ 95% CI", "$ws_v$", "$ws_v$ 95% CI", "$ws$", "$ws$ 95% CI"]
ix_filter = [labels.index(el) for el in labels_set]
handles = [handles[ix] for ix in ix_filter]
labels = [labels[ix] for ix in ix_filter]
ax.legend(handles, labels, ncol=3, loc="lower left")

plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig("wind_100m_dudgeon_hour.png", dpi=240, bbox_to_inches="tight")
plt.show()


# In[17]:


df_mean = df.abs().groupby(by=[df.index.month, df.index.hour]).mean()
df_std = df.abs().groupby(by=[df.index.month, df.index.hour]).mean()
df_n = df.abs().groupby(by=[df.index.month, df.index.hour]).count()

z = 1.96
df_ci = z * df_std / df_n.applymap(np.sqrt)


# In[18]:


df_ci


# In[19]:


month_map = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

mapped = [f"{month_map[m]}-{h}" for m, h in df_mean.index.values]


# In[20]:


xticks_major = list(range(1, 13))
xticks_minor = list(range(0, 24))
xlabels = [month_map[m] for m in range(1, 13)]

ax.grid( 'off', axis='x' )
ax.grid( 'off', axis='x', which='minor' )

# vertical alignment of xtick labels
va = [ 0, -.05, 0, -.05, -.05, -.05 ]
for t, y in zip( ax.get_xticklabels( ), va ):
    t.set_y( y )

ax.tick_params( axis='x', which='minor', direction='out', length=30 )
ax.tick_params( axis='x', which='major', bottom='off', top='off' )


# In[21]:


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

ax.set_title("100m Windspeed Distribution at Dudgeon Windfarm\nbetween 2015/03/26 and 2017/09/30")

plt.grid(alpha=0.7)
plt.grid(alpha=0.2, which="minor")

handles, labels = ax.get_legend_handles_labels()
labels_set = ["$ws_u$", "$ws_u$ 95% CI", "$ws_v$", "$ws_v$ 95% CI", "$ws$", "$ws$ 95% CI"]
ix_filter = [labels.index(el) for el in labels_set]
handles = [handles[ix] for ix in ix_filter]
labels = [labels[ix] for ix in ix_filter]
ax.legend(handles, labels, ncol=3, loc="lower left")


plt.tight_layout()
plt.savefig("wind_100m_dudgeon_month_hour.png", dpi=240, bbox_to_inches="tight")
plt.show()


# In[ ]:





# In[ ]:





# In[22]:


df10 = pd.read_csv("dudgeon_weather.csv")
df100 = pd.read_csv("dudgeon_wind100m.csv")


# In[23]:


df100.head()


# In[24]:


df10.head()


# In[25]:


df_weather = df10[["datetime", "waveheight_sig"]].merge(df100[["datetime", "windspeed"]], on="datetime")
df_weather.columns = ["datetime", "windspeed", "waveheight_sig"]
df_weather.rename(columns={'waveheight_sig': 'waveheight'},inplace=True)
df_weather.to_csv("dudgeon_wave_100m_wind.csv", index=False)


# In[26]:


df_weather.head()


# In[ ]:




