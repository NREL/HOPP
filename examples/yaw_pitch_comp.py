"""
yaw_pitch_comp.py

Compensates for yaw and blade pitch in the GM 1.9 MW turbine, very simply

site, pitch, yaw misalignment = yaw_pitch_comp(site, minutely_wind_dir_filepath, hourly_yaw_pitch_filepath)

site: PySAM Site-class object with hourly wind data imported

minutely_wind_dir_filepath: path to a .csv with minutely wind direciton in degrees
                            (downloaded from M2, matches hours of site year)

hourly_yaw_pitch_filepath:  path to a .csv with hourly turbine yaw and blade pitch
                            (downloaded from Grafana, matches hours of site year)

pitch: array of blade pitch in degrees

yaw_misalignment: array of degree misalignment between wind vector and yaw

"""

import numpy as np
import pandas as pd
from scipy.stats import circmean
from scipy.stats import circstd
import matplotlib.pyplot as plt

def yaw_pitch_comp(site, minutely_wind_dir_filepath, hourly_yaw_pitch_filepath):

    # Import data
    minutely_wind_dir_df = pd.read_csv(minutely_wind_dir_filepath)
    minutely_wind_dir = minutely_wind_dir_df.loc[:,'Direction'].values
    hourly_yaw_pitch_df = pd.read_csv(hourly_yaw_pitch_filepath)
    hourly_yaw = hourly_yaw_pitch_df.loc[:,'Yaw'].values
    hourly_pitch = hourly_yaw_pitch_df.loc[:,'Pitch'].values
    
    # Convert wind direction to hourly
    hourly_wind_dir = np.zeros(8760)
    for idx in range(8760):
        hourly_idxs = np.arange(idx*60,(idx+1)*60+1)
        dirs_rads = minutely_wind_dir[hourly_idxs]/180*np.pi
        hourly_wind_dir[idx] = circmean(dirs_rads)*180/np.pi

    # Convert wind dirction to wind vector (flip 180 degrees)
    hourly_wind_vec = np.zeros(8760)
    for idx in range(8760):
        raw_wind_vec = hourly_wind_dir[idx]-180
        if raw_wind_vec < 0:
            hourly_wind_vec[idx] = raw_wind_vec+360
        else:
            hourly_wind_vec[idx] = raw_wind_vec

    # Calculate yaw misalignment
    yaw_misalignment = np.zeros(8760)
    for idx in range(8760):
        raw_misalignment = np.abs(hourly_yaw[idx]-hourly_wind_vec[idx])
        if raw_misalignment > 180:
            yaw_misalignment[idx] = np.abs(raw_misalignment-360)
        else:
            yaw_misalignment[idx] = raw_misalignment

    # Correct wind speed
    new_wind_data = list()
    raw_wind_data = site.wind_resource.data['data']
    old_wind_array = np.zeros(8760)
    new_wind_array = np.zeros(8760)
    for idx, data in enumerate(raw_wind_data):
        yaw_correction = np.cos(yaw_misalignment[idx]/180*np.pi)
        pitch_correction = np.cos(hourly_pitch[idx]/180*np.pi)
        corrected_wind_speed = data[3]*yaw_correction*pitch_correction
        corrected_wind_speed = np.max([0,corrected_wind_speed])
        newdata = data[0:3]
        newdata.append(corrected_wind_speed)
        new_wind_data.append(newdata)
        old_wind_array[idx] = data[3]
        new_wind_array[idx] = corrected_wind_speed
    site.wind_resource.data['data'] = new_wind_data

    # plt.plot(hourly_yaw)
    # plt.plot(hourly_wind_vec)
    
    # plt.plot(old_wind_array)
    # plt.plot(new_wind_array)
    
    # plt.show()

    return site, hourly_pitch, yaw_misalignment