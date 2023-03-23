import numpy as np
import os.path
from pathlib import Path
import pandas as pd


def site_details_creator(desired_lats, desired_lons, year="2012"):
    """
    Creates a "site_details" dataframe for analyzing
    :return: all_sites Dataframe of site_num, lat, lon, solar_filenames, wind_filenames
    """

    if type(desired_lats) == int or type(desired_lats) == float:
        N_lat = 1
    else:
        N_lat = len(desired_lats)

    if type(desired_lons) == int or type(desired_lons) == float:
        N_lon = 1
    else:
        N_lon = len(desired_lons)

    site_nums = np.linspace(1, N_lat * N_lon, N_lat * N_lon)
    site_nums = site_nums.astype(int)
    count = 0
    desired_lons_grid = np.zeros(N_lat * N_lon)
    desired_lats_grid = np.zeros(N_lat * N_lon)
    if N_lat * N_lon == 1:
        desired_lats_grid = [desired_lats]
        desired_lons_grid = [desired_lons]
    else:
        for desired_lon in desired_lons:
            for desired_lat in desired_lats:
                desired_lons_grid[count] = desired_lon
                desired_lats_grid[count] = desired_lat
                count = count + 1

    all_sites = pd.DataFrame(
        {'site_nums': site_nums, 'Lat': desired_lats_grid[:len(desired_lats_grid)],
         'Lon': desired_lons_grid[:len(desired_lons_grid)]})

    # Fill the wind and solar resource locations with blanks (for resource API use)
    solar_filenames = []
    wind_filenames = []
    years = []
    for i in range(len(all_sites)):
        solar_filenames.append('')
        wind_filenames.append('')
        years.append(year)

    all_sites['solar_filenames'] = solar_filenames
    all_sites['wind_filenames'] = wind_filenames
    all_sites['year'] = years

    return all_sites


