import numpy as np
import os.path
from pathlib import Path
import pandas as pd


def resource_loader_file(resource_dir, desired_lats, desired_lons, year="2012"):
    """
    Determines the wind and solar resource files which are nearest the desired_lats and desired_lons and
    adds the site_num, lat, lon, solar_filenames and wind_filenames to the 'all_sites' Dataframe
    :param resource_dir: Resource directory to search for wind and solar resource files
    :param desired_lats: Desired Latitudes
    :param desired_lons: Desired Longitudes
    :return: all_sites Dataframe of site_num, lat, lon, solar_filenames, wind_filenames
    """
    # directory to resource_files
    main_dir = Path(__file__).parent.parent
    npy_dir = main_dir / 'resource_files_big/npy_files/'
    # resource_dir = main_dir / 'resource_files_big'
    solar_dir = (resource_dir / 'solar').resolve()
    wind_dir = (resource_dir / 'wind').resolve()
    if type(desired_lats) == int or type(desired_lats) == float:
        N_lat = 1
    else:
        N_lat = len(desired_lats)

    if type(desired_lons) == int or type(desired_lons) == float:
        N_lon = 1
    else:
        N_lon = len(desired_lons)

    # Get list of files in the directory
    files_solar = []
    files_wind = []
    for file in os.listdir(solar_dir):
        if file.endswith(".csv"):
            if file.rsplit('_')[4].rsplit('.')[0] == str(year):
                files_solar.append(file)

    for file in os.listdir(wind_dir):
        if file.endswith(".srw"):
            if file.rsplit('_')[3] == str(year):
                files_wind.append(file)

    # Get Solar Data
    x_lon_solar = np.zeros(len(files_solar))
    y_lat_solar = np.zeros(len(files_solar))

    for i in range(0, len(files_solar)):
        strFile = solar_dir / files_solar[i]
        df = pd.read_csv(strFile, nrows=1)
        # print(df['Longitude'][0])
        if str(df['Longitude'][0]) == '-101.94':
            x_lon_solar[i] = files_solar[i][:-13].rsplit('_')[1]
        else:
            x_lon_solar[i] = df['Longitude'].values

        if str(df['Latitude'][0]) == '35.21':
            y_lat_solar[i] = files_solar[i].rsplit('_')[0]
        else:
            y_lat_solar[i] = df['Latitude'].values

        # Get Wind Data
        x_lon_wind = np.zeros(len(files_wind))
        y_lat_wind = np.zeros(len(files_wind))

        # get size of the files
        strFile = wind_dir / files_wind[0]

    for i in range(0, len(files_wind)):
        strFile = wind_dir / files_wind[i]
        df = pd.read_csv(strFile, nrows=1)
        if (df.columns[5] == '39.759235') & (df.columns[6] == '-105.21756'):
            y_lat_wind[i] = files_wind[i][:-4].rsplit('t')[1].rsplit('_')[0]
            x_lon_wind[i] = files_wind[i][:-4].rsplit('_')[1]
        else:
            x_lon_wind[i] = float(df.columns[6])
            y_lat_wind[i] = float(df.columns[5])

        # Create site description arrays for Solar and Wind
    solar_sites = pd.DataFrame({'Lat': y_lat_solar[:len(x_lon_solar)], 'Lon': x_lon_solar[:len(x_lon_solar)],
                                'Filename': files_solar[:len(x_lon_solar)]})

    wind_sites = pd.DataFrame({'Lat': y_lat_wind[:len(x_lon_wind)], 'Lon': x_lon_wind[:len(x_lon_wind)],
                               'Filename': files_wind[:len(x_lon_wind)]})

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
    nearest_solar_files = []
    nearest_wind_files = []
    years = []

    # Find the solar and wind files corresponding to the nearest locations to the desired lat/lon
    for i in range(len(all_sites)):
        solar_dist = np.sqrt(
            (all_sites['Lat'][i] - solar_sites['Lat']) ** 2 + (all_sites['Lon'][i] - solar_sites['Lon']) ** 2)
        solar_idx = np.where(solar_dist == np.min(solar_dist))
        nearest_solar_file = os.path.join(solar_dir, solar_sites['Filename'][solar_idx[0][0]])
        nearest_solar_files.append(nearest_solar_file)

        wind_dist = np.sqrt(
            (all_sites['Lat'][i] - wind_sites['Lat']) ** 2 + (all_sites['Lon'][i] - wind_sites['Lon']) ** 2)
        wind_idx = np.where(wind_dist == np.min(wind_dist))
        nearest_wind_file = os.path.join(wind_dir, wind_sites['Filename'][wind_idx[0][0]])
        nearest_wind_files.append(nearest_wind_file)
        years.append(str(year))

    all_sites['solar_filenames'] = nearest_solar_files
    all_sites['wind_filenames'] = nearest_wind_files
    all_sites['year'] = years

    return all_sites


