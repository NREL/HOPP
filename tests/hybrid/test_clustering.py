import pytest
from pytest import approx
import hopp.clustering as clustering
import numpy as np
import pandas as pd
import copy
import csv


def parse_price_data(price_data_file=None):
    price_data = []
    if price_data_file is not None:
        with open(price_data_file) as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                price_data.append(float(row[0]))
    else:
        np.random.seed(0)
        for i in range(int(8)):
            price_data.extend([np.random.rand()]*3)
        P_day = copy.deepcopy(price_data)
        for i in range(364):
            price_data.extend(P_day)
        price_data = [x/10. for x in price_data]  # [$/kWh]
    return price_data

def parse_wind_file(filename, height=None):
    """Outputs a numpy array of wind speeds
    if height is not specified, values from first height are returned"""
    df = pd.read_csv(filename, sep=',', skiprows=2, header=[0,2])
    if height is None:
        column_index = 0
    else:
        heights = [float(x) for x in list(df.columns.get_level_values(1).unique())]     # preserves same order in file
        column_index = heights.index(float(height))
    return (df['Speed'].iloc[:,column_index]).to_numpy()


def test_parse_wind_file():
    # Test single height wind file
    single_height_file = "resource_files/wind/35.2018863_-101.945027_windtoolkit_2012_60min_100m.srw"
    wind_data = parse_wind_file(single_height_file)
    assert len(wind_data) == 8760
    assert sum(wind_data) == approx(75760, 1e-4)

    # Test multiple-height wind file
    multiple_height_file = "resource_files/wind/35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
    wind_data = parse_wind_file(multiple_height_file)       # don't specify height -> should return first height
    assert len(wind_data) == 8760
    assert sum(wind_data) == approx(72098, 1e-4)
    wind_data = parse_wind_file(multiple_height_file, height=80)
    assert len(wind_data) == 8760
    assert sum(wind_data) == approx(72098, 1e-4)
    wind_data = parse_wind_file(multiple_height_file, height=100)
    assert len(wind_data) == 8760
    assert sum(wind_data) == approx(75760, 1e-4)


def test_minimum_specification():
    clusterer = clustering.Clustering(
        power_sources=['tower'],
        solar_resource_file="resource_files/solar/35.2018863_-101.945027_psmv3_60_2012.csv")
    clusterer.run_clustering()
    n_clusters = len(clusterer.clusters['count'])
    assert n_clusters == 20
    assert clusterer.get_sim_start_end_times(0) == (864, 960)
    assert clusterer.get_sim_start_end_times(n_clusters - 1) == (8496, 8592)
    assert list(clusterer.clusters['exemplars']) == \
        [18,  21,  23,  29,  34,  42,  64,  69,  96, 107, 111, 118, 131, 134, 139, 147, 164, 172, 175, 177]


def test_alternate_solar_file():
    clusterer = clustering.Clustering(
        power_sources=['tower'],
        solar_resource_file="resource_files/solar/34.865371_-116.783023_psmv3_60_tmy.csv")
    clusterer.run_clustering()
    n_clusters = len(clusterer.clusters['count'])
    assert n_clusters == 20
    assert clusterer.get_sim_start_end_times(0) == (48, 144)
    assert clusterer.get_sim_start_end_times(n_clusters - 1) == (8016, 8112)
    assert list(clusterer.clusters['exemplars']) == \
        [1, 5, 6, 15, 19, 23, 39, 78, 89, 95, 102, 124, 131, 136, 139, 143, 158, 162, 164, 167]


def test_default_weights_and_divisions():
    clusterer = clustering.Clustering(
        power_sources=['trough'],
        solar_resource_file="resource_files/solar/34.865371_-116.783023_psmv3_60_tmy.csv")
    clusterer.run_clustering()
    n_clusters = len(clusterer.clusters['count'])
    assert n_clusters == 20
    assert clusterer.get_sim_start_end_times(0) == (48, 144)
    assert clusterer.get_sim_start_end_times(n_clusters - 1) == (8016, 8112)
    assert list(clusterer.clusters['exemplars']) == \
        [1, 5, 6, 15, 19, 23, 39, 78, 89, 95, 102, 124, 131, 136, 139, 143, 158, 162, 164, 167]

    clusterer = clustering.Clustering(
        power_sources=['pv'],
        solar_resource_file="resource_files/solar/34.865371_-116.783023_psmv3_60_tmy.csv")
    clusterer.run_clustering()
    n_clusters = len(clusterer.clusters['count'])
    assert n_clusters == 20
    assert clusterer.get_sim_start_end_times(0) == (288, 384)
    assert clusterer.get_sim_start_end_times(n_clusters - 1) == (8640, 8736)
    assert list(clusterer.clusters['exemplars']) == \
        [6, 18, 22, 36, 44, 49, 64, 90, 102, 105, 108, 110, 116, 128, 136, 140, 151, 172, 174, 180]

    clusterer = clustering.Clustering(
        power_sources=['wind'],
        solar_resource_file="resource_files/solar/34.865371_-116.783023_psmv3_60_tmy.csv")
    clusterer.run_clustering()
    n_clusters = len(clusterer.clusters['count'])
    assert n_clusters == 20
    assert clusterer.get_sim_start_end_times(0) == (0, 96)
    assert clusterer.get_sim_start_end_times(n_clusters - 1) == (8400, 8496)
    assert list(clusterer.clusters['exemplars']) == \
        [0, 6, 7, 31, 49, 50, 53, 55, 82, 86, 109, 116, 134, 144, 149, 153, 165, 168, 172, 175]

    clusterer = clustering.Clustering(
        power_sources=['trough', 'pv', 'battery'],
        solar_resource_file="resource_files/solar/34.865371_-116.783023_psmv3_60_tmy.csv")
    clusterer.run_clustering()
    n_clusters = len(clusterer.clusters['count'])
    assert n_clusters == 19
    assert clusterer.get_sim_start_end_times(0) == (48, 144)
    assert clusterer.get_sim_start_end_times(n_clusters - 1) == (8016, 8112)
    assert list(clusterer.clusters['exemplars']) == \
        [1, 2, 23, 27, 34, 39, 57, 58, 89, 95, 102, 124, 131, 147, 155, 158, 163, 164, 167]

    clusterer = clustering.Clustering(
        power_sources=['trough', 'pv'],
        solar_resource_file="resource_files/solar/34.865371_-116.783023_psmv3_60_tmy.csv")
    clusterer.run_clustering()
    n_clusters = len(clusterer.clusters['count'])
    assert n_clusters == 20
    assert clusterer.get_sim_start_end_times(0) == (48, 144)
    assert clusterer.get_sim_start_end_times(n_clusters - 1) == (8016, 8112)
    assert list(clusterer.clusters['exemplars']) == \
        [1, 5, 6, 15, 19, 23, 39, 78, 89, 102, 114, 124, 131, 136, 139, 143, 158, 162, 164, 167]

    clusterer = clustering.Clustering(
        power_sources=['trough', 'wind'],
        solar_resource_file="resource_files/solar/34.865371_-116.783023_psmv3_60_tmy.csv")
    clusterer.run_clustering()
    n_clusters = len(clusterer.clusters['count'])
    assert n_clusters == 20
    assert clusterer.get_sim_start_end_times(0) == (192, 288)
    assert clusterer.get_sim_start_end_times(n_clusters - 1) == (8256, 8352)
    assert list(clusterer.clusters['exemplars']) == \
        [4, 5, 19, 25, 39, 60, 80, 94, 96, 115, 123, 131, 134, 136, 148, 152, 164, 167, 169, 172]

    clusterer = clustering.Clustering(
        power_sources=['pv', 'battery'],
        solar_resource_file="resource_files/solar/34.865371_-116.783023_psmv3_60_tmy.csv")
    clusterer.run_clustering()
    n_clusters = len(clusterer.clusters['count'])
    assert n_clusters == 20
    assert clusterer.get_sim_start_end_times(0) == (384, 480)
    assert clusterer.get_sim_start_end_times(n_clusters - 1) == (8448, 8544)
    assert list(clusterer.clusters['exemplars']) == \
        [8, 18, 22, 27, 58, 76, 92, 102, 105, 109, 116, 122, 125, 130, 136, 144, 153, 167, 172, 176]

    clusterer = clustering.Clustering(
        power_sources=['pv', 'wind'],
        solar_resource_file="resource_files/solar/34.865371_-116.783023_psmv3_60_tmy.csv")
    clusterer.run_clustering()
    n_clusters = len(clusterer.clusters['count'])
    assert n_clusters == 20
    assert clusterer.get_sim_start_end_times(0) == (240, 336)
    assert clusterer.get_sim_start_end_times(n_clusters - 1) == (8256, 8352)
    assert list(clusterer.clusters['exemplars']) == \
        [5, 7, 8, 31, 45, 49, 50, 62, 96, 105, 115, 126, 128, 134, 144, 150, 153, 165, 168, 172]

    clusterer = clustering.Clustering(
        power_sources=['wind', 'battery'],
        solar_resource_file="resource_files/solar/34.865371_-116.783023_psmv3_60_tmy.csv")
    clusterer.run_clustering()
    n_clusters = len(clusterer.clusters['count'])
    assert n_clusters == 20
    assert clusterer.get_sim_start_end_times(0) == (288, 384)
    assert clusterer.get_sim_start_end_times(n_clusters - 1) == (8640, 8736)
    assert list(clusterer.clusters['exemplars']) == \
        [6, 7, 31, 32, 33, 36, 49, 50, 55, 82, 101, 109, 144, 152, 153, 165, 166, 171, 172, 180]

    clusterer = clustering.Clustering(
        power_sources=['pv', 'wind', 'battery'],
        solar_resource_file="resource_files/solar/34.865371_-116.783023_psmv3_60_tmy.csv")
    clusterer.run_clustering()
    n_clusters = len(clusterer.clusters['count'])
    assert n_clusters == 20
    assert clusterer.get_sim_start_end_times(0) == (192, 288)
    assert clusterer.get_sim_start_end_times(n_clusters - 1) == (8256, 8352)
    assert list(clusterer.clusters['exemplars']) == \
        [4, 5, 7, 8, 15, 31, 47, 49, 50, 88, 96, 105, 115, 126, 130, 134, 148, 150, 168, 172]

def test_too_high_clusters():
    clusterer = clustering.Clustering(
        power_sources=['tower'],
        solar_resource_file="resource_files/solar/35.2018863_-101.945027_psmv3_60_2012.csv")
    clusterer.n_cluster = 1000                              # make unachievably high
    clusterer.run_clustering()
    n_clusters = len(clusterer.clusters['count'])
    assert n_clusters == 181
    assert clusterer.get_sim_start_end_times(0) == (0, 96)
    assert clusterer.get_sim_start_end_times(n_clusters - 1) == (8640, 8736)


def test_various_simulation_days():
    clusterer = clustering.Clustering(
        power_sources=['tower'],
        solar_resource_file="resource_files/solar/35.2018863_-101.945027_psmv3_60_2012.csv")

    clusterer.ndays = 1
    clusterer.run_clustering()
    n_clusters = len(clusterer.clusters['count'])
    assert n_clusters == 20
    assert clusterer.get_sim_start_end_times(0) == (72, 144)
    assert clusterer.get_sim_start_end_times(n_clusters - 1) == (8688, 8760)
    assert list(clusterer.clusters['exemplars']) == \
        [3, 7, 8, 23, 42, 65, 84, 117, 126, 190, 237, 254, 259, 281, 282, 294, 306, 310, 333, 362]

    clusterer.ndays = 3
    clusterer.run_clustering()
    n_clusters = len(clusterer.clusters['count'])
    assert n_clusters == 20
    assert clusterer.get_sim_start_end_times(0) == (0, 120)
    assert clusterer.get_sim_start_end_times(n_clusters - 1) == (8424, 8544)
    assert list(clusterer.clusters['exemplars']) == \
        [0, 7, 15, 26, 27, 42, 43, 63, 73, 74, 75, 77, 78, 80, 84, 89, 103, 107, 112, 117]


def test_custom_weights_and_divisions():
    clusterer = clustering.Clustering(
        power_sources=['tower'],
        solar_resource_file="resource_files/solar/35.2018863_-101.945027_psmv3_60_2012.csv")
    clusterer.use_default_weights = True
    clusterer.run_clustering()
    n_clusters = len(clusterer.clusters['count'])
    sim_start_end_times_first = clusterer.get_sim_start_end_times(0)
    sim_start_end_times_last = clusterer.get_sim_start_end_times(n_clusters - 1)
    exemplars = list(clusterer.clusters['exemplars'])
    weights, divisions, bounds = clusterer.get_default_weights()

    # Reinstantiate clusterer object
    clusterer = clustering.Clustering(
        power_sources=['tower'],
        solar_resource_file="resource_files/solar/35.2018863_-101.945027_psmv3_60_2012.csv")
    clusterer.use_default_weights = False
    clusterer.weights = weights
    clusterer.divisions = divisions
    clusterer.bounds = bounds
    clusterer.run_clustering()
    assert len(clusterer.clusters['count']) == n_clusters
    assert clusterer.get_sim_start_end_times(0) == sim_start_end_times_first
    assert clusterer.get_sim_start_end_times(n_clusters - 1) == sim_start_end_times_last
    assert list(clusterer.clusters['exemplars']) == exemplars


def test_initial_state_heuristics():
    # Battery heuristics
    clusterer = clustering.Clustering(
        power_sources=['tower', 'pv', 'battery'],
        solar_resource_file="resource_files/solar/35.2018863_-101.945027_psmv3_60_2012.csv")
    clusterer.run_clustering()
    n_clusters = len(clusterer.clusters['count'])
    cluster_id = 0
    assert clusterer.battery_soc_heuristic(clusterid=cluster_id) == 20
    assert clusterer.battery_soc_heuristic(clusterid=n_clusters-1) == 20
    initial_battery_states = {
        'day': [clusterer.sim_start_days[cluster_id] - 4, clusterer.sim_start_days[cluster_id] - 3],
        'soc': [0, 100],             # state-of-charge [%]
    }
    # Note: sim_start_days has the day of the year (Jan. 1 = 0) of the first 'production' day in each exemplar group, not the 'previous' day.
    #  The initial state given by the heuristic is for the beginning of the 'previous' day.
    #  Therefore, if you want to get the initial state for the cluster ID corresponding to days 2-5 (production days 3 and 4),
    #   the closest states you can provide are for (the beginning of) days 0 and 1, as the calculated initial state would be for
    #   the beginning of day 2, the 'previous' day.
    assert clusterer.battery_soc_heuristic(clusterid=cluster_id, initial_states=initial_battery_states) == approx(98.18, rel=1e-3)

    # CSP heuristics
    clusterer = clustering.Clustering(
        power_sources=['tower', 'pv', 'battery'],
        solar_resource_file="resource_files/solar/35.2018863_-101.945027_psmv3_60_2012.csv")
    clusterer.run_clustering()
    n_clusters = len(clusterer.clusters['count'])
    cluster_id = 0
    assert clusterer.csp_initial_state_heuristic(clusterid=cluster_id) == (10, False, 0.)    # == (initial_soc, is_cycle_on, initial_cycle_load)
    assert clusterer.csp_initial_state_heuristic(clusterid=n_clusters-1) == (10, False, 0.)
    initial_csp_states = {
        'day': [clusterer.sim_start_days[cluster_id] - 4, clusterer.sim_start_days[cluster_id] - 3],
        'soc': [20, 89],              # state-of-charge [%]
        'load': [0, 0],                 # power cycle load [%]
    }
    assert clusterer.csp_initial_state_heuristic(clusterid=cluster_id, solar_multiple=3, initial_states=initial_csp_states) == \
        (approx(89., 1e-3), False, 0.)   # == (initial_soc, is_cycle_on, initial_cycle_load)


def test_price_parameter():
    clusterer = clustering.Clustering(
        power_sources=['tower'],
        solar_resource_file="resource_files/solar/35.2018863_-101.945027_psmv3_60_2012.csv",
        price_data=parse_price_data("resource_files/grid/pricing-data-2015-IronMtn-002_factors.csv"))
    clusterer.run_clustering()
    n_clusters = len(clusterer.clusters['count'])
    assert n_clusters == 20
    assert clusterer.get_sim_start_end_times(0) == (960, 1056)
    assert clusterer.get_sim_start_end_times(n_clusters - 1) == (8496, 8592)
    assert list(clusterer.clusters['exemplars']) == \
        [20, 21, 22, 29, 34, 42, 62, 69, 95, 96, 110, 111, 120, 131, 138, 139, 151, 164, 166, 177]


def test_wind_defaults():
    clusterer = clustering.Clustering(
        power_sources=['wind'],
        solar_resource_file="resource_files/solar/35.2018863_-101.945027_psmv3_60_2012.csv")
    clusterer.run_clustering()
    n_clusters = len(clusterer.clusters['count'])
    assert n_clusters == 20
    assert clusterer.get_sim_start_end_times(0) == (240, 336)
    assert clusterer.get_sim_start_end_times(n_clusters - 1) == (7920, 8016)
    assert list(clusterer.clusters['exemplars']) == \
        [5, 6, 7, 10, 28, 32, 41, 52, 57, 72, 86, 90, 108, 111, 121, 126, 131, 134, 154, 165]


def test_wind_resource_parameter():
    clusterer = clustering.Clustering(
        power_sources=['wind'],
        solar_resource_file="resource_files/solar/35.2018863_-101.945027_psmv3_60_2012.csv",
        wind_resource_data=parse_wind_file("resource_files/wind/35.2018863_-101.945027_windtoolkit_2012_60min_100m.srw"))
    clusterer.run_clustering()
    n_clusters = len(clusterer.clusters['count'])
    assert n_clusters == 20
    assert clusterer.get_sim_start_end_times(0) == (336, 432)
    assert clusterer.get_sim_start_end_times(n_clusters - 1) == (8640, 8736)
    assert list(clusterer.clusters['exemplars']) == \
        [7, 38, 41, 54, 55, 58, 61, 86, 95, 116, 125, 138, 146, 154, 158, 159, 162, 174, 175, 180]


def test_annual_array_from_cluster_exemplars():
    # Run clustering on weather file
    clusterer = clustering.Clustering(
        power_sources=['tower'],
        solar_resource_file="resource_files/solar/34.865371_-116.783023_psmv3_60_tmy.csv")
    clusterer.ndays = 2
    clusterer.run_clustering()

    # Format result to get all days to simulate
    simulation_days = []
    for day in clusterer.sim_start_days:
        simulation_days.append(day + 1)     # change from base 0 to 1 (Jan. 1 is now = 1)
        simulation_days.append(day + 2)

    # Read separately simulated powers (for all days) and zero-out non-exemplar days
    filename = 'tests/hybrid/tower_model_annual_powers.csv'
    df = pd.read_csv(filename, sep=',', header=0, parse_dates=[0])
    df['day_of_year'] = pd.DatetimeIndex(df['timestamp']).dayofyear       # add day of year column
    exemplar_days_indices = df['day_of_year'].isin(simulation_days)         # exemplar days filter
    df = df.assign(system_power_kW_exemplars_only=(df.system_power_kW).where(exemplar_days_indices, 0))

    # Generate powers for all days from only exemplar days
    df['system_power_kW_from_exemplars'] = clusterer.compute_annual_array_from_cluster_exemplar_data(
        list(df['system_power_kW_exemplars_only']))

    # Ensure exemplar days haven't changed
    assert all(df['system_power_kW'][exemplar_days_indices] == df['system_power_kW_from_exemplars'][exemplar_days_indices])

    # Basic tests
    assert len(df['system_power_kW_from_exemplars']) == 8760
    assert min(df['system_power_kW_from_exemplars']) == approx(-4388.5, 1e-4)
    assert max(df['system_power_kW_from_exemplars']) == approx(117783, 1e-4)
    assert sum(df['system_power_kW_from_exemplars']) == approx(566354062, 1e-4)
    
    # Compare annual datas
    assert (sum(df['system_power_kW_from_exemplars']) / \
        sum(df['system_power_kW']) - 1) * 100 == approx(-0.448, 1e-3)                           # percent diff of sums [%]
    assert (((df['system_power_kW_from_exemplars'] - df['system_power_kW'])**2).sum() / \
        len(df['system_power_kW']))**(1/2) == approx(27134, 1e-3)                               # root-mean-square deviation


def test_cluster_avgs_from_timeseries():
    # Run clustering on weather file
    clusterer = clustering.Clustering(
        power_sources=['tower'],
        solar_resource_file="resource_files/solar/34.865371_-116.783023_psmv3_60_tmy.csv")
    clusterer.ndays = 2
    clusterer.run_clustering()

    # Format result to get all days to simulate
    simulation_days = []
    for day in clusterer.sim_start_days:
        simulation_days.append(day + 1)     # change from base 0 to 1 (Jan. 1 is now = 1)
        simulation_days.append(day + 2)

    # Read separately simulated powers (for all days) and zero-out non-exemplar days
    filename = 'tests/hybrid/tower_model_annual_powers.csv'
    df = pd.read_csv(filename, sep=',', header=0, parse_dates=[0])
    df['day_of_year'] = pd.DatetimeIndex(df['timestamp']).dayofyear       # add day of year column
    exemplar_days_indices = df['day_of_year'].isin(simulation_days)         # exemplar days filter
    df = df.assign(system_power_kW_exemplars_only=(df.system_power_kW).where(exemplar_days_indices, 0))

    # Generate powers for all days from only exemplar days
    df['system_power_kW_from_exemplars'] = clusterer.compute_annual_array_from_cluster_exemplar_data(
        list(df['system_power_kW_exemplars_only']))

    # Compute cluster averages
    cluster_averages = clusterer.compute_cluster_avg_from_timeseries(
        list(df['system_power_kW_from_exemplars']))

    assert len(cluster_averages) == len(clusterer.clusters['count'])
    list_lengths = [len(list_) for list_ in cluster_averages]
    assert min(list_lengths) == (1 + clusterer.ndays + 1) * 24 * int(len(df['system_power_kW_from_exemplars']) / 8760)
    assert max(list_lengths) == min(list_lengths)
    assert sum(cluster_averages[0]) == approx(495893, 1e-3)
    assert sum(cluster_averages[-1]) == approx(2734562, 1e-3)
