import os
import shutil
from pytest import approx

from tools.resource.resource_tools import *
from tools.resource.resource_loader.resource_loader_files import resource_loader_file

from examples.analysis.single_location import run_all_hybrid_calcs, run_hopp_calc
from examples.analysis import resource_dir


class TestHOPP:
    def test_all_hybrid_calcs(self):
        """
        Test run_hopp_calc function
        """
        # prepare results folder
        parent_path = os.path.abspath(os.path.dirname(__file__))
        main_path = os.path.abspath(os.path.join(parent_path, 'analysis'))
        print("Parent path: ", parent_path)
        print("Main path", main_path)
        results_dir = os.path.join(parent_path, 'results')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        # directory to resource_files
        print("Resource Dir:", resource_dir)

        # Establish Project Scenarios and Parameter Ranges:
        scenario_descriptions = ['Wind Only', 'Solar Only', 'Hybrid - Wind & Solar', 'Solar Addition', 'Wind Overbuild',
                                 'Solar Overbuild', 'Hybrid Vs. Wind + Solar Greenfield']
        bos_details = dict()
        bos_details['BOSSource'] = 'JSONLookup'  # Cost/MW, JSONLookup, HybridBOSSE, HybridBOSSE_manual
        bos_details['BOSFile'] = 'UPDATED_BOS_Summary_Results.json'
        bos_details['BOSScenario'] = 'TBD in analysis'  # Will be set to Wind Only, Solar Only,
        # Variable Ratio Wind and Solar Greenfield, or Solar Addition
        bos_details['BOSScenarioDescription'] = ''  # Blank or 'Overbuild'
        bos_details['Modify Costs'] = True
        bos_details['wind_capex_reduction'] = 0
        bos_details['solar_capex_reduction'] = 0
        bos_details['wind_bos_reduction'] = 0
        bos_details['solar_bos_reduction'] = 0
        bos_details['wind_capex_reduction_hybrid'] = 0
        bos_details['solar_capex_reduction_hybrid'] = 0
        bos_details['wind_bos_reduction_hybrid'] = 0
        bos_details['solar_bos_reduction_hybrid'] = 0

        load_resource_from_file = True
        solar_from_file = True
        wind_from_file = True
        on_land_only = False
        in_usa_only = True  # Only use one of (in_usa / on_land) flags

        # Determine Analysis Locations and Details
        year = 2012
        N_lat = 1  # number of data points
        N_lon = 1
        # desired_lats = np.linspace(23.833504, 49.3556, N_lat)
        # desired_lons = np.linspace(-129.22923, -65.7146, N_lon)
        desired_lats = 35.21
        desired_lons = -101.94


        # Find wind and solar filenames in resource directory
        # which have the closest Lat/Lon to the desired coordinates:
        site_details = resource_loader_file(resource_dir, desired_lats, desired_lons)  # Return contains
        # a DataFrame of [site_num, lat, lon, solar_filenames, wind_filenames]
        site_details.to_csv(os.path.join(resource_dir, 'site_details.csv'))

        # Filtering which sites are included
        site_details = filter_sites(site_details, location='usa only')

        print("Resource Data Loaded")

        solar_tracking_mode = 'Fixed'  # Currently not making a difference
        ppa_prices = [0.05]  # 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        solar_bos_reduction_options = [0]  # 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        hub_height_options = [100]  # 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
        correct_wind_speed_for_height = True
        wind_sizes = [100]
        solar_sizes = [100]
        hybrid_sizes = [200]

        for ppa_price in ppa_prices:
            for solar_bos_reduction in solar_bos_reduction_options:
                for hub_height in hub_height_options:
                    for i, wind_size in enumerate(wind_sizes):
                        wind_size = wind_sizes[i]
                        solar_size = solar_sizes[i]
                        hybrid_size = hybrid_sizes[i]

                        # Establish args for analysis
                        bos_details['solar_bos_reduction_hybrid'] = solar_bos_reduction

                        # Run hybrid calculation for all sites
                        save_all_runs = run_all_hybrid_calcs(site_details, scenario_descriptions, results_dir,
                                                             load_resource_from_file, wind_size,
                                                             solar_size, hybrid_size, bos_details,
                                                             ppa_price, solar_tracking_mode, hub_height,
                                                             correct_wind_speed_for_height)

                        # Save master dataframe containing all locations to .csv
                        all_run_filename = 'All_Runs_{}_WindSize_{}_MW_SolarSize_{}_MW_ppa_price_$' \
                                           '{}_solar_bos_reduction_fraction_{}_{}m_hub_height.csv' \
                            .format(bos_details['BOSScenarioDescription'], wind_size, solar_size, ppa_price,
                                    solar_bos_reduction, hub_height)

                        save_all_runs.to_csv(os.path.join(results_dir,
                                                          all_run_filename))

                        #TODO: Remove this after the expected calc has been produced
                        # save_all_runs.to_csv(os.path.join(path,
                        #                                   'expected_run_all_hybrid_calcs_result.csv'))

                        save_all_runs = pd.DataFrame()  # Reset the save_all_runs dataframe between loops

                        df_produced = pd.read_csv(os.path.join(results_dir,
                                                               all_run_filename))
                        df_expected = pd.read_csv(os.path.join(parent_path, 'expected_run_all_hybrid_calcs_result.csv'))

                        assert df_produced.equals(df_expected)
        shutil.rmtree(results_dir)

    def test_run_hopp_calc(self):
        scenario_description = 'greenfield'

        bos_details = dict()
        bos_details['BOSSource'] = 'JSONLookup'  # Cost/MW, JSONLookup, HybridBOSSE, HybridBOSSE_manual
        bos_details['BOSFile'] = 'UPDATED_BOS_Summary_Results.json'
        bos_details['BOSScenario'] = 'TBD in analysis'  # Will be set to Wind Only, Solar Only,
        # Variable Ratio Wind and Solar Greenfield, or Solar Addition
        bos_details['BOSScenarioDescription'] = ''  # Blank or 'Overbuild'
        bos_details['Modify Costs'] = True
        bos_details['wind_capex_reduction'] = 0
        bos_details['solar_capex_reduction'] = 0
        bos_details['wind_bos_reduction'] = 0
        bos_details['solar_bos_reduction'] = 0
        bos_details['wind_capex_reduction_hybrid'] = 0
        bos_details['solar_capex_reduction_hybrid'] = 0
        bos_details['wind_bos_reduction_hybrid'] = 0
        bos_details['solar_bos_reduction_hybrid'] = 0
        individual_bos_details = bos_details
        solar_size_mw = 100
        wind_size_mw = 100
        total_hybrid_plant_capacity_mw = solar_size_mw + wind_size_mw
        nameplate_mw = 100
        interconnection_size_mw = 100

        resource_filename_solar = os.path.abspath('test_solar_resource.csv')
        resource_filename_wind = os.path.abspath('test_wind_resource.srw')

        load_resource_from_file = True
        ppa_price = 0.05
        results_dir = 'results'
        site = {'Lat': 23.83, 'Lon': -100.17, 'roll_tz': -5, 'resource_filename_solar': resource_filename_solar,
                'resource_filename_wind': resource_filename_wind}
        all_outputs = run_hopp_calc(site, scenario_description, individual_bos_details,
                                    total_hybrid_plant_capacity_mw,
                                    solar_size_mw, wind_size_mw, nameplate_mw, interconnection_size_mw,
                                    load_resource_from_file, ppa_price, results_dir)

        expected_outputs = {'Solar AEP (GWh)': [174.843], 'Wind AEP (GWh)': [395.919],
                            'AEP (GWh)': [570.763], 'Solar Capacity Factor': [19.959],
                            'Wind Capacity Factor': [45.196], 'Capacity Factor': [32.577],
                            'Capacity Factor of Interconnect': [59.714],
                            'Percentage Curtailment': [8.351], 'BOS Cost': [423245759],
                            'BOS Cost percent reduction': [0], 'Cost / MWh Produced': [741.54],
                            'NPV ($-million)': [-163.01], 'IRR (%)': [-1.3928],
                            'PPA Price Used': [0.05], 'LCOE - Real': [6.4218],
                            'LCOE - Nominal': [8.3008], 'Revenue (TOD)': [772673773],
                            'Pearson R Wind V Solar': [-0.003928]}

        for k, v in expected_outputs.items():
            assert(k in all_outputs.keys())
            assert(all_outputs[k] == approx(v, 1e-3))

