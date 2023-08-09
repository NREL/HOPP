from pytest import approx
import os
from pathlib import Path
import json
import shutil
from dotenv import load_dotenv
import pandas as pd

from tools.resource.resource_tools import *
from hopp.sites import flatirons_site as sample_site
from examples.H2_Analysis.hopp_for_h2 import hopp_for_h2
from hopp.sites import SiteInfo
from hopp.keys import set_developer_nrel_gov_key

# Set API key
load_dotenv()
NREL_API_KEY = os.getenv("NREL_API_KEY")
set_developer_nrel_gov_key(NREL_API_KEY)  # Set this key manually here if you are not setting it using the .env

class TestHOPPForH2:
    examples_dir = Path(__file__).absolute().parent.parent.parent / "examples" / 'H2_Analysis'
    test_dir = Path(__file__).absolute().parent
    def test_hopp_for_h2(self):
        scenario = {}
        scenario['Useful Life'] = 30
        scenario['Powercurve File'] = self.examples_dir / "powercurve_2018_COE"
        scenario['Tower Height'] = 90
        scenario['Rotor Diameter'] = 116
        scenario['Debt Equity'] = 90
        scenario['ITC Available'] = 'no'
        scenario['PTC Available'] = 'yes'

        interconnection_size_mw = 150
        wind_size_mw = 150
        solar_size_mw = 50
        electrolyzer_size = 5
        storage_size_mw = 50
        storage_size_mwh = 100
        storage_hours = storage_size_mwh / storage_size_mw
        wind_cost_kw = 1454
        solar_cost_kw = 1080
        storage_cost_kw = 101
        storage_cost_kwh = 116
        kw_continuous = 1000 * electrolyzer_size
        load = [kw_continuous for x in range(0, 8760)]
        custom_powercurve = True
        if os.path.exists(scenario['Powercurve File']):
            with open(scenario['Powercurve File'], 'r') as f:
                powercurve_data = json.load(f)
                turb_size = max(powercurve_data['turbine_powercurve_specification']['turbine_power_output'])
        technologies = {
            'pv': {
                'system_capacity_kw': solar_size_mw * 1e3
                },
            'wind': {
                'num_turbines': int(wind_size_mw / turb_size),
                'turbine_rating_kw': turb_size,
                'hub_height': scenario['Tower Height'],
                'rotor_diameter': scenario['Rotor Diameter']
                },
            'battery': {
                'system_capacity_kwh': storage_size_mwh * 1000,
                'system_capacity_kw': storage_size_mw * 1000
                }
            }

        solar_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
        wind_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
        sample_site['site_num'] = 1
        Site = SiteInfo(sample_site, solar_resource_file=solar_resource_file, wind_resource_file=wind_resource_file)

        hybrid_plant, combined_pv_wind_power_production_hopp, combined_pv_wind_curtailment_hopp, \
            energy_shortfall_hopp, annual_energies, wind_plus_solar_npv, npvs, lcoe = hopp_for_h2(
                Site, scenario, technologies, wind_size_mw, solar_size_mw, storage_size_mw, storage_size_mwh,
                storage_hours, wind_cost_kw, solar_cost_kw, storage_cost_kw, storage_cost_kwh, kw_continuous,
                load, custom_powercurve, interconnection_size_mw, grid_connected_hopp=True
            )

        df_produced = pd.DataFrame()
        df_produced['combined_pv_wind_power_production_hopp'] = [sum(combined_pv_wind_power_production_hopp)]
        df_produced['combined_pv_wind_curtailment_hopp'] = [sum(combined_pv_wind_curtailment_hopp)]
        df_produced['energy_shortfall_hopp'] = [sum(energy_shortfall_hopp)]
        df_produced['annual_energies'] = annual_energies
        df_produced['wind_plus_solar_npv'] = wind_plus_solar_npv
        df_produced['npvs'] = npvs
        df_produced['lcoe'] = lcoe

        results_path = os.path.join(self.test_dir, 'results')
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        df_produced.to_csv(os.path.join(results_path, 'hopp_for_h2_test_results_produced.csv'), index=False)
        df_produced = pd.read_csv(os.path.join(results_path, 'hopp_for_h2_test_results_produced.csv'))
        df_expected = pd.read_csv(os.path.join(self.test_dir, 'expected_hopp_for_h2_test_results.csv'))

        assert df_produced['combined_pv_wind_power_production_hopp'].values == approx(df_expected['combined_pv_wind_power_production_hopp'].values, 1e-4)
        assert df_produced['combined_pv_wind_curtailment_hopp'].values == approx(df_expected['combined_pv_wind_curtailment_hopp'].values, 1e-4)
        assert df_produced['energy_shortfall_hopp'].values == approx(df_expected['energy_shortfall_hopp'].values, 1e-3)
        assert df_produced['wind_plus_solar_npv'].values == approx(df_expected['wind_plus_solar_npv'].values, 1e-2)
        shutil.rmtree(results_path)

if __name__=="__main__":
    test = TestHOPPForH2()
    test.test_hopp_for_h2()
