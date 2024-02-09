from math import sin, pi
import os

import pytest
import PySAM.Singleowner as so
import responses

from hopp.simulation.technologies.pv.pv_plant import PVPlant, PVConfig
from hopp.simulation.technologies.wind.wind_plant import WindPlant, WindConfig
from hopp.simulation.technologies.reopt import REopt

# from tests import TEST_ROOT_DIR
from hopp.utilities.utils_for_tests import create_default_site_info

from tests import TEST_ROOT_DIR

filepath = os.path.dirname(os.path.abspath(__file__))
lat = 39.7555
lon = -105.2211

@responses.activate
def test_ReOPT():
    # Load recorded API responses into registry
    post_path = TEST_ROOT_DIR / "hopp" / "api_responses" / "reopt_response_post.yaml"
    responses._add_from_file(file_path=post_path)
    get_path = TEST_ROOT_DIR / "hopp" / "api_responses" / "reopt_response.yaml"
    responses._add_from_file(file_path=get_path)

    # get resource and create model
    site = create_default_site_info()

    load = [1000*(sin(x) + pi)for x in range(0, 8760)]
    urdb_label = "5ca4d1175457a39b23b3d45e" # https://openei.org/apps/IURDB/rate/view/5ca3d45ab718b30e03405898

    config = PVConfig.from_dict({'system_capacity_kw': 20000})
    solar_model = PVPlant(site, config=config)
    wind_config = WindConfig.from_dict({'num_turbines': 10, "turbine_rating_kw": 2000})
    wind_model = WindPlant(site, config=wind_config)
    wind_model._system_model.Resource.wind_resource_filename = os.path.join(
        "data", "39.7555_-105.2211_windtoolkit_2012_60min_60m.srw")
    fin_model = so.default("GenericSystemSingleOwner")

    fileout = os.path.join(filepath, "REoptResultsNoExportAboveLoad.json")

    reopt = REopt(lat=lat,
                  lon=lon,
                  load_profile=load,
                  urdb_label=urdb_label,
                  solar_model=solar_model,
                  wind_model=wind_model,
                  fin_model=fin_model,
                  interconnection_limit_kw=20000,
                  fileout=fileout)
    reopt.set_rate_path(os.path.join(filepath, 'data'))

    reopt_site = reopt.post['Scenario']['Site']
    pv = reopt_site['PV']
    assert(pv['dc_ac_ratio'] == pytest.approx(1.3, 0.01))
    wind = reopt_site['Wind']
    assert(wind['pbi_us_dollars_per_kwh'] == pytest.approx(0.026))

    results = reopt.get_reopt_results(poll_interval=0)
    assert(isinstance(results, dict))
    print(results["outputs"]["Scenario"]["Site"]["Wind"]['year_one_to_grid_series_kw'])
    if 'error' in results['outputs']['Scenario']["status"]:
        if 'error' in results["messages"].keys():
            if 'Optimization exceeded timeout' in results["messages"]['error']:
                assert True
            else:
                print(results["messages"]['error'])
        elif 'warning' in results["messages"].keys():
            print(results["messages"]['warnings'])
            assert True
    else:
        assert (results["outputs"]["Scenario"]["Site"]["Wind"]["size_kw"] >= 0)

    os.remove(fileout)
