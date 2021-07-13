from math import sin, pi
import pytest
import PySAM.Singleowner as so
import os

from hybrid.sites import *
from hybrid.pv_source import *
from hybrid.wind_source import *
from hybrid.sites import SiteInfo
from hybrid.reopt import REopt
from hybrid.keys import set_nrel_key_dot_env

set_nrel_key_dot_env()
filepath = os.path.dirname(os.path.abspath(__file__))


def test_ReOPT():

    lat = 39.7555
    lon = -105.2211

    # get resource and create model
    site = SiteInfo(flatirons_site)

    load = [1000*(sin(x) + pi)for x in range(0, 8760)]
    urdb_label = "5ca4d1175457a39b23b3d45e" # https://openei.org/apps/IURDB/rate/view/5ca3d45ab718b30e03405898

    solar_model = PVPlant(site, {'system_capacity_kw': 20000})
    wind_model = WindPlant(site, {'num_turbines': 10, "turbine_rating_kw": 2000})
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
    assert(pv['dc_ac_ratio'] == pytest.approx(1.2, 0.01))
    wind = reopt_site['Wind']
    assert(wind['pbi_us_dollars_per_kwh'] == pytest.approx(0.015))

    results = reopt.get_reopt_results(force_download=True)
    assert(isinstance(results, dict))
    print(results["outputs"]["Scenario"]["Site"]["Wind"]['year_one_to_grid_series_kw'])
    assert (results["outputs"]["Scenario"]["Site"]["Wind"]["size_kw"] == pytest.approx(20000, 1))
    assert(results["outputs"]["Scenario"]["Site"]["Financial"]["lcc_us_dollars"] == pytest.approx(17008573.0, 1))
    assert(results["outputs"]["Scenario"]["Site"]["Financial"]["lcc_bau_us_dollars"] == pytest.approx(15511546.0, 1))
    assert(results["outputs"]["Scenario"]["Site"]["ElectricTariff"]["year_one_export_benefit_us_dollars"] == pytest.approx(-15158711.0, 1))

    os.remove(fileout)
