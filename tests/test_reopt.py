from hybrid.reopt import REopt
from math import sin, pi
from tests.data.defaults_data import defaults

import pytest

import os
filepath = os.path.dirname(os.path.abspath(__file__))

def test_ReOPT():

    lat = 39.7555
    lon = -105.2211
    wholesale_rate = 0.15
    wholesale_rate_above_site_load = 0.10
    load = [1000*(sin(x) + pi)for x in range(0, 8760)]
    urdb_label = "5ca3d45ab718b30e03405898" # https://openei.org/apps/IURDB/rate/view/5ca3d45ab718b30e03405898


    defaults['Wind']['Windpower']['Resource']['wind_resource_filename'] = \
        os.path.join("data", "39.7555_-105.2211_windtoolkit_2012_60min_60m.srw")

    reopt = REopt(lat=lat,
                  lon=lon,
                  wholesale_rate_dollar_per_kwh=wholesale_rate,
                  load_profile=load,
                  urdb_label=urdb_label,
                  tech_defaults=defaults,
                  fileout=os.path.join(filepath, "data", "REoptResultsNoExportAboveLoad.json"))
    reopt.set_rate_path(os.path.join(filepath, 'data'))

    pv = reopt.PV
    assert(pv['dc_ac_ratio'] == pytest.approx(1.18, 0.01))
    wind = reopt.Wind
    assert(wind['pbi_us_dollars_per_kwh'] == pytest.approx(0.022))

    results = reopt.get_reopt_results(force_download=False)
    assert(isinstance(results, dict))
    assert (results["outputs"]["Scenario"]["Site"]["Wind"]["size_kw"], pytest.approx(11311.1104, 1))
    assert(results["outputs"]["Scenario"]["Site"]["Financial"]["lcc_us_dollars"], pytest.approx(-8821660.0, 1))
    assert(results["outputs"]["Scenario"]["Site"]["Financial"]["lcc_bau_us_dollars"], pytest.approx(16337412.0, 1))
    assert(results["outputs"]["Scenario"]["Site"]["ElectricTariff"]["year_one_export_benefit_us_dollars"], pytest.approx(4126366.0, 1))

    reopt = REopt(lat=lat,
                  lon=lon,
                  wholesale_rate_dollar_per_kwh=wholesale_rate,
                  wholesale_rate_above_site_load_us_dollars_per_kwh=wholesale_rate_above_site_load,
                  load_profile=load,
                  urdb_label=urdb_label,
                  tech_defaults=defaults,
                  fileout=os.path.join(filepath, "data", "REoptResultsExportAboveLoad.json"))
    results = reopt.get_reopt_results(force_download=False)
    assert (isinstance(results, dict))
    assert (results["outputs"]["Scenario"]["Site"]["Financial"]["lcc_us_dollars"], pytest.approx(-91029018631.0, 1))
    assert (results["outputs"]["Scenario"]["Site"]["Wind"]["size_kw"], pytest.approx(100000000.0, 1))
    assert (results["outputs"]["Scenario"]["Site"]["ElectricTariff"]["year_one_export_benefit_us_dollars"],
            pytest.approx(424331867359.0, 1))