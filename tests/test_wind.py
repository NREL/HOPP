import pytest
import defaults.wind_singleowner as wind_defaults
import math
import PySAM.Windpower as windpower

from hybrid.site import Site
from hybrid.power_source import WindPlant


@pytest.fixture
def site():
    lat = 35.2018863
    lon = -101.945027
    verts = [[3.0599999999976717, 288.87000000011176],
             [0.0, 1084.0300000002608],
             [1784.0499999999884, 1084.2400000002235],
             [1794.0900000000256, 999.6399999996647],
             [1494.3400000000256, 950.9699999997392],
             [712.640000000014, 262.79999999981374],
             [1216.9800000000396, 272.3600000003353],
             [1217.7600000000093, 151.62000000011176],
             [708.140000000014, 0.0]]

    # get resource and create model
    return Site(lat, lon, verts)


def test_wind_powercurve():
    model = windpower.default("WindpowerSingleowner")
    model.Turbine.wind_turbine_rotor_diameter = 75

    # calculate system capacity.  To evaluate other turbines, update the defaults dictionary
    model.Turbine.calculate_powercurve(wind_defaults.wind_default_rated_output,
                                       model.Turbine.wind_turbine_rotor_diameter,
                                       wind_defaults.wind_default_max_tip_speed,
                                       wind_defaults.wind_default_max_tip_speed_ratio,
                                       wind_defaults.wind_default_cut_in_speed,
                                       wind_defaults.wind_default_cut_out_speed,
                                       wind_defaults.wind_default_drive_train)

    windspeeds_truth = [round(x, 2) for x in wind_defaults.powercurveWS]
    windspeeds_calc = [round(x, 2) for x in model.Turbine.wind_turbine_powercurve_windspeeds]
    powercurve_truth = [round(x, 0) for x in wind_defaults.powercurveKW]
    powercurve_calc = [round(x, 0) for x in model.Turbine.wind_turbine_powercurve_powerout]

    assert all([a == b for a, b in zip(windspeeds_truth, windspeeds_calc)])
    assert all([a == b for a, b in zip(powercurve_truth, powercurve_calc)])


def test_changing_n_turbines(site):
    wind_site = site

    # test with gridded layout
    model = WindPlant(wind_site, 20000, grid_not_row_layout=True)
    assert(model.system_capacity == 20000)
    for n in range(1, 20):
        model.num_turbines = n
        assert model.num_turbines == n, "n turbs should be " + str(n)
        assert model.system_capacity == pytest.approx(20000, 1), "system capacity different when n turbs " + str(n)

    # test with row layout


def test_changing_rotor_diam(site):
    wind_site = site
    model = WindPlant(wind_site, 20000)
    assert model.system_capacity == 20000
    ratings = range(50, 70, 5)
    for d in ratings:
        model.rotor_diameter = d
        assert model.rotor_diameter == d, "rotor diameter should be " + str(d)
        assert model.turb_rating == 800, "new rating different when rotor diamter is " + str(d)


def test_changing_rotor_diam_recalc(site):
    wind_site = site
    model = WindPlant(wind_site, 19500, size_adjustment='n_turb')
    assert model.system_capacity == 19500
    ratings = range(50, 70, 5)
    for d in ratings:
        model.rotor_diameter = d
        assert model.rotor_diameter == d, "rotor diameter should be " + str(d)
        assert model.turb_rating == 1500, "new rating different when rotor diameter is " + str(d)


def test_changing_turbine_rating(site):
    wind_site = site
    # powercurve scaling
    model = WindPlant(wind_site, 48000)
    n_turbs = model.num_turbines
    for n in range(1000, 3000, 150):
        model.turb_rating = n
        assert model.system_capacity == model.turb_rating * n_turbs, "system size error when rating is " + str(n)


def test_changing_powercurve(site):
    wind_site = site
    # with power curve recalculation requires diameter changes
    model = WindPlant(wind_site, 48000)
    n_turbs = model.num_turbines
    d_to_r = model.rotor_diameter / model.turb_rating
    for n in range(1000, 3001, 500):
        d = math.ceil(n * d_to_r * 1)
        model.modify_powercurve(d, n)
        assert model.turb_rating == pytest.approx(n, 0.1), "turbine rating should be " + str(n)
        assert model.system_capacity == pytest.approx(model.turb_rating * n_turbs, 0.1), "size error when rating is " + str(n)


def test_changing_system_capacity(site):
    wind_site = site
    # adjust number of turbines, system capacity won't be exactly as requested
    model = WindPlant(wind_site, 20000, size_adjustment='n_turb')
    rating = model.turb_rating
    for n in range(1000, 20000, 1000):
        model.system_capacity = n
        assert model.turb_rating == rating, str(n)
        assert model.system_capacity == rating * round(n/rating)

    # adjust turbine rating first, system capacity will be exact
    model = WindPlant(wind_site, 20000, size_adjustment='rating')
    for n in range(1000, 20000, 1000):
        model.system_capacity = n
        assert model.system_capacity == n



