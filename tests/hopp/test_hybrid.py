from pytest import approx, fixture
from pathlib import Path

from hopp.sites import SiteInfo, flatirons_site
from hopp.layout.hybrid_layout import WindBoundaryGridParameters, PVGridParameters
from hopp.hybrid_simulation import HybridSimulation


@fixture
def site():
    solar_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "solar" / "35.2018863_-101.945027_psmv3_60_2012.csv"
    wind_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m_100m.srw"
    return SiteInfo(flatirons_site, solar_resource_file=solar_resource_file, wind_resource_file=wind_resource_file)

@fixture
def wavesite():
    data = {
    "lat": 44.6899,
    "lon": 124.1346,
    "year": 2010,
    "tz": -7,
    'no_solar': "True",
    'no_wind': "True"
    }
    wave_resource_file = Path(__file__).absolute().parent.parent.parent / "resource_files" / "wave" / "Wave_resource_timeseries.csv"
    return(SiteInfo(data,wave_resource_file=wave_resource_file))
interconnection_size_kw = 15000
pv_kw = 5000
wind_kw = 10000
batt_kw = 5000
technologies = {'pv': {
                    'system_capacity_kw': pv_kw,
                    'layout_params': PVGridParameters(x_position=0.5,
                                                      y_position=0.5,
                                                      aspect_power=0,
                                                      gcr=0.5,
                                                      s_buffer=2,
                                                      x_buffer=2)
                },
                'wind': {
                    'num_turbines': 5,
                    'turbine_rating_kw': wind_kw / 5,
                    'layout_mode': 'boundarygrid',
                    'layout_params': WindBoundaryGridParameters(border_spacing=2,
                                                                border_offset=0.5,
                                                                grid_angle=0.5,
                                                                grid_aspect_power=0.5,
                                                                row_phase_offset=0.5)
                },
                'wave':{
                    'device_rating_kw': 286, 
                    'num_devices': 100, 
                    'wave_power_matrix': [
                                [0.000000, 0.500000, 1.500000, 2.500000, 3.500000, 4.500000, 5.500000, 6.500000, 7.500000, 8.500000, 9.500000, 10.500000, 11.500000, 12.500000, 13.500000, 14.500000, 15.500000, 16.500000, 17.500000, 18.500000, 19.500000, 20.500000],
                                [0.250000, 0.000000, 0.000000, 0.000000, 0.000000, 0.400000, 0.600000, 0.800000, 1.000000, 1.100000, 1.100000, 1.000000, 0.800000, 0.700000, 0.600000, 0.500000, 0.400000, 0.300000, 0.300000, 0.200000, 0.200000, 0.000000],
                                [0.750000, 0.000000, 0.000000, 0.000000, 0.000000, 3.200000, 5.300000, 7.400000, 9.100000, 9.800000, 9.500000, 8.600000, 7.400000, 6.200000, 5.100000, 4.100000, 3.400000, 2.800000, 2.300000, 1.900000, 1.600000, 0.000000],
                                [1.250000, 0.000000, 0.000000, 0.000000, 0.000000, 9.000000, 14.800000, 20.500000, 25.000000, 26.800000, 25.900000, 23.300000, 20.000000, 16.800000, 13.800000, 11.300000, 9.200000, 7.600000, 6.300000, 5.200000, 4.300000, 0.000000],
                                [1.750000, 0.000000, 0.000000, 0.000000, 0.000000, 17.600000, 28.900000, 39.900000, 48.300000, 51.600000, 49.700000, 44.700000, 38.400000, 32.200000, 26.500000, 21.700000, 17.800000, 14.600000, 12.100000, 10.000000, 8.400000, 0.000000],
                                [2.250000, 0.000000, 0.000000, 0.000000, 0.000000, 29.000000, 47.500000, 65.400000, 78.800000, 83.800000, 80.600000, 72.400000, 62.300000, 52.200000, 43.000000, 35.300000, 28.900000, 23.800000, 19.700000, 16.300000, 13.700000, 0.000000],
                                [2.750000, 0.000000, 0.000000, 0.000000, 0.000000, 43.200000, 70.700000, 97.000000, 116.300000, 123.100000, 118.100000, 106.100000, 91.300000, 76.500000, 63.200000, 51.900000, 42.500000, 35.000000, 28.900000, 24.100000, 20.100000, 0.000000],
                                [3.250000, 0.000000, 0.000000, 0.000000, 0.000000, 60.200000, 98.300000, 134.500000, 160.500000, 169.300000, 162.100000, 145.500000, 125.200000, 105.000000, 86.800000, 71.300000, 58.500000, 48.200000, 39.900000, 33.200000, 27.800000, 0.000000],
                                [3.750000, 0.000000, 0.000000, 0.000000, 0.000000, 79.900000, 130.400000, 177.800000, 211.200000, 222.000000, 212.200000, 190.400000, 164.000000, 137.600000, 113.800000, 93.600000, 76.900000, 63.300000, 52.500000, 43.700000, 36.600000, 0.000000],
                                [4.250000, 0.000000, 0.000000, 0.000000, 0.000000, 102.400000, 166.700000, 226.700000, 268.300000, 281.100000, 268.200000, 240.500000, 207.200000, 174.100000, 144.100000, 118.500000, 97.400000, 80.300000, 66.600000, 55.500000, 46.500000, 0.000000],
                                [4.750000, 0.000000, 0.000000, 0.000000, 0.000000, 127.600000, 207.400000, 281.200000, 286.000000, 286.000000, 286.000000, 286.000000, 255.000000, 214.300000, 177.500000, 146.100000, 120.200000, 99.200000, 82.200000, 68.600000, 57.600000, 0.000000],
                                [5.250000, 0.000000, 0.000000, 0.000000, 0.000000, 155.400000, 252.400000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 258.200000, 214.000000, 176.300000, 145.100000, 119.800000, 99.400000, 83.000000, 69.700000, 0.000000],
                                [5.750000, 0.000000, 0.000000, 0.000000, 0.000000, 186.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 253.600000, 209.000000, 172.200000, 142.200000, 118.100000, 98.600000, 82.800000, 0.000000],
                                [6.250000, 0.000000, 0.000000, 0.000000, 0.000000, 219.200000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 244.200000, 201.200000, 166.400000, 138.200000, 115.500000, 97.100000, 0.000000],
                                [6.750000, 0.000000, 0.000000, 0.000000, 0.000000, 255.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 281.900000, 232.400000, 192.200000, 159.700000, 133.500000, 112.300000, 0.000000],
                                [7.250000, 0.000000, 0.000000, 0.000000, 0.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 265.600000, 219.800000, 182.800000, 152.900000, 128.700000, 0.000000],
                                [7.750000, 0.000000, 0.000000, 0.000000, 0.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 249.000000, 207.200000, 173.400000, 146.000000, 0.000000],
                                [8.250000, 0.000000, 0.000000, 0.000000, 0.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 279.900000, 233.000000, 195.100000, 164.400000, 0.000000],
                                [8.750000, 0.000000, 0.000000, 0.000000, 0.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 260.200000, 218.000000, 183.800000, 0.000000],
                                [9.250000, 0.000000, 0.000000, 0.000000, 0.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 242.100000, 204.100000, 0.000000],
                                [9.750000, 0.000000, 0.000000, 0.000000, 0.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 286.000000, 267.400000, 225.600000, 0.000000]
                            ],
                    'cost_model_inputs' :{
                        'reference_model_num':3,
                        'water_depth': 100,
                        'distance_to_shore': 80,
                        'number_rows': 10,
                        'devices_per_row':10,
                        'device_spacing':600,
                        'row_spacing': 600,
                        'cable_system_overbuild': 20
                }},
                'battery': {
                    'system_capacity_kwh': batt_kw * 4,
                    'system_capacity_kw': 5000
                }}
def test_hybrid_wave_only(wavesite):
    wave_only = {'wave': technologies['wave']}
    hybrid_plant = HybridSimulation(wave_only,wavesite,interconnect_kw = interconnection_size_kw)
    hybrid_plant.simulate(25)
    aeps = hybrid_plant.annual_energies
    assert aeps.pv == 0
    assert aeps.wind == 0
    assert aeps.wave == approx(121325260, 1e3)
    assert aeps.hybrid == approx(121325260, 1e3)

def test_hybrid_wind_only(site):
    wind_only = {'wind': technologies['wind']}
    hybrid_plant = HybridSimulation(wind_only, site, interconnect_kw=interconnection_size_kw)
    hybrid_plant.layout.plot()
    hybrid_plant.ppa_price = (0.01, )
    hybrid_plant.simulate(25)
    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values

    assert aeps.pv == 0
    assert aeps.wind == approx(33615479, 1e3)
    assert aeps.hybrid == approx(33615479, 1e3)

    assert npvs.pv == 0
    assert npvs.wind == approx(-13692784, 1e3)
    assert npvs.hybrid == approx(-13692784, 1e3)


def test_hybrid_pv_only(site):
    solar_only = {'pv': technologies['pv']}
    hybrid_plant = HybridSimulation(solar_only, site, interconnect_kw=interconnection_size_kw)
    hybrid_plant.layout.plot()
    hybrid_plant.ppa_price = (0.01, )
    hybrid_plant.pv.dc_degradation = [0] * 25
    hybrid_plant.simulate()
    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values

    assert aeps.pv == approx(9884106.55, 1e-3)
    assert aeps.wind == 0
    assert aeps.hybrid == approx(9884106.55, 1e-3)

    assert npvs.pv == approx(-5121293, 1e3)
    assert npvs.wind == 0
    assert npvs.hybrid == approx(-5121293, 1e3)


def test_hybrid(site):
    """
    Performance from Wind is slightly different from wind-only case because the solar presence modified the wind layout
    """
    solar_wind_hybrid = {key: technologies[key] for key in ('pv', 'wind')}
    hybrid_plant = HybridSimulation(solar_wind_hybrid, site, interconnect_kw=interconnection_size_kw)
    hybrid_plant.layout.plot()
    hybrid_plant.ppa_price = (0.01, )
    hybrid_plant.pv.dc_degradation = [0] * 25
    hybrid_plant.simulate()
    # plt.show()
    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values

    assert aeps.pv == approx(8703525.94, 13)
    assert aeps.wind == approx(33615479.57, 1e3)
    assert aeps.hybrid == approx(41681662.63, 1e3)

    assert npvs.pv == approx(-5121293, 1e3)
    assert npvs.wind == approx(-13909363, 1e3)
    assert npvs.hybrid == approx(-19216589, 1e3)


def test_hybrid_with_storage_dispatch(site):
    hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_kw)
    hybrid_plant.ppa_price = (0.03, )
    hybrid_plant.pv.dc_degradation = [0] * 25
    hybrid_plant.simulate()
    aeps = hybrid_plant.annual_energies
    npvs = hybrid_plant.net_present_values
    taxes = hybrid_plant.federal_taxes
    apv = hybrid_plant.energy_purchases_values
    debt = hybrid_plant.debt_payment
    esv = hybrid_plant.energy_sales_values
    depr = hybrid_plant.federal_depreciation_totals
    insr = hybrid_plant.insurance_expenses
    om = hybrid_plant.om_total_expenses
    rev = hybrid_plant.total_revenues
    tc = hybrid_plant.tax_incentives

    assert aeps.pv == approx(9857158, rel=0.5)
    assert aeps.wind == approx(33345892, rel=0.5)
    assert aeps.battery == approx(-153102, rel=0.5)
    assert aeps.hybrid == approx(43071511, rel=0.5)

    assert npvs.pv == approx(-1299631, rel=5e-2)
    assert npvs.wind == approx(-4066481, rel=5e-2)
    assert npvs.battery == approx(-7063175, rel=5e-2)
    assert npvs.hybrid == approx(-12629236, rel=5e-2)

    assert taxes.pv[1] == approx(105870, rel=5e-2)
    assert taxes.wind[1] == approx(404415, rel=5e-2)
    assert taxes.battery[1] == approx(301523, rel=5e-2)
    assert taxes.hybrid[1] == approx(817962, rel=5e-2)

    assert apv.pv[1] == approx(0, rel=5e-2)
    assert apv.wind[1] == approx(0, rel=5e-2)
    assert apv.battery[1] == approx(158296, rel=5e-2)
    assert apv.hybrid[1] == approx(33518, rel=5e-2)

    assert debt.pv[1] == approx(0, rel=5e-2)
    assert debt.wind[1] == approx(0, rel=5e-2)
    assert debt.battery[1] == approx(0, rel=5e-2)
    assert debt.hybrid[1] == approx(0, rel=5e-2)

    assert esv.pv[1] == approx(295715, rel=5e-2)
    assert esv.wind[1] == approx(1000377, rel=5e-2)
    assert esv.battery[1] == approx(176007, rel=5e-2)
    assert esv.hybrid[1] == approx(1323544, rel=5e-2)

    assert depr.pv[1] == approx(762811, rel=5e-2)
    assert depr.wind[1] == approx(2651114, rel=5e-2)
    assert depr.battery[1] == approx(1486921, rel=5e-2)
    assert depr.hybrid[1] == approx(4900847, rel=5e-2)

    assert insr.pv[0] == approx(0, rel=5e-2)
    assert insr.wind[0] == approx(0, rel=5e-2)
    assert insr.battery[0] == approx(0, rel=5e-2)
    assert insr.hybrid[0] == approx(0, rel=5e-2)

    assert om.pv[1] == approx(74993, rel=5e-2)
    assert om.wind[1] == approx(420000, rel=5e-2)
    assert om.battery[1] == approx(75000, rel=5e-2)
    assert om.hybrid[1] == approx(569993, rel=5e-2)

    assert rev.pv[1] == approx(295715, rel=5e-2)
    assert rev.wind[1] == approx(1000377, rel=5e-2)
    assert rev.battery[1] == approx(176007, rel=5e-2)
    assert rev.hybrid[1] == approx(1323544, rel=5e-2)

    assert tc.pv[1] == approx(1123104, rel=5e-2)
    assert tc.wind[1] == approx(504569, rel=5e-2)
    assert tc.battery[1] == approx(0, rel=5e-2)
    assert tc.hybrid[1] == approx(1659156, rel=5e-2)


def test_hybrid_om_costs_error(site):
    hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_kw,
                                    dispatch_options={'battery_dispatch': 'one_cycle_heuristic'})
    hybrid_plant.ppa_price = (0.03, )
    hybrid_plant.pv.dc_degradation = [0] * 25
    hybrid_plant.battery._financial_model.SystemCosts.om_production = (1,)
    try:
        hybrid_plant.simulate()
    except ValueError as e:
        assert e


def test_hybrid_om_costs(site):
    hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_kw,
                                    dispatch_options={'battery_dispatch': 'one_cycle_heuristic'})
    hybrid_plant.ppa_price = (0.03, )
    hybrid_plant.pv.dc_degradation = [0] * 25

    # set all O&M costs to 0 to start
    hybrid_plant.wind.om_fixed = 0
    hybrid_plant.wind.om_capacity = 0
    hybrid_plant.wind.om_variable = 0
    hybrid_plant.pv.om_fixed = 0
    hybrid_plant.pv.om_capacity = 0
    hybrid_plant.pv.om_variable = 0
    hybrid_plant.battery.om_fixed = 0
    hybrid_plant.battery.om_capacity = 0
    hybrid_plant.battery.om_variable = 0

    # test variable costs
    hybrid_plant.wind.om_variable = 5
    hybrid_plant.pv.om_variable = 2
    hybrid_plant.battery.om_variable = 3
    hybrid_plant.simulate()
    var_om_costs = hybrid_plant.om_variable_expenses
    total_om_costs = hybrid_plant.om_total_expenses
    for i in range(len(var_om_costs.hybrid)):
        assert var_om_costs.pv[i] + var_om_costs.wind[i] + var_om_costs.battery[i] == approx(var_om_costs.hybrid[i], rel=1e-1)
        assert total_om_costs.pv[i] == approx(var_om_costs.pv[i])
        assert total_om_costs.wind[i] == approx(var_om_costs.wind[i])
        assert total_om_costs.battery[i] == approx(var_om_costs.battery[i])
        assert total_om_costs.hybrid[i] == approx(var_om_costs.hybrid[i])
    hybrid_plant.wind.om_variable = 0
    hybrid_plant.pv.om_variable = 0
    hybrid_plant.battery.om_variable = 0

    # test fixed costs
    hybrid_plant.wind.om_fixed = 5
    hybrid_plant.pv.om_fixed = 2
    hybrid_plant.battery.om_fixed = 3
    hybrid_plant.simulate()
    fixed_om_costs = hybrid_plant.om_fixed_expenses
    total_om_costs = hybrid_plant.om_total_expenses
    for i in range(len(fixed_om_costs.hybrid)):
        assert fixed_om_costs.pv[i] + fixed_om_costs.wind[i] + fixed_om_costs.battery[i] \
               == approx(fixed_om_costs.hybrid[i])
        assert total_om_costs.pv[i] == approx(fixed_om_costs.pv[i])
        assert total_om_costs.wind[i] == approx(fixed_om_costs.wind[i])
        assert total_om_costs.battery[i] == approx(fixed_om_costs.battery[i])
        assert total_om_costs.hybrid[i] == approx(fixed_om_costs.hybrid[i])
    hybrid_plant.wind.om_fixed = 0
    hybrid_plant.pv.om_fixed = 0
    hybrid_plant.battery.om_fixed = 0

    # test capacity costs
    hybrid_plant.wind.om_capacity = 5
    hybrid_plant.pv.om_capacity = 2
    hybrid_plant.battery.om_capacity = 3
    hybrid_plant.simulate()
    cap_om_costs = hybrid_plant.om_capacity_expenses
    total_om_costs = hybrid_plant.om_total_expenses
    for i in range(len(cap_om_costs.hybrid)):
        assert cap_om_costs.pv[i] + cap_om_costs.wind[i] + cap_om_costs.battery[i] \
               == approx(cap_om_costs.hybrid[i])
        assert total_om_costs.pv[i] == approx(cap_om_costs.pv[i])
        assert total_om_costs.wind[i] == approx(cap_om_costs.wind[i])
        assert total_om_costs.battery[i] == approx(cap_om_costs.battery[i])
        assert total_om_costs.hybrid[i] == approx(cap_om_costs.hybrid[i])
    hybrid_plant.wind.om_capacity = 0
    hybrid_plant.pv.om_capacity = 0
    hybrid_plant.battery.om_capacity = 0


def test_hybrid_tax_incentives(site):
    hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_kw,
                                    dispatch_options={'battery_dispatch': 'one_cycle_heuristic'})
    hybrid_plant.ppa_price = (0.03, )
    hybrid_plant.pv.dc_degradation = [0] * 25
    hybrid_plant.wind._financial_model.TaxCreditIncentives.ptc_fed_amount = (1,)
    hybrid_plant.pv._financial_model.TaxCreditIncentives.ptc_fed_amount = (2,)
    hybrid_plant.battery._financial_model.TaxCreditIncentives.ptc_fed_amount = (3,)
    hybrid_plant.wind._financial_model.TaxCreditIncentives.ptc_fed_escal = 0
    hybrid_plant.pv._financial_model.TaxCreditIncentives.ptc_fed_escal = 0
    hybrid_plant.battery._financial_model.TaxCreditIncentives.ptc_fed_escal = 0
    hybrid_plant.simulate()

    ptc_wind = hybrid_plant.wind._financial_model.value("cf_ptc_fed")[1]
    assert ptc_wind == approx(hybrid_plant.wind._financial_model.value("ptc_fed_amount")[0]*hybrid_plant.wind.annual_energy_kw, rel=1e-3)

    ptc_pv = hybrid_plant.pv._financial_model.value("cf_ptc_fed")[1]
    assert ptc_pv == approx(hybrid_plant.pv._financial_model.value("ptc_fed_amount")[0]*hybrid_plant.pv.annual_energy_kw, rel=1e-3)

    ptc_batt = hybrid_plant.battery._financial_model.value("cf_ptc_fed")[1]
    assert ptc_batt == approx(hybrid_plant.battery._financial_model.value("ptc_fed_amount")[0]
           * hybrid_plant.battery._financial_model.LCOS.batt_annual_discharge_energy[1], rel=1e-3)

    ptc_hybrid = hybrid_plant.grid._financial_model.value("cf_ptc_fed")[1]
    ptc_fed_amount = hybrid_plant.grid._financial_model.value("ptc_fed_amount")[0]
    assert ptc_fed_amount == approx(1.231561210964179, rel=1e-3)
    assert ptc_hybrid == approx(ptc_fed_amount * hybrid_plant.grid._financial_model.Outputs.cf_energy_net[1], rel=1e-3)
