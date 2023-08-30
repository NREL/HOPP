import pytest
from hopp.simulation.technologies.steel.example_steel_run_script import h2_main_steel
'''
Values for tests were determined from [1]: Bhaskar, Abhinav, Rockey Abhishek, Mohsen Assadi, and Homan Nikpey Somehesaraei. 2022. "Decarbonizing primary steel production : Techno-economic assessment of a hydrogen based green steel production plant in Norway." Journal of Cleaner Production 350: 131339. doi: https://doi.org/10.1016/j.jclepro.2022.131339.
And hand calcs to verify the results
'''
def test_lcos():
    
    lcos = h2_main_steel()[0]

    assert pytest.approx(lcos,.1) == 613

def test_lcos_2():

    lcoe = .60 #cents/kwh
    steel_output_desired = 240320 #kg/hr
    mw_h2 = .5 #mil usd/mw
    elec_specification = 50 #kwh/kgh2
    lifetime = 20

    lcos = h2_main_steel(lcoe,steel_output_desired,MW_h2=mw_h2,elec_spec=elec_specification,lifetime=lifetime)[0]

    assert pytest.approx(lcos,.01) == 616.85


def test_total_operating_cost():

    operating_cost_yr = h2_main_steel()[3]

    assert pytest.approx(operating_cost_yr) == 522431219

def test_total_operating_cost_2():

    lcoe = .60 #cents/kwh
    steel_output_desired = 240320 #kg/hr
    mw_h2 = .5 #mil usd/mw
    elec_specification = 50 #kwh/kgh2
    
    operating_cost_yr = h2_main_steel(lcoe,steel_output_desired,MW_h2=mw_h2,elec_spec=elec_specification)[3]

    assert pytest.approx(operating_cost_yr,.1) == 1036847503

def test_total_electricity_cost():

    electricity_cost_yr = h2_main_steel()[1]

    assert pytest.approx(electricity_cost_yr) == 251928316

def test_total_electricity_cost_2():

    lcoe = .60 #cents/kwh
    steel_output_desired = 240320 #kg/hr
    mw_h2 = .5 #mil usd/mw
    elec_specification = 50 #kwh/kgh2

    electricity_cost_yr = h2_main_steel(lcoe,steel_output_desired,MW_h2=mw_h2,elec_spec=elec_specification)[1]

    assert pytest.approx(electricity_cost_yr,.1) == 495841707

def test_total_capital_cost():

    total_capital_cost = h2_main_steel()[2]

    assert pytest.approx(total_capital_cost,.1) == 974104842

def test_total_capital_cost_2():

    lcoe = .60 #cents/kwh
    steel_output_desired = 240320 #tls/hr
    mw_h2 = .5 #mil usd/mw
    elec_specification = 50 #kwh/kgh2

    total_capital_cost = h2_main_steel(lcoe,steel_output_desired,MW_h2=mw_h2,elec_spec=elec_specification)[2]

    assert pytest.approx(total_capital_cost) == 1843501768
