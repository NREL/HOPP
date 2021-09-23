import pytest
import pandas as pd

# TODO: update with pySSC
#import PySAM_DAOTk.TcsmoltenSalt as Tower
#import PySAM_DAOTk.TroughPhysical as Trough

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.tower_source import TowerPlant
from hybrid.trough_source import TroughPlant

@pytest.fixture
def site():
    return SiteInfo(flatirons_site)

def test_default_tower_model(site):
    """Testing PySAM tower model using heuristic dispatch method """
    tower_config = {'cycle_capacity_kw': 115 * 1000,
                    'solar_multiple': 2.4,
                    'tes_hours': 10.0}

    #filename = "C:/Users/WHamilt2/Documents/GitHub/HOPP/resource_files/solar/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv"

    tower_model = Tower.default('MSPTSingleOwner')
    # tower_model.value('solar_resource_file', filename)
    # tower_model.execute()
    # annual_energy = tower_model.value('annual_energy')  # 570167485.2074118
    # print("CSP Tower annual energy (Daggett, CA): " + str(annual_energy))

    # Getting default values
    tower_config = {'cycle_capacity_kw': tower_model.value('P_ref') * 1000.,
                    'solar_multiple': tower_model.value('solarm'),
                    'tes_hours': tower_model.value('tshours')}

    model = TowerPlant(site, tower_config)
    print(str(model.value('disp_pc_onoff_perm', 0.0)))
    print(str(model.value('disp_pc_onoff_perm')))
    sr_data = model.value('solar_resource_data')
    model.simulate(1)
    print("CSP Tower annual energy (TowerPlant): " + str(model.value('annual_energy')))  # 512658870.449323

    model._system_model.value('q_dot_pc_target_su') # daotk specific output

    tower_model = Tower.default('MSPTSingleOwner')
    tower_model.value('solar_resource_data', sr_data)
    tower_model.execute()
    annual_energy = tower_model.value('annual_energy')
    print("CSP Tower annual energy (direct): " + str(annual_energy))  # 512658870.449323

    assert annual_energy > 0.0  # make sure model generates useful energy
    assert model.value('annual_energy') == pytest.approx(annual_energy, 1e-5)


def test_default_trough_model(site):
    """Testing PySAM trough model using heuristic dispatch method """
    trough_config = {'cycle_capacity_kw': 110 * 1000,
                    'solar_multiple': 2.0,
                    'tes_hours': 6.0}

    # Getting default values
    trough_model = Trough.default('PhysicalTroughSingleOwner')
    trough_config = {'cycle_capacity_kw': trough_model.value('P_ref') * 1000.,
                    'solar_multiple': trough_model.value('specified_solar_multiple'),
                    'tes_hours': trough_model.value('tshours')}

    model = TroughPlant(site, trough_config)
    sr_data = model._system_model.value('solar_resource_data')
    #filename = model.value('file_name')        # This doesn't exist in trough
    model.simulate(1)
    print("CSP Trough annual energy (TroughPlant): " + str(model.value('annual_energy')))  # 333950296.71266896


    trough_model = Trough.default('PhysicalTroughSingleOwner')
    trough_model.value('solar_resource_data', sr_data)
    #trough_model.value('file_name', sr_data)
    trough_model.execute()
    annual_energy = trough_model.value('annual_energy')
    print("CSP Trough annual energy (direct): " + str(annual_energy))   # 333950296.71266896

    assert annual_energy > 0.0
    assert model.value('annual_energy') == pytest.approx(annual_energy, 1e-5)

