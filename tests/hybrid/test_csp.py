import pytest
import math
import PySAM_DAOTk.TcsmoltenSalt as TowerMoltenSalt
import PySAM_DAOTk.TroughPhysical as TroughPhysical

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.tower_source import TowerPlant
from hybrid.trough_source import TroughPlant

@pytest.fixture
def site():
    return SiteInfo(flatirons_site)


def test_default_tower_model(site):
    """Testing PySAM tower model using heuristic dispatch method """
    tower_config = {'cycle_capacity_kw': 50 * 1000,
                    'solar_multiple': 2.4,
                    'tes_hours': 8.0}

    model = TowerPlant(site, tower_config)

    model.simulate(1)


def test_default_trough_model(site):
    """Testing PySAM trough model using heuristic dispatch method """
    tower_config = {'cycle_capacity_kw': 50 * 1000,
                    'solar_multiple': 2.4,
                    'tes_hours': 8.0}

    model = TroughPlant(site, tower_config)

    model.simulate(1)

