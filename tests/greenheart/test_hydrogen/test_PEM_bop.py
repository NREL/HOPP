from pytest import approx, fixture
import numpy as np

from greenheart.simulation.technologies.hydrogen.electrolysis.PEM_BOP.PEM_BOP import (
    pem_bop,
)


@fixture
def bop_energy():
    power_profile_kw = np.array(
        [
            0,
            9999,  # just below turndown ratio
            10000,  # exactly at turndown ratio
            82746,
            93774,  # max power in csv. largest operating ratio
            100000,  # full power
        ]
    )

    electrolyzer_rating_mw = 100  # MW
    turndown_ratio = 0.1

    bop_energy = pem_bop(
        power_profile_to_electrolyzer_kw=power_profile_kw,
        electrolyzer_rated_mw=electrolyzer_rating_mw,
        electrolyzer_turn_down_ratio=turndown_ratio,
    )
    return bop_energy


def test_bop_energy(subtests, bop_energy):
    with subtests.test("No power"):
        assert bop_energy[0] == 0
    with subtests.test("below turndown"):
        assert bop_energy[1] == 0
    with subtests.test("at turndown"):
        assert bop_energy[2] == approx(11032.668, 1e-2)
    with subtests.test("mid-range power"):
        assert bop_energy[3] == approx(6866.87, 1e-2)
    with subtests.test("max power in curve"):
        assert bop_energy[4] == approx(7847.85)
    with subtests.test("full power"):
        assert bop_energy[5] == approx(7847.85)
