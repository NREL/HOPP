from greenheart.simulation.technologies.hydrogen.electrolysis.PEM_costs_Singlitico_model import PEMCostsSingliticoModel
from pytest import approx
import numpy as np


TOL = 1e-3

BASELINE = np.array(
    [
        # onshore, [capex, opex]
        [
            [50.7105172052493, 1.2418205567631722],
        ],
        # offshore, [capex, opex]
        [
            [67.44498788298158, 2.16690312809502],
        ],
    ]
)

class TestPEMCostsSingliticoModel():

    def test_calc_capex(self):
        P_elec = 0.1 # [GW]
        RC_elec = 700 # [MUSD/GW]

        # test onshore capex
        pem_onshore = PEMCostsSingliticoModel(elec_location=0)
        capex_onshore = pem_onshore.calc_capex(P_elec, RC_elec)

        assert capex_onshore == approx(BASELINE[0][0][0], TOL)

        # test offshore capex
        pem_offshore = PEMCostsSingliticoModel(elec_location=1)
        capex_offshore = pem_offshore.calc_capex(P_elec, RC_elec)

        assert capex_offshore == approx(BASELINE[1][0][0], TOL)

    def test_calc_opex(self):
        P_elec = 0.1 # [GW]
        capex_onshore = BASELINE[0][0][0]
        capex_offshore = BASELINE[1][0][0]

        # test onshore opex
        pem_onshore = PEMCostsSingliticoModel(elec_location=0)
        opex_onshore = pem_onshore.calc_opex(P_elec, capex_onshore)

        assert opex_onshore == approx(BASELINE[0][0][1], TOL)

        # test offshore opex
        pem_offshore = PEMCostsSingliticoModel(elec_location=1)
        opex_offshore = pem_offshore.calc_opex(P_elec, capex_offshore)

        assert opex_offshore == approx(BASELINE[1][0][1], TOL)

    def test_run(self):
        P_elec = 0.1 # [GW]
        RC_elec = 700 # [MUSD/GW]

        # test onshore opex
        pem_onshore = PEMCostsSingliticoModel(elec_location=0)
        capex_onshore, opex_onshore = pem_onshore.run(P_elec, RC_elec)

        assert capex_onshore == approx(BASELINE[0][0][0], TOL)
        assert opex_onshore == approx(BASELINE[0][0][1], TOL)

        # test offshore opex
        pem_offshore = PEMCostsSingliticoModel(elec_location=1)
        capex_offshore, opex_offshore = pem_offshore.run(P_elec, RC_elec)

        assert capex_offshore == approx(BASELINE[1][0][0], TOL)
        assert opex_offshore == approx(BASELINE[1][0][1], TOL)


if __name__ == "__main__":
    test_set = TestPEMCostsSingliticoModel()
