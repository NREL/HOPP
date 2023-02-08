"""
Author: Christopher Bay
Date: 01/24/2023
Institution: National Renewable Energy Laboratory
Description: This file implements electrolzyer CapEx and OpEx models from [1].
Sources:
    - [1] Singlitico, Alessandro, Jacob Østergaard, and Spyros Chatzivasileiadis. "Onshore, offshore or
        in-turbine electrolysis? Techno-economic overview of alternative integration designs for green hydrogen
        production into Offshore Wind Power Hubs." Renewable and Sustainable Energy Transition 1 (2021): 100005.
"""

import numpy as np
import matplotlib.pyplot as plt


class PEMCostsSingliticoModel():

    def __init__(
        self,
        elec_location: int,
    ):
        """
        Initialize object for PEM costs based on [1].

        Args:
            elec_location (int): Parameter for indicating the electrolyzer location;
                0 is for onshore, 1 is for offshore or in-turbine.
        """
        # Values for CapEX & OpEx taken from [1], Table B.2, PEMEL.
        # Installation costs include land, contingency, contractors, legal fees, construction,
        # engineering, yard improvements, buildings, electrics, piping, instrumentation,
        # and installation and grid connection.
        self.IF = 0.33 # instatllation fraction [% RC_elec]
        self.RP_elec = 10 # reference power [MW]

        # Values for OpEx taken from [1], Table B.3, PEMEL.
        self.RP_SR = 5 # reference power [MW]
        self.RU_SR = 0.41 # reference cost share [%], for a reference power, RP_SR, of 5MW
        self.P_stack_max_bar = 2 # average max size [MW]
        self.SF_SR_0 = 0.11 # average scale factor


        self.OS = elec_location # 1 for offshore or in-turbine electrolyzer location, 0 for onshore; from [1],
                                # Table B.1 notes for CapEx_el
        
        # NOTE: This is used in the stack replacement cost code that is currently commented out; more work needs
        # to be done to make sure this is set and used correctly.
        # self.P_elec_bar = 1 * 10**3 # scaled maximum [MW] from [1], Table B.1 notes for OpEx_elec_eq

        # NOTE: This is used in the stack replacement cost code that is currently commented out.
        # self.OH_max = 85000 # Lifetime maximum operating hours [h], taken from [1], Table 1, PEMEL

    def run(
        self,
        P_elec: float,
        RC_elec: float,
    ) -> tuple:
        """
        Computes the CapEx and OpEx costs for a single electrolyzer.

        Args:
            P_elec (float): Nominal capacity of the electrolyzer [GW].
            RC_elec (float): Reference cost of the electrolyzer [MUSD/GW].

        Returns:
            tuple: CapEx and OpEx costs for a single electrolyzer.
        """
        capex = self.calc_capex(P_elec, RC_elec)
        opex = self.calc_opex(P_elec, capex)

        return capex, opex

    def calc_capex(
        self,
        P_elec: float,
        RC_elec: float,
    ) -> float:
        """
        CapEx for a single electrolyzer, given the electrolyzer capacity and reference cost.
        Equation from [1], Table B.1, CapEx_EL. For in-turbine electrolyzers,
        it is assumed that the maximum electrolyzer size is equal to the turbine rated capacity.

        NOTE: If the single electrolyzer capacity exceeds 100MW, the CapEx becomes fixed at the cost of a
        100MW system, due to decreasing economies of scale (based on assumption from [1]).
        As such, if you use the output to calculate a cost per unit of electrolyzer, you will need to divide
        the cost by 100MW and not the user-specified size of the electrolyzer for sizes above 100 MW.

        Args:
            P_elec (float): Nominal capacity of the electrolyzer [GW].
            RC_elec (float): Reference cost of the electrolyzer [MUSD/GW].

        Returns:
            float: CapEx for electrolyzer [MUSD].
        """
        # Choose the scale factor based on electrolyzer size, [1], Table B.2.
        if P_elec < 10 / 10**3:
            self.SF_elec = -0.21 # scale factor, -0.21 for <10MW, -0.14 for >10MW
        else:
            self.SF_elec = -0.14 # scale factor, -0.21 for <10MW, -0.14 for >10MW
        
        # If electrolyzer capacity is >100MW, fix unit cost to 100MW electrolyzer as economies of scale
        # stop at sizes above this, according to assumption in [1].
        if P_elec > 100 / 10**3:
            P_elec = 0.1

        # Return the cost of a single electrolyzer of the specified capacity in millions of USD (or the supplied currency).
        # MUSD = GW   * MUSD/GW *           -             *      GW   * MW/GW /      MW       **      -
        return P_elec * RC_elec * (1 + self.IF * self.OS) *  ((P_elec * 10**3 / self.RP_elec) ** self.SF_elec)

    def calc_opex(
        self,
        P_elec: float,
        capex_elec: float,
        RC_elec: float = None,
        OH: float = None,
    ) -> float:
        """
        OpEx for a single electrolyzer, given the electrolyzer capacity and reference cost.
        Equations from [1], Table B.1, OpEx_elec_eq and OpEx_elec_neq.
        The returned OpEx cost include equipment and non-equipment costs, but excludes the stack replacement cost.

        NOTE: If the single electrolyzer capacity exceeds 100MW, the OpEx becomes fixed at the cost of a
        100MW system, due to decreasing economies of scale (based on assumption from [1]).
        As such, if you use the output to calculate a cost per unit of electrolyzer, you will need to divide
        the cost by 100MW and not the user-specified size of the electrolyzer for sizes above 100 MW.

        NOTE: Code for the stack replacement cost is included below, but does not currently match results
        from [1]. DO NOT USE in the current form.

        Args:
            P_elec (float): Nominal capacity of the electrolyzer [GW].
            capex_elec (float): CapEx for electrolyzer [MUSD].
            RC_elec (float, optional): Reference cost of the electrolyzer [MUSD/GW]. Defaults to None. Not currently used.
            OH (float, optional): Operating hours [h]. Defaults to None. Not currently used.

        Returns:
            float: OpEx for electrolyzer [MUSD].
        """
        # If electrolyzer capacity is >100MW, fix unit cost to 100MW electrolyzer as economies of scale
        # stop at sizes above this, according to assumption in [1].
        if P_elec > 100 / 10**3:
            P_elec = 0.1

        # Including material cost for planned and unplanned maintenance, labor cost in central Europe, which
        # all depend on a system scale. Excluding the cost of electricity and the stack replacement,
        # calculated separately. Scaled maximum to P_elec_bar = 1 GW.
        # MUSD*MW         MUSD    *              -                *    -   *    GW   * MW/GW
        opex_elec_eq = capex_elec * (1 - self.IF * (1 + self.OS)) * 0.0344 * (P_elec * 10**3) ** -0.155

        # Covers the other operational expenditure related to the facility level. This includes site
        # management, land rent and taxes, administrative fees (insurance, legal fees...), and site maintenance.
        # MUSD                    MUSD     
        opex_elec_neq = 0.04 * capex_elec * self.IF * (1 + self.OS)

        # NOTE: The stack replacement costs below  don't match the results in [1] supplementary materials.
        # ***DO NOT USE*** stack replacement cost in its current form.

        # Choose the scale factor based on electrolyzer size, [1], Table B.2.
        # if P_elec < 10 / 10**3:
        #     self.SF_elec = -0.21 # scale factor, -0.21 for <10MW, -0.14 for >10MW
        # else:
        #     self.SF_elec = -0.14 # scale factor, -0.21 for <10MW, -0.14 for >10MW

        # Approximation of stack costs and replacement cost depending on the electrolyzer equipment costs.
        # Paid only the year in which the replacement is needed.
        # MUSD/GW    %     * MUSD/GW *       -       *      MW     /      MW       **       -   
        # RC_SR = self.RU_SR * RC_elec * (1 - self.IF) * (self.RP_SR / self.RP_elec) ** self.SF_elec
        # # -                 -          *               MW        /         MW
        # SF_SR = 1 - (1 - self.SF_SR_0) * np.exp(-self.P_elec_bar / self.P_stack_max_bar)
        # # SF_SR = 1 - (1 - self.SF_SR_0) * np.exp(-P_elec * 10**3 / self.P_stack_max_bar)
        # # MUSD           GW   * MUSD/GW *  GW   * MW/GW      MW       **   -   *  h /      h    
        # opex_elec_sr = P_elec * RC_SR * (P_elec * 10**3 / self.RP_SR) ** SF_SR * OH / self.OH_max

        return opex_elec_eq + opex_elec_neq

if __name__ == "__main__":
    # setup up PEM for offshore/in-turbine costs
    pem = PEMCostsSingliticoModel(elec_location=1)

    RC_elec = 600 # reference cost of electrolyzer [MUSD/GW]
    P_elec = 0.000625 # electrolzyer capacity [GW]

    # calculate CapEx and OpEx per unit costs
    capex = pem.calc_capex(P_elec, RC_elec)
    opex = pem.calc_opex(P_elec, capex)

    print('capex [MUSD/GW]: ', capex / P_elec)
    print('opex [MUSD/GW]: ', opex / P_elec)

    # capex = []
    # capex_norm = []
    # opex = []
    # opex_norm = []
    # elec_nom_tot = np.arange(0.5, 12.1, 0.5)
    # # elec_nom = np.linspace(0.000625, 0.015, 24)
    # elec_nom = np.arange(0.5, 12.1, 0.5)
    # # elec_nom = np.arange(0.1, 0.5, 0.1)
    # for nom in elec_nom:
    #     capex.append(pem.calc_capex(nom, RC_elec))
    #     capex_norm.append(pem.calc_capex(nom, RC_elec) / nom)
    #     opex.append(pem.calc_opex(nom, capex[-1]))
    #     opex_norm.append(pem.calc_opex(nom, capex[-1]) / 0.1)

    # # opex_eq = np.array([val[0] for val in opex])
    # # opex_sr = np.array([val[1] for val in opex])
    # # opex_neq = np.array([val[2] for val in opex])

    # # base_capex = [1473.96293,
    # #     1279.788173,
    # #     1173.874669,
    # #     1085.613416,
    # #     1032.656664,
    # #     1050.308914,
    # #     997.3521624,
    # #     979.6999117,
    # #     944.3954104,
    # #     944.3954104,
    # #     926.7431598,
    # #     891.4386584,
    # #     873.7864078,
    # #     873.7864078,
    # #     838.4819064,
    # #     838.4819064,
    # #     838.4819064,
    # #     838.4819064,
    # #     838.4819064,
    # #     838.4819064,
    # #     785.5251545,
    # #     803.1774051,
    # #     803.1774051,
    # #     767.8729038,
    # # ]

    # # fig, ax = plt.subplots(1)
    # # plt.plot(elec_nom_tot, capex)
    # # plt.plot(elec_nom_tot, base_capex)
    # # # plt.plot(elec_nom, np.array(base_capex) * 4.3)
    # # # plt.plot(elec_nom, np.array(capex) * 3)
    # # plt.title('CapEx')
    # # # ax.set_aspect('equal')

    # fig, ax = plt.subplots(1)
    # plt.plot(elec_nom_tot, opex_norm)
    # plt.title('OpEx')
    # # ax.set_aspect('equal')
    # plt.show()

    # # fig, ax = plt.subplots(1)
    # # plt.plot(elec_nom_tot, opex_sr)
    # # plt.title('OpEx SR')
    # # # ax.set_aspect('equal')
    # # plt.show()

