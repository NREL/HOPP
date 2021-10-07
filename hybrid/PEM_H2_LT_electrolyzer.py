## Low-Temperature PEM Electrolyzer Model
"""
Python model of H2 PEM low-temp electrolyzer.

Quick Hydrogen Physics:

1 kg H2 <-> 11.1 N-m3 <-> 33.3 kWh (LHV) <-> 39.4 kWh (HHV)

High mass energy density (1 kg H2= 3,77 l gasoline)
Low volumetric density (1 Nm³ H2= 0,34 l gasoline

Hydrogen production from water electrolysis (~5 kWh/Nm³ H2)

Power:1 MW electrolyser <-> 200 Nm³/h  H2 <-> ±18 kg/h H2
Energy:+/-55 kWh of electricity --> 1 kg H2 <-> 11.1 Nm³ <-> ±10 liters
demineralized water

Power production from a hydrogen PEM fuel cell from hydrogen (+/-50%
efficiency):
Energy: 1 kg H2 --> 16 kWh
"""
import math
import numpy as np
import sys
import pandas as pd
from matplotlib import pyplot as plt

np.set_printoptions(threshold=sys.maxsize)


class PEM_electrolyzer_LT:
    """
    Create an instance of a low-temperature PEM Electrolyzer System. Each
    stack in the electrolyzer system in this model is rated at 1 MW_DC.

    Parameters
    _____________
    np_array P_input_external_kW
        1-D array of time-series external power supply

    string voltage_type
        Nature of voltage supplied to electrolyzer from the external power
        supply ['variable' or 'constant]

    float power_supply_rating_MW
        Rated power of external power supply

    Returns
    _____________

    """

    def __init__(self, input_dict, output_dict):
        self.input_dict = input_dict
        self.output_dict = output_dict

        # array of input power signal
        self.input_dict['P_input_external_kW'] = input_dict['P_input_external_kW']
        self.electrolyzer_system_size_MW = input_dict['electrolyzer_system_size_MW']

        # self.input_dict['voltage_type'] = 'variable'  # not yet implemented
        self.input_dict['voltage_type'] = 'constant'
        self.stack_input_voltage_DC = 250

        # Assumptions:
        self.min_V_cell = 1.62  # Only used in variable voltage scenario
        self.p_s_h2_bar = 31  # H2 outlet pressure
        self.stack_input_current_lower_bound = 500
        self.stack_rating_kW = 1000  # 1 MW
        self.cell_active_area = 1250
        self.N_cells = 130

        # Constants:
        self.moles_per_g_h2 = 0.49606
        self.V_TN = 1.48  # Thermo-neutral Voltage (Volts)
        self.F = 96485  # Faraday's Constant (C/mol)
        self.R = 8.314  # Ideal Gas Constant (J/mol/K)

        self.external_power_supply()

    def external_power_supply(self):
        """
        External power source (grid or REG) which will need to be stepped
        down and converted to DC power for the electrolyzer.

        Please note, for a wind farm as the electrolyzer's power source,
        the model assumes variable power supplied to the stack at fixed
        voltage (fixed voltage, variable power and current)

        TODO: extend model to accept variable voltage, current, and power
        This will replicate direct DC-coupled PV system operating at MPP
        """
        power_converter_efficiency = 0.95
        if self.input_dict['voltage_type'] == 'constant':

            self.input_dict['P_input_external_kW'] = \
                np.where(self.input_dict['P_input_external_kW'] >
                         (self.electrolyzer_system_size_MW * 1000),
                         (self.electrolyzer_system_size_MW * 1000),
                         self.input_dict['P_input_external_kW'])

            self.output_dict['curtailed_P_kW'] = \
                np.where(self.input_dict['P_input_external_kW'] >
                         (self.electrolyzer_system_size_MW * 1000),
                         (self.input_dict['P_input_external_kW'] -
                          (self.electrolyzer_system_size_MW * 1000)), 0)

            self.output_dict['current_input_external_Amps'] = \
                (self.input_dict['P_input_external_kW'] * 1000 *
                 power_converter_efficiency) / (self.stack_input_voltage_DC *
                                                self.system_design())

            self.output_dict['stack_current_density_A_cm2'] = \
                self.output_dict['current_input_external_Amps'] / self.cell_active_area

            self.output_dict['current_input_external_Amps'] = \
                np.where(self.output_dict['current_input_external_Amps'] <
                         self.stack_input_current_lower_bound, 0,
                         self.output_dict['current_input_external_Amps'])

        else:
            pass  # TODO: extend model to variable voltage and current source

    def system_design(self):
        """
        For now, system design is solely a function of max. external power
        supply; i.e., a rated power supply of 50 MW means that the electrolyzer
        system developed by this model is also rated at 50 MW

        TODO: Extend model to include this capability.
        Assume that a PEM electrolyzer behaves as a purely resistive load
        in a circuit, and design the configuration of the entire electrolyzer
        system - which may consist of multiple stacks connected together in
        series, parallel, or a combination of both.
        """
        h2_production_multiplier = (self.electrolyzer_system_size_MW * 1000) / \
                                   self.stack_rating_kW
        self.output_dict['electrolyzer_system_size_MW'] = self.electrolyzer_system_size_MW
        return h2_production_multiplier

    def cell_design(self):
        """
        Creates an I-V (polarization) curve of each cell in a stack.

        Please note that this method is currently not used in the model. It
        will be used once the electrolyzer model is expanded to variable
        voltage supply as well as implementation of the self.system_design()
        method

        Motivation:

        The most common representation of the electrolyzer performance is the
        polarization curve that represents the relation between the current density
        and the voltage (V):
        Source: https://www.sciencedirect.com/science/article/pii/S0959652620312312

        V = N_c(E_cell + V_Act,c + V_Act,a + iR_cell)

        where N_c is the number of electrolyzer cells,E_cell is the open circuit
        voltage VAct,and V_Act,c are the anode and cathode activation over-potentials,
        i is the current density and iRcell is the electrolyzer cell resistance
        (ohmic losses).

        Use this to make a V vs. A (Amperes/cm2) graph which starts at 1.23V because
        thermodynamic reaction of water formation/splitting dictates that standard
        electrode potential has a ∆G of 237 kJ/mol (where: ∆H = ∆G + T∆S)
        """

        # Cell level inputs:
        N_cells = 130
        electrode_surface_area_cm2 = self.cell_active_area / N_cells
        cell_rating_watts = (self.stack_rating_kW * 1000) / N_cells

        # V_cell_max = 3.0    #Volts
        # V_cell_I_density_max = 2.50     #mA/cm2
        E_rev = 1.23  # (in Volts) Reversible potential at 25degC
        E_th = 1.48  # (in Volts) Thermoneutral potential at 25degC
        T_C = 80  # Celsius
        T_K = T_C + 273.15  # in Kelvins
        # E_cell == Open Circuit Voltage
        E_cell = 1.5184 - (1.5421 * (10 ** (-3)) * T_K) + \
                 (9.523 * (10 ** (-5)) * T_K * math.log(T_K)) + \
                 (9.84 * (10 ** (-8)) * (T_K ** 2))
        # V_act = V_act_c + V_Act_a
        R = 8.314  # Ideal Gas Constant (J/mol/K)
        i = self.output_dict['stack_current_density_A_cm2']
        F = 96485  # Faraday's Constant (C/mol)

        # Following coefficient values obtained from Yigit and Selamet (2016) -
        # https://www.sciencedirect.com/science/article/pii/S0360319916318341?via%3Dihub
        a_a = 2  # Anode charge transfer coefficient
        a_c = 0.5  # Cathode charge transfer coefficient
        i_o_a = 2 * (10 ** (-7))
        i_o_c = 2 * (10 ** (-3))
        V_act = (((R * T_K) / (a_a * F)) * np.arcsinh(i / (2 * i_o_a))) + (
                ((R * T_K) / (a_c * F)) * np.arcsinh(i / (2 * i_o_c)))
        lambda_water_content = ((-2.89556 + (0.016 * T_K)) + 1.625) / 0.1875
        delta = 0.0003  # membrane thickness (cm) - assuming a 3-µm thick membrane
        sigma = ((0.005139 * lambda_water_content) - 0.00326) * math.exp(
            1268 * ((1 / 303) - (1 / T_K)))  # Material thickness # material conductivity
        R_cell = (delta / sigma)
        V_cell = E_cell + V_act + (i * R_cell)
        V_cell = np.where(V_cell < E_rev, E_rev, V_cell)
        V_stack = N_cells * V_cell  # Stack operational voltage

    def dynamic_operation(self):
        """
        Model the electrolyzer's realistic response/operation under variable RE

        TODO: add this capability to the model
        """
        # When electrolyzer is already at or near its optimal operation
        # temperature (~80degC)
        warm_startup_time_secs = 30
        cold_startup_time_secs = 5 * 60  # 5 minutes

    def water_electrolysis_efficiency(self):
        """
        https://www.sciencedirect.com/science/article/pii/S2589299119300035#b0500

        According to the first law of thermodynamics energy is conserved.
        Thus, the conversion efficiency calculated from the yields of
        converted electrical energy into chemical energy. Typically,
        water electrolysis efficiency is calculated by the higher heating
        value (HHV) of hydrogen. Since the electrolysis process water is
        supplied to the cell in liquid phase efficiency can be calculated by:

        n_T = V_TN / V_cell

        where, V_TN is the thermo-neutral voltage (min. required V to
        electrolyze water)

        Parameters
        ______________

        Returns
        ______________

        """

        n_T = self.V_TN / (self.stack_input_voltage_DC / self.N_cells)
        return n_T

    def faradaic_efficiency(self):
        """`
        Text background from:
        [https://www.researchgate.net/publication/344260178_Faraday%27s_
        Efficiency_Modeling_of_a_Proton_Exchange_Membrane_Electrolyzer_
        Based_on_Experimental_Data]

        In electrolyzers, Faraday’s efficiency is a relevant parameter to
        assess the amount of hydrogen generated according to the input
        energy and energy efficiency. Faraday’s efficiency expresses the
        faradaic losses due to the gas crossover current. The thickness
        of the membrane and operating conditions (i.e., temperature, gas
        pressure) may affect the Faraday’s efficiency.

        Equation for n_F obtained from:
        https://www.sciencedirect.com/science/article/pii/S0360319917347237#bib27

        Parameters
        ______________
        float f_1
            Coefficient - value at operating temperature of 80degC (mA2/cm4)

        float f_2
            Coefficient - value at operating temp of 80 degC (unitless)

        np_array current_input_external_Amps
            1-D array of current supplied to electrolyzer stack from external
            power source


        Returns
        ______________

        float n_F
            Faradaic efficiency (unitless)

        """
        f_1 = 250  # Coefficient (mA2/cm4)
        f_2 = 0.996  # Coefficient (unitless)
        I_cell = self.output_dict['current_input_external_Amps'] * 1000

        # Faraday efficiency
        n_F = (((I_cell / self.cell_active_area) ** 2) /
               (f_1 + ((I_cell / self.cell_active_area) ** 2))) * f_2

        return n_F

    def compression_efficiency(self):
        """
        In industrial contexts, the remaining hydrogen should be stored at
        certain storage pressures that vary depending on the intended
        application. In the case of subsequent compression, pressure-volume
        work, Wc, must be performed. The additional pressure-volume work can
        be related to the heating value of storable hydrogen. Then, the total
        efficiency reduces by the following factor:
        https://www.mdpi.com/1996-1073/13/3/612/htm

        Due to reasons of material properties and operating costs, large
        amounts of gaseous hydrogen are usually not stored at pressures
        exceeding 100 bar in aboveground vessels and 200 bar in underground
        storages
        https://www.sciencedirect.com/science/article/pii/S0360319919310195

        Partial pressure of H2(g) calculated using:
        The hydrogen partial pressure is calculated as a difference between
        the  cathode  pressure, 101,325 Pa, and the water saturation
        pressure
        [Source: Energies2018,11,3273; doi:10.3390/en11123273]

        """
        n_limC = 0.825  # Limited efficiency of gas compressors (unitless)
        H_LHV = 241  # Lower heating value of H2 (kJ/mol)
        K = 1.4  # Average heat capacity ratio (unitless)
        C_c = 2.75  # Compression factor (ratio of pressure after and before compression)
        n_F = self.faradaic_efficiency()
        j = self.output_dict['stack_current_density_A_cm2']
        n_x = ((1 - n_F) * j) * self.cell_active_area
        n_h2 = j * self.cell_active_area
        Z = 1  # [Assumption] Average compressibility factor (unitless)
        T_in_C = 80  # Assuming electrolyzer operates at 80degC
        T_in = 273.15 + T_in_C  # (Kelvins) Assuming electrolyzer operates at 80degC
        W_1_C = (K / (K - 1)) * ((n_h2 - n_x) / self.F) * self.R * T_in * Z * \
                ((C_c ** ((K - 1) / K)) - 1)  # Single stage compression

        # Calculate partial pressure of H2 at the cathode:
        A = 8.07131
        B = 1730.63
        C = 233.426
        p_h2o_sat = 10 ** (A - (B / (C + T_in_C)))  # Pa
        p_cat = 101325  # Cathode pressure (Pa)
        p_h2_cat = p_cat - p_h2o_sat
        p_s_h2_Pa = self.p_s_h2_bar * 1e5

        s_C = math.log((p_s_h2_Pa / p_h2_cat), 10) / math.log(C_c, 10)
        W_C = round(s_C) * W_1_C  # Pressure-Volume work - energy reqd. for compression
        net_energy_carrier = n_h2 - n_x  # C/s
        net_energy_carrier = np.where((n_h2 - n_x) == 0, 1, net_energy_carrier)
        n_C = 1 - ((W_C / (((net_energy_carrier) / self.F) * H_LHV * 1000)) * (1 / n_limC))
        n_C = np.where((n_h2 - n_x) == 0, 0, n_C)
        return n_C

    def total_efficiency(self):
        """
        Aside from efficiencies accounted for in this model
        (water_electrolysis_efficiency, faradaic_efficiency, and
        compression_efficiency) all process steps such as gas drying above
        2 bar or water pumping can be assumed as negligible. Ultimately, the
        total efficiency or system efficiency of a PEM electrolysis system is:

        n_T = n_p_h2 * n_F_h2 * n_c_h2
        https://www.mdpi.com/1996-1073/13/3/612/htm
        """
        n_p_h2 = self.water_electrolysis_efficiency()
        n_F_h2 = self.faradaic_efficiency()
        n_c_h2 = self.compression_efficiency()

        n_T = n_p_h2 * n_F_h2 * n_c_h2
        self.output_dict['total_efficiency'] = n_T
        return n_T

    def h2_production_rate(self):
        """
        H2 production rate calculated using Faraday's Law of Electrolysis
        (https://www.sciencedirect.com/science/article/pii/S0360319917347237#bib27)

        Parameters
        _____________

        float f_1
            Coefficient - value at operating temperature of 80degC (mA2/cm4)

        float f_2
            Coefficient - value at operating temp of 80 degC (unitless)

        np_array
            1-D array of current supplied to electrolyzer stack from external
            power source


        Returns
        _____________

        """
        # Single stack calculations:
        n_Tot = self.total_efficiency()
        h2_production_rate = n_Tot * ((self.N_cells *
                                       self.output_dict['current_input_external_Amps']) /
                                      (2 * self.F))  # mol/s
        h2_production_rate_g_s = h2_production_rate / self.moles_per_g_h2
        h2_produced_kg_hr = h2_production_rate_g_s * 3.6

        self.output_dict['stack_h2_produced_kg_hr'] = h2_produced_kg_hr

        # Total electrolyzer system calculations:
        h2_produced_kg_hr_system = self.system_design() * h2_produced_kg_hr
        # h2_produced_kg_hr_system = h2_produced_kg_hr
        self.output_dict['h2_produced_kg_hr_system'] = h2_produced_kg_hr_system

        return h2_produced_kg_hr_system

    def degradation(self):
        """
        TODO
        Add a time component to the model - for degradation ->
        https://www.hydrogen.energy.gov/pdfs/progress17/ii_b_1_peters_2017.pdf
        """
        pass

    def water_supply(self):
        """
        Calculate water supply rate based system efficiency and H2 production
        rate
        TODO: Add this capability to the model
        """
        max_water_feed_mass_flow_rate_kg_hr = 411  # kg per hour
        pass

    def h2_storage(self):
        """
        Model to estimate Ideal Isorthermal H2 compression at 70degC
        https://www.sciencedirect.com/science/article/pii/S036031991733954X

        The amount of hydrogen gas stored under pressure can be estimated
        using the van der Waals equation

        p = [(nRT)/(V-nb)] - [a * ((n^2) / (V^2))]

        where p is pressure of the hydrogen gas (Pa), n the amount of
        substance (mol), T the temperature (K), and V the volume of storage
        (m3). The constants a and b are called the van der Waals coefficients,
        which for hydrogen are 2.45 × 10−2 Pa m6mol−2 and 26.61 × 10−6 ,
        respectively.
        """

        pass


if __name__=="__main__":
    # Example on how to use this model:
    in_dict = dict()
    in_dict['electrolyzer_system_size_MW'] = 15
    out_dict = dict()

    electricity_profile = pd.read_csv('sample_wind_electricity_profile.csv')
    in_dict['P_input_external_kW'] = electricity_profile.iloc[:, 1].to_numpy()

    el = PEM_electrolyzer_LT(in_dict, out_dict)
    el.h2_production_rate()
    print("Hourly H2 production by stack (kg/hr): ", out_dict['stack_h2_produced_kg_hr'][0:50])
    print("Hourly H2 production by system (kg/hr): ", out_dict['h2_produced_kg_hr_system'][0:50])
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('PEM H2 Electrolysis Results for ' +
                str(out_dict['electrolyzer_system_size_MW']) + ' MW System')

    axs[0, 0].plot(out_dict['stack_h2_produced_kg_hr'])
    axs[0, 0].set_title('Hourly H2 production by stack')
    axs[0, 0].set_ylabel('kg_h2 / hr')
    axs[0, 0].set_xlabel('Hour')

    axs[0, 1].plot(out_dict['h2_produced_kg_hr_system'])
    axs[0, 1].set_title('Hourly H2 production by system')
    axs[0, 1].set_ylabel('kg_h2 / hr')
    axs[0, 1].set_xlabel('Hour')

    axs[1, 0].plot(in_dict['P_input_external_kW'])
    axs[1, 0].set_title('Hourly Energy Supplied by Wind Farm (kWh)')
    axs[1, 0].set_ylabel('kWh')
    axs[1, 0].set_xlabel('Hour')

    total_efficiency = out_dict['total_efficiency']
    system_h2_eff = (1 / total_efficiency) * 33.3
    system_h2_eff = np.where(total_efficiency == 0, 0, system_h2_eff)

    axs[1, 1].plot(system_h2_eff)
    axs[1, 1].set_title('Total Stack Energy Usage per mass net H2')
    axs[1, 1].set_ylabel('kWh_e/kg_h2')
    axs[1, 1].set_xlabel('Hour')

    plt.show()
    print("Annual H2 production (kg): ", np.sum(out_dict['h2_produced_kg_hr_system']))
    print("Annual energy production (kWh): ", np.sum(in_dict['P_input_external_kW']))
    print("H2 generated (kg) per kWH of energy generated by wind farm: ",
          np.sum(out_dict['h2_produced_kg_hr_system']) / np.sum(in_dict['P_input_external_kW']))
