import math
import numpy as np
import sys
import pandas as pd

class Compressor():
    def __init__(self, input_dict, output_dict):
        self.input_dict = input_dict
        self.output_dict = output_dict

        # inputs
        self.flow_rate_kg_hr = input_dict['flow_rate_kg_hr']
        self.P_outlet = input_dict['P_outlet']
        self.compressor_rating_kWe = input_dict['compressor_rating_kWe']
        self.mean_time_between_failure = input_dict['mean_time_between_failure']
        self.total_hydrogen_throughput = input_dict['total_hydrogen_throughput']
        self.comp_efficiency = 0.50
        self.num_compressors = 2

    def compressor_power(self):
        """ Compression from 20 bar to 350 bar (pressure vessel storage)
            or compression from 20 bar to 100 bar (underground pipe storage)
            https://www.energy.gov/sites/default/files/2014/03/f9/nexant_h2a.pdf
        TODO: Add CoolProp to be able to calculate all power for different compressions"""
        Z = 1.03198     # mean compressibility factor
        R = 4.1240      # [kJ/kg K] hydrogen gas constant
        k = 1.667       # ratio of specific heats
        T = 25+273.15   # [C] suction and interstage gas temperature
        P_inlet = 20    # [bar] from electrolyzer

        if self.P_outlet == 350 or self.P_outlet == 100:    #[bar]
            comp_energy_per_kg = Z * R * T * (1/self.comp_efficiency) * (k/(k-1)) * ((self.P_outlet/P_inlet)**((k-1)/k)-1) / 3600     # [kWh/kg]
            compressor_power = self.flow_rate_kg_hr * comp_energy_per_kg #[kW]
        else:
            print("Error. P_outlet must be 100 or 350 bar.")
        self.output_dict['comp_energy_per_kg'] = comp_energy_per_kg
        self.output_dict['compressor_power'] = compressor_power
        return comp_energy_per_kg, compressor_power

    def compressor_capex(self):
        """Minimum 2 compressors required due to unreliability"""
        F_install = 1.2     # installation factor (<250 kg/hr)
        F_install_250 = 2.0     # installation factor (>250 kg/hr)
        F_indir = 1.27      # direct and indirect capital cost factor 

        C_cap = 19207*self.compressor_rating_kWe**(0.6089) # purchased equipment capital cost 

        if self.flow_rate_kg_hr < 250:
            compressor_capex = C_cap * F_install * F_indir * self.num_compressors #[USD]
        else:
            compressor_capex = C_cap * F_install_250 * F_indir * self.num_compressors #[USD]
        self.output_dict['compressor_capex'] = compressor_capex
        return compressor_capex

    def compressor_opex(self):
        """"mean_time_between_failure [days]: max 365
            total_hydrogen_throughput: annual amount of hydrogen compressed [kg/yr]"""
        if self.mean_time_between_failure <= 50:       #[days]
            maintenance_cost = 0.71     #[USD/kg H2]
            compressor_opex = maintenance_cost * self.total_hydrogen_throughput  #[USD/yr]
        elif 50 < self.mean_time_between_failure <= 100:
            maintenance_cost = 0.71 + ((self.mean_time_between_failure - 50)*((0.36 - 0.71)/(100-50)))     #[USD/kg H2]
            compressor_opex = maintenance_cost * self.total_hydrogen_throughput  #[USD/yr]
        elif 100 < self.mean_time_between_failure <= 200:
            maintenance_cost = 0.36 + ((self.mean_time_between_failure - 100)*((0.19 - 0.36)/(200-100)))     #[USD/kg H2]
            compressor_opex = maintenance_cost * self.total_hydrogen_throughput  #[USD/yr]
        elif 200 < self.mean_time_between_failure <= 365:
            maintenance_cost = 0.11 + ((self.mean_time_between_failure - 200)*((0.11 - 0.19)/(365-200)))     #[USD/kg H2]
            compressor_opex = maintenance_cost * self.total_hydrogen_throughput  #[USD/yr]
        else:
            print("Error. mean_time_between_failure <= 365 days.")
        self.output_dict['compressor_opex'] = compressor_opex
        return compressor_opex


if __name__ =="__main__":

    in_dict = dict()
    in_dict['flow_rate_kg_hr'] = 500
    in_dict['P_outlet'] = 350
    in_dict['compressor_rating_kWe'] = 802
    in_dict['mean_time_between_failure'] = 200
    in_dict['total_hydrogen_throughput'] = 5000000
    out_dict = dict()

    test = Compressor(in_dict, out_dict)
    test.compressor_power()
    test.compressor_capex()
    test.compressor_opex()
    print("compressor_power (kW): ", out_dict['compressor_power'])
    print("Compressor capex [USD]: ", out_dict['compressor_capex'])
    print("Compressor opex [USD/yr]: ", out_dict['compressor_opex'])