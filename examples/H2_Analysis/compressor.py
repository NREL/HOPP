import math
import numpy as np
import sys
import pandas as pd

class Compressor():


    def compressor_power(flow_rate_kg_hr, P_outlet, comp_efficiency = .50):
        """ Compression from 20 bar to 350 bar (pressure vessel storage)
            or compression from 20 bar to 100 bar (underground pipe storage)
            https://www.energy.gov/sites/default/files/2014/03/f9/nexant_h2a.pdf
        TODO: Add CoolProp to be able to calculate all power for different compressions"""
        Z = 1.03198     # mean compressibility factor
        R = 4.1240      # [kJ/kg K] hydrogen gas constant
        k = 1.667       # ratio of specific heats
        T = 25+273.15   # [C] suction and interstage gas temperature
        P_inlet = 20    # [bar]

        if P_outlet == 350:    #[bar]
            comp_energy_per_kg = Z * R * T * (1/comp_efficiency) * (k/(k-1)) * ((P_outlet/P_inlet)**((k-1)/k)-1) / 3600     # [kWh/kg]
            compressor_power = flow_rate_kg_hr * Z * R * T * (1/comp_efficiency) * (k/(k-1)) * ((P_outlet/P_inlet)**((k-1)/k)-1) / 3600 #[kW]
        elif P_outlet == 100:  #[bar]
            comp_energy_per_kg = Z * R * T * (1/comp_efficiency) * (k/(k-1)) * ((P_outlet/P_inlet)**((k-1)/k)-1) / 3600     # [kWh/kg]
            compressor_power = flow_rate_kg_hr * Z * R * T * (1/comp_efficiency) * (k/(k-1)) * ((P_outlet/P_inlet)**((k-1)/k)-1) / 3600 #[kWh]
        else:
            print("Error. P_outlet must be 100 or 350 bar.")
        return comp_energy_per_kg, compressor_power

    def compressor_capex(compressor_rating_kWe, flow_rate_kg_hr, num_compressors = 2):
        """Minimum 2 compressors required due to unreliability"""
        F_install = 1.2     # installation factor (<250 kg/hr)
        F_install_250 = 2.0     # installation factor (>250 kg/hr)
        F_indir = 1.27      # direct and indirect capital cost factor 

        C_cap = 19207*compressor_rating_kWe**(0.6089) # purchased equipment capital cost 

        if flow_rate_kg_hr < 250:
            compressor_capex = C_cap * F_install * F_indir * num_compressors #[USD]
        else:
            compressor_capex = C_cap * F_install_250 * F_indir * num_compressors #[USD]
        return compressor_capex

    def compressor_opex(mean_time_between_failure,total_hydrogen_throughput):
        """"mean_time_between_failure [days]: max 365
            total_hydrogen_throughput: annual amount of hydrogen compressed [kg/yr]"""
        if mean_time_between_failure <= 50:       #[days]
            maintenance_cost = 0.71     #[USD/kg H2]
            compressor_opex = maintenance_cost * total_hydrogen_throughput  #[USD/yr]
        elif 50 < mean_time_between_failure <= 100:
            maintenance_cost = 0.71 + ((mean_time_between_failure - 50)*((0.36 - 0.71)/(100-50)))     #[USD/kg H2]
            compressor_opex = maintenance_cost * total_hydrogen_throughput  #[USD/yr]
        elif 100 < mean_time_between_failure <= 200:
            maintenance_cost = 0.36 + ((mean_time_between_failure - 100)*((0.19 - 0.36)/(200-100)))     #[USD/kg H2]
            compressor_opex = maintenance_cost * total_hydrogen_throughput  #[USD/yr]
        elif 200 < mean_time_between_failure <= 365:
            maintenance_cost = 0.11 + ((mean_time_between_failure - 200)*((0.11 - 0.19)/(365-200)))     #[USD/kg H2]
            compressor_opex = maintenance_cost * total_hydrogen_throughput  #[USD/yr]
        else:
            print("Error. mean_time_between_failure <= 365 days.")
            return compressor_opex
# compressor_rating_kWe = 802 #kWe
