
class Compressor():

    def compressor_power(flow_rate_kg_hr, P_outlet, comp_efficiency = .50):
        """ Compression from 20 bar to 350 bar (pressure vessel storage)
            or compression from 20 bar to 100 bar (underground pipe storage)
            https://www.energy.gov/sites/default/files/2014/03/f9/nexant_h2a.pdf
        TODO: Add CoolProp to be able to calculate all power for different compressions"""
        
        Z = 1.03198     # mean compressibility factor
        R = 8.3144      # [kJ/kg-mole K] universal gas constant
        n = 2           # number of stages in compressor
        k = 1.41        # ratio of specific heats
        T = 37.8        # [C] suction and interstage gas temperature
        P_inlet = 20    # [bar]

        if P_outlet == 350:    #[bar]
            compressor_power = Z * R * T * (1/comp_efficiency) * (k/(k -1)) * ((P_outlet/P_inlet)**((k-1)/(n*k))-1) / 3600 #[kWh/kg]
        elif P_outlet == 100:  #[bar]
            compressor_power = Z * R * T * (1/comp_efficiency) * (k/(k -1)) * ((P_outlet/P_inlet)**((k-1)/(n*k))-1) / 3600 #[kWh/kg]
        else:
            print("Error. P_outlet must be 100 or 350 bar.")
        return compressor_power

    def compressor_capex(compressor_rating_kWe, flow_rate_kg_hr):
        F_install = 1.2     # installation factor (<250 kg/hr)
        F_install_250 = 2.0     # installation factor (>250 kg/hr)
        F_indir = 1.27      # direct and indirect capital cost factor 

        C_cap = 19207*compressor_rating_kWe**(0.6089) # purchased equipment capital cost 

        if flow_rate_kg_hr < 250:
            compressor_capex = C_cap * F_install * F_indir  #[USD]
        else:
            compressor_capex = C_cap * F_install_250 * F_indir  #[USD]
        return compressor_capex



# compressor_rating_kWe = 802 #kWe
