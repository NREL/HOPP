from greenheart.to_organize.H2_Analysis.simple_cash_annuals import simple_cash_annuals

class Compressor():
    def __init__(self, input_dict, output_dict):
        self.input_dict = input_dict
        self.output_dict = output_dict

        # inputs
        self.flow_rate_kg_hr = input_dict['flow_rate_kg_hr']                        #[kg/hr] # per compressor
        self.P_outlet = input_dict['P_outlet']                                      #[bar]
        self.compressor_rating_kWe = input_dict['compressor_rating_kWe']            #[kWe] # per compressor
        self.mean_time_between_failure = input_dict['mean_time_between_failure']    #[days]
        self.total_hydrogen_throughput = input_dict['total_hydrogen_throughput']    #[kg-H2/yr]

        try:
            number_of_compressors = input_dict["number_of_compressors"]
        except:
            print("Assuming 3 compressors")
            number_of_compressors = 3
        try:
            plant_life = input_dict["plant_life"]
        except:
            print("Assuming 30 year plant life")
            plant_life = 30
        try:
            useful_life = input_dict["useful_life"]
        except:
            print("Assuming 15 year useful life")
            useful_life = 15

        # assumptions
        self.comp_efficiency = 0.50
        self.num_compressors = number_of_compressors # was 3
        self.plant_life = plant_life # [years], was 30
        self.useful_life = useful_life # [years], was 15

    def compressor_power(self):
        """ Compression from 20 bar to 250 bar (pressure vessel storage)
            or compression from 20 bar to 100 bar (underground pipe storage)
            https://www.energy.gov/sites/default/files/2014/03/f9/nexant_h2a.pdf
        TODO: Add CoolProp to be able to calculate all power for different compressions"""
        Z = 1.03198     # mean compressibility factor
        R = 4.1240      # [kJ/kg K] hydrogen gas constant
        k = 1.667       # ratio of specific heats
        T = 25+273.15   # [C] suction and interstage gas temperature
        P_inlet = 20    # [bar] from electrolyzer

        if self.P_outlet == 250 or self.P_outlet == 100:    #[bar]
            comp_energy_per_kg = Z * R * T * (1/self.comp_efficiency) * (k/(k-1)) * ((self.P_outlet/P_inlet)**((k-1)/k)-1) / 3600     # [kWh/kg -per compressor]
            compressor_power = self.num_compressors * self.flow_rate_kg_hr * comp_energy_per_kg #[kW] - cumulative across all compressors
        else:
            raise(ValueError("Error. P_outlet must be 100 or 250 bar."))
        self.output_dict['comp_energy_per_kg'] = comp_energy_per_kg
        self.output_dict['compressor_power'] = compressor_power
        return comp_energy_per_kg, compressor_power

    def compressor_costs(self):
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

        #Compressor opex
        insurance = 0.01    #percent of total capital investment
        property_taxes = 0.015  #percent of total capital investment
        license_permits = 0.01  #percent of total capital investment
        op_and_maint = 0.04     #percent of total installed capital
        labor = 4.2977 * self.flow_rate_kg_hr**0.2551
        overhead = 0.5 * labor

        cost_factors = insurance + property_taxes + license_permits
        compressor_opex = (cost_factors * C_cap) + (op_and_maint * compressor_capex) + labor + overhead

        # """"TODO: Add mean_time_between_failure [days]: max 365
        #     total_hydrogen_throughput: annual amount of hydrogen compressed [kg/yr]
        #     This report gives station costs as a function of MTBF but not broken down to single compressor level
        #     https://www.nrel.gov/docs/fy14osti/58564.pdf"""
        # if self.mean_time_between_failure <= 50:       #[days]
        #     maintenance_cost = 0.71     #[USD/kg H2]
        #     compressor_opex = self.num_compressors * maintenance_cost * self.total_hydrogen_throughput  #[USD/yr]
        # elif 50 < self.mean_time_between_failure <= 100:
        #     maintenance_cost = 0.71 + ((self.mean_time_between_failure - 50)*((0.36 - 0.71)/(100-50)))     #[USD/kg H2]
        #     compressor_opex = self.num_compressors * maintenance_cost * self.total_hydrogen_throughput  #[USD/yr]
        # elif 100 < self.mean_time_between_failure <= 200:
        #     maintenance_cost = 0.36 + ((self.mean_time_between_failure - 100)*((0.19 - 0.36)/(200-100)))     #[USD/kg H2]
        #     compressor_opex = self.num_compressors * maintenance_cost * self.total_hydrogen_throughput  #[USD/yr]
        # elif 200 < self.mean_time_between_failure <= 365:
        #     maintenance_cost = 0.11 + ((self.mean_time_between_failure - 200)*((0.11 - 0.19)/(365-200)))     #[USD/kg H2]
        #     compressor_opex = self.num_compressors * maintenance_cost * self.total_hydrogen_throughput  #[USD/yr]
        # else:
        #     print("Error. mean_time_between_failure <= 365 days.")
        self.output_dict['compressor_opex'] = compressor_opex

        """Assumed useful life = payment period for capital expenditure.
           compressor amortization interest = 3%"""

        compressor_annuals = simple_cash_annuals(self.plant_life, self.useful_life,\
            self.output_dict['compressor_capex'],self.output_dict['compressor_opex'], 0.03)

        self.output_dict['compressor_annuals'] = compressor_annuals
        return compressor_capex, compressor_opex, compressor_annuals

if __name__ =="__main__":

    in_dict = dict()
    in_dict['flow_rate_kg_hr'] = 126
    in_dict['P_outlet'] = 250
    in_dict['compressor_rating_kWe'] = 802
    in_dict['mean_time_between_failure'] = 200
    in_dict['total_hydrogen_throughput'] = 5000000
    out_dict = dict()

    test = Compressor(in_dict, out_dict)
    test.compressor_power()
    test.compressor_costs()
    print("compressor_power (kW): ", out_dict['compressor_power'])
    print("Compressor capex [USD]: ", out_dict['compressor_capex'])
    print("Compressor opex [USD/yr]: ", out_dict['compressor_opex'])
    print(out_dict['compressor_annuals'])