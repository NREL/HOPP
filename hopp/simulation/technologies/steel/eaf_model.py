def establish_save_output_dict():
        """
        Establishes and returns a 'save_outputs_dict' dict
        for saving the relevant analysis variables for each site.
        """
        save_outputs_dict = dict()

        
        save_outputs_dict['Steel Output Actual (kg)'] = list()
        save_outputs_dict['Carbon Mass Needed (kg)'] = list()
        save_outputs_dict['Lime Mass Needed (kg)'] = list()
        save_outputs_dict['Energy needed for Electric Arc Furnance (kWh)'] = list()
        save_outputs_dict['Total Indirect Emissions (Ton CO2)'] = list()
        save_outputs_dict['Total Direct Emissions (Ton CO2)'] = list()
        save_outputs_dict['Total Emissions (Ton CO2)'] = list()
        save_outputs_dict['Shaft Total Capital Cost (Mil USD)'] = list()
        save_outputs_dict['Shaft Operational Cost (Mil USD per year)'] = list()
        save_outputs_dict['Shaft Maintenance Cost (Mil USD per year)'] = list()
        save_outputs_dict['Shaft Depreciation Cost (Mil USD per year)'] = list()
        save_outputs_dict['Total Carbon Cost (Mil USD per year)'] = list()
        save_outputs_dict['Total Labor Cost (Mil USD per ton)'] = list()
        save_outputs_dict['Total Emission Cost (Mil USD per year)'] = list()
        save_outputs_dict['Total Lime Cost (Mil USD per year)'] = list()


        return save_outputs_dict


class eaf_model():
    '''
    Author: Charles Kiefer
    Date: 8/14/23
    
    This class holds functions relating to the modeling of a Electronic Arc Furnace
    in relation to the production of Green Steel.  Pure iron, carbon and slime enter furnace,
    steel and CO2 leave the system.  The EAF acts a energy transfer method to allow for the energy
    needed to embed carbon in iron

    Sources will be in each function

    Args:
    steel_output_desired (kg) or (kg/hr): (float) resulting desired steel output
    steel_prod_yr (ton/yr): (float) plant capacity steel produced per year. For financial outputs

    returns:
    save_outputs_dict saves returns

    Steel Output Desired (kg):
        ['Steel Output Actual (kg)']:
            -Iron ore that wasn't reduced in the HDRI will reduce and become steel
        
        ['Carbon Mass Needed (kg)']
        ['Lime Mass Needed (kg)']:
            -Lime is a slag former and binds with impurites in the iron to produce more pure steel
        
        ['Carbon Mass Not Absorbed (kg)']:
            -Not all the carbon in the EAF will become steel.  Excess Carbon will become CO2
        
        ['Energy needed for Electric Arc Furnance (kWh)']
        ['Total Indirect Emissions (Ton CO2)']:
            -Indirect emissions come from the grid emissions like coal plants and transportation 
        
        ['Total Direct Emissions (Ton CO2)']
        ['Total Emissions (Ton CO2)']
        ['Shaft Total Capital Cost (Mil USD)']
        ['Shaft Operational Cost (Mil USD per year)']
        ['Shaft Maintenance Cost (Mil USD per year)']
        ['Shaft Depreciation Cost (Mil USD per year)']
        ['Total Carbon Cost (Mil USD per year)']
        ['Total Labor Cost (Mil USD per ton)']
        ['Total Emission Cost (Mil USD per year)']
        ['Total Lime Cost (Mil USD per year)']
    '''
    def __init__(self):
        '''
        Initializes and stores needed data for functions relating to Electric Arc Furnace
        for Green Steel

        Sources:
        [1]: Bhaskar, Abhinav, Rockey Abhishek, Mohsen Assadi, and Homan Nikpey Somehesaraei. 2022. "Decarbonizing primary steel production : Techno-economic assessment of a hydrogen based green steel production plant in Norway." Journal of Cleaner Production 350: 131339. doi: https://doi.org/10.1016/j.jclepro.2022.131339. 
        [2]: Chase, M.W., Jr. 1998. "NIST-JANAF Themochemical Tables, Fourth Edition." J. Phys. Chem. Ref. Data, Monograph 9 1-1951. doi:https://doi.org/10.18434/T4D303.

        '''
        self.eta_el = .6 #(%) electrical efficiency [1]

        self.stream_temp_in = 973 #(k) Temperature of metallic stream input eaf [1]
        self.eaf_temp = 1923 #(k) Temperature metallic stream out. Temp needed to achieve [1]

        self.mass_carbon_tls = 10 #(kg/tls) [1]
        self.mass_lime_tls = 50 #(kg/tls) [1]

        self.el_eaf = None #(kwh)

        self.eaf_co2 = 0.050  #ton/tls [1]
        self.cao_emission = 0.056 #tco2/tls [1]
        self.co2_eaf_electrode = 0.0070 #tco2/tls [1]
        self.pellet_production = 0.12 #tco2/tls [1]

        self.steel_out_actual = None #(kg)

        self.hfe_melting = 247 #(kj/kg) Energy needed to melt iron into a liquid [1] 

        self.indirect_emissions_total = None
        self.direct_emissions_total = None
        self.total_emissions = None

        self.lang_factor = 3 #(no units) Capital cost multiplier [1]
        self.plant_life = 40 #(years) [1]
        self.labor_cost_tls = 20 #(USD/ton/year) [1] $40 is flat rate for hdri and eaf together
        self.eaf_total_capital_cost = None #(Million USD)
        self.eaf_cost_per_ton_yr = 140 #(USD/tls/yr)
        self.eaf_op_cost_tls = 32 #(USD/tls/yr of eaf) [1]
        self.eaf_operational_cost_yr = None #(Million USD)
        self.maintenance_cost_percent = .015 #(% of capital cost)[1]
        self.eaf_maintenance_cost_yr = None #(Million USD)
        self.depreciation_cost = None #(Million USD)
        self.total_labor_cost_yr = None #(Million USD)
        self.emission_cost = 30 #(USD/ton CO2) [1]
        self.emission_factor = .413 #(ton CO2/kwh) [1]
        self.carbon_cost_tls = 200 #(USD/ton coal) [1]
        self.coal_total_cost_yr = None #(Mil USD/year)
        self.lime_cost = 112 #(USD/ton lime) [1]
        self.lime_cost_total = None #(USD/year)
        
    
    def mass_model(self, steel_out_desired):
        '''
        Mass model calculates the masses inputted and outputted of 
        an EAF system for output of tonne liquid steel
        
        Args:
        steel_output_desired (kg) or (kg/hr): (float) resulting desired steel output

        Sources:
        Model derived from: Bhaskar, Abhinav, Rockey Abhishek, Mohsen Assadi, and Homan Nikpey Somehesaraei. 2022. "Decarbonizing primary steel production : Techno-economic assessment of a hydrogen based green steel production plant in Norway." Journal of Cleaner Production 350: 131339. doi: https://doi.org/10.1016/j.jclepro.2022.131339.
        '''
        from hopp.simulation.technologies.steel.hdri_model import hdri_model

        model_instance = hdri_model()
        hdri_mass_model_outputs = model_instance.mass_model(steel_out_desired)
        
        m3 = steel_out_desired #kg

        m2_feo = hdri_mass_model_outputs[2]
        
        m6 = self.mass_carbon_tls #kg/tls
        m7 = self.mass_lime_tls #kg/tls

        self.carbon_total = m6*steel_out_desired/1000 #(kg)
        self.lime_total = m7*steel_out_desired/1000 #(kg)

        m2_feo_reduced = (m2_feo*.7)  #(kg) assumed 70% feo in EAF gets reduced       
        m3_actual = m2_feo_reduced + m3 #(kg) Actual steel outputted from excess iron oxidation
        
        self.steel_out_actual = m3_actual #(kg)

        save_outputs_dict = establish_save_output_dict()

        save_outputs_dict['Steel Output Actual (kg)'].append(self.steel_out_actual)
        save_outputs_dict['Carbon Mass Needed (kg)'].append(self.carbon_total)
        save_outputs_dict['Lime Mass Needed (kg)'].append(self.lime_total)

        return (save_outputs_dict, self.steel_out_actual, self.carbon_total, self.lime_total)
    
    def energy_model(self, steel_out_desired):
        '''
        This function calculates the energy balance of the hdri EAF. Negative values designate
        heat leaving the system. Positive values designate heat needs to enter the system.  

        Energy belance values should be positive.

        Args:
        steel_output_desired (kg) or (kg/hr): (float) resulting desired steel output

        Sources:
        Model derived from: Bhaskar, Abhinav, Rockey Abhishek, Mohsen Assadi, and Homan Nikpey Somehesaraei. 2022. "Decarbonizing primary steel production : Techno-economic assessment of a hydrogen based green steel production plant in Norway." Journal of Cleaner Production 350: 131339. doi: https://doi.org/10.1016/j.jclepro.2022.131339.

        '''
        from hopp.simulation.technologies.steel.hdri_model import hdri_model
        from hopp.simulation.technologies.steel.enthalpy_functions import fe_enthalpy, fe_enthalpy

        model_instance = hdri_model()

        hdri_mass_model_outputs = model_instance.mass_model(steel_out_desired)

        m2_fe = hdri_mass_model_outputs[1]
     
        
        hfe_T2 = fe_enthalpy(self.stream_temp_in)    #Enthalpy of DRI entering Eaf (kj/g)
        hfe_T3 = fe_enthalpy(self.eaf_temp)    #Enthalpy steel exiting EAF (kj/g)
        h3 = ((hfe_T3 - hfe_T2)*m2_fe*1000) + (m2_fe*self.hfe_melting)  #Total Enthalpy at output Kj

        h3_kwh = h3 / 3600 #(kwh)

        self.el_eaf = h3_kwh/self.eta_el #kwh
    

        save_outputs_dict = establish_save_output_dict()

        save_outputs_dict['Energy needed for Electric Arc Furnance (kWh)'].append(self.el_eaf)
        
        return (save_outputs_dict,self.el_eaf)
    
    
    
    def emission_model(self, steel_out_desired):
        '''
        This function models the emissions from the EAF.  This incorporates direct and indirect emissions

        Args:
        steel_output_desired (kg) or (kg/hr): (float) resulting desired steel output

        Sources:
        Model derived from: Bhaskar, Abhinav, Rockey Abhishek, Mohsen Assadi, and Homan Nikpey Somehesaraei. 2022. "Decarbonizing primary steel production : Techno-economic assessment of a hydrogen based green steel production plant in Norway." Journal of Cleaner Production 350: 131339. doi: https://doi.org/10.1016/j.jclepro.2022.131339.

        '''
        from hopp.simulation.technologies.steel.hdri_model import hdri_model

        eaf_model.energy_model(self,steel_out_desired)
        model_instance = hdri_model()

        el_heater = model_instance.heater_mass_energy_model(steel_out_desired)[1]

        indirect_emissions = ((self.el_eaf + el_heater)*self.emission_factor) #tco2 or tco2/hr

        direct_emissions = (self.eaf_co2 + self.cao_emission + self.co2_eaf_electrode 
                            + self.pellet_production) #tco2/tls or tco2/hr/tls
        
        indirect_emissions_total = indirect_emissions #tco2 or tco2/hr
        direct_emissions_total = direct_emissions*steel_out_desired/1000 #tco2 tco2/hr


        self.indirect_emissions_total = indirect_emissions_total
        self.direct_emissions_total = direct_emissions_total
        self.total_emissions = self.indirect_emissions_total + self.direct_emissions_total

        save_outputs_dict = establish_save_output_dict()

        save_outputs_dict['Total Indirect Emissions (Ton CO2)'].append(self.indirect_emissions_total)
        save_outputs_dict['Total Direct Emissions (Ton CO2)'].append(self.direct_emissions_total)
        save_outputs_dict['Total Emissions (Ton CO2)'].append(self.total_emissions)

        return (save_outputs_dict,self.indirect_emissions_total, self.direct_emissions_total, self.total_emissions)
        
        
        
        
    def financial_model(self, steel_prod_yr):
        '''
        This function returns the financials for a rated capacity plant for EAF

        Args:
        steel_prod_yr (ton/yr): (float) plant capacity steel produced per year. For financial outputs

        Sources:
        Model derived from: Bhaskar, Abhinav, Rockey Abhishek, Mohsen Assadi, and Homan Nikpey Somehesaraei. 2022. "Decarbonizing primary steel production : Techno-economic assessment of a hydrogen based green steel production plant in Norway." Journal of Cleaner Production 350: 131339. doi: https://doi.org/10.1016/j.jclepro.2022.131339.

        '''

        self.eaf_total_capital_cost = ((self.eaf_cost_per_ton_yr*steel_prod_yr)/10**6)*self.lang_factor #Mil USD

        self.eaf_operational_cost_yr = self.eaf_op_cost_tls*steel_prod_yr/10**6 #Mil USD per year

        self.eaf_maintenance_cost_yr = self.maintenance_cost_percent*self.eaf_total_capital_cost #Mil USD per year

        self.depreciation_cost = self.eaf_total_capital_cost/self.plant_life #Mil USD per year

        total_coal = self.mass_carbon_tls * steel_prod_yr/1000 #tonne coal
        total_lime = self.mass_lime_tls * steel_prod_yr/1000 # tonne lime

        self.coal_total_cost_yr = (self.carbon_cost_tls * total_coal)/10**6 #(Mill USD per year)

        self.lime_cost_total = (self.lime_cost * total_lime)/10**6

        self.total_labor_cost_yr = self.labor_cost_tls*steel_prod_yr/10**6

        direct_emissions_tls = (self.eaf_co2 + self.cao_emission + self.co2_eaf_electrode 
                            + self.pellet_production) #tco2/tls
        
        direct_emissions_yr = direct_emissions_tls * steel_prod_yr/10**6

        self.total_emission_cost = direct_emissions_yr * self.emission_cost

        save_outputs_dict = establish_save_output_dict()

        save_outputs_dict['Shaft Total Capital Cost (Mil USD)'].append(self.eaf_total_capital_cost)
        save_outputs_dict['Shaft Operational Cost (Mil USD per year)'].append(self.eaf_operational_cost_yr)
        save_outputs_dict['Shaft Maintenance Cost (Mil USD per year)'].append(self.eaf_maintenance_cost_yr)
        save_outputs_dict['Shaft Depreciation Cost (Mil USD per year)'].append(self.depreciation_cost)
        save_outputs_dict['Total Carbon Cost (Mil USD per year)'].append(self.coal_total_cost_yr)
        save_outputs_dict['Total Labor Cost (Mil USD per ton)'].append(self.total_labor_cost_yr)
        save_outputs_dict['Total Lime Cost (Mil USD per year)'].append(self.lime_cost_total)
        save_outputs_dict['Total Emission Cost (Mil USD per year)'].append(self.total_emission_cost)

        
        return (save_outputs_dict, self.eaf_total_capital_cost, self.eaf_operational_cost_yr, self.eaf_maintenance_cost_yr,
                self.depreciation_cost, self.coal_total_cost_yr, self.total_labor_cost_yr, self.lime_cost_total,
                self.total_emission_cost)



if __name__ == '__main__':
    model_instance = eaf_model()

    steel_output_desired = 1000 #(kg or kg/hr)

    mass_outputs = model_instance.mass_model(steel_output_desired)
    energy_outputs = model_instance.energy_model(steel_output_desired)
    emission_outputs = model_instance.emission_model(steel_output_desired)

    steel_output_desired_yr = 2000000 #(ton/yr)

    financial_outputs = model_instance.financial_model(steel_output_desired_yr)

    

