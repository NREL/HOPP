"""
Author: Kaitlin Brunik
Created: 7/20/2023
Institution: National Renewable Energy Lab
Description: This file outputs capital and operational costs of salt cavern hydrogen storage.
It needs to be updated to with operational dynamics.
Costs are in 2018 USD

Sources:
    - [1] Papadias 2021: https://www.sciencedirect.com/science/article/pii/S0360319921030834?via%3Dihub
    - [2] Papadias 2021: Bulk Hydrogen as Function of Capacity.docx documentation at hopp/hydrogen/h2_storage
    - [3] HDSAM V4.0 Gaseous H2 Geologic Storage sheet
"""

import numpy as np
from greenheart.simulation.technologies.hydrogen.h2_transport.h2_compression import Compressor

class SaltCavernStorage():
    """
    - Costs are in 2018 USD
    """

    def __init__(self, input_dict):  
        """
        Initialize SaltCavernStorage.

        Args:
            input_dict (dict):
                - H2_storage_kg (float): total capacity of hydrogen storage [kg]
                - storage_duration_hrs (float): (optional if H2_storage_kg set) [hrs]
                - flow_rate_kg_hr (float): (optional if H2_storage_kg set) [kg/hr]
                - system_flow_rate (float): [kg/day]
                - model (str): ('papadias' or 'hdsam')
                - labor_rate (float): (optional, default: 37.40) [$2018/hr]
                - insurance (float): (optional, default: 1%) [decimal percent]
                - property_taxes (float): (optional, default: 1%) [decimal percent]
                - licensing_permits (float): (optional, default: 0.01%) [decimal percent]
        Returns:
            - salt_cavern_storage_capex_per_kg (float): the installed capital cost per kg h2 in 2018 [USD/kg]
            - installed_capex (float): the installed capital cost in 2018 [USD] (including compressor)
            - storage_compressor_capex (float): the installed capital cost in 2018 for the compressor [USD]
            - total_opex (float): the OPEX (annual, fixed) in 2018 excluding electricity costs [USD/kg-yr]
            - output_dict (dict):
                - salt_cavern_storage_capex (float): installed capital cost in 2018 [USD]
                - salt_cavern_storage_opex (float): OPEX (annual, fixed) in 2018  [USD/yr]
        """         
        self.input_dict = input_dict
        self.output_dict = {}

        #inputs
        if 'H2_storage_kg' in input_dict:
            self.H2_storage_kg = input_dict['H2_storage_kg']        #[kg]
        elif 'storage_duration_hrs' and 'flow_rate_kg_hr' in input_dict:
            self.H2_storage_kg = input_dict['storage_duration_hrs'] * input_dict['flow_rate_kg_hr']  
        else:
            raise Exception('input_dict must contain H2_storage_kg or storage_duration_hrs and flow_rate_kg_hr')

        if 'system_flow_rate' not in input_dict.keys():
                raise ValueError("system_flow_rate required for salt cavern storage model.")
        else:
            self.system_flow_rate = input_dict['system_flow_rate']

        if 'model' in input_dict:
            self.model = input_dict['model']    #[papadias, hdsam]
        else:
            raise Exception('input_dict must contain model type of either `papadias` or `hdsam`')

        self.labor_rate = input_dict.get('labor_rate', 37.39817) # $(2018)/hr
        self.insurance = input_dict.get('insurance', 1/100) # % of total capital investment
        self.property_taxes = input_dict.get('property_taxes', 1/100) # % of total capital investment
        self.licensing_permits = input_dict.get('licensing_permits',0.1/100) # % of total capital investment
        self.comp_om = input_dict.get('compressor_om',4/100)    # % of compressor capital investment
        self.facility_om = input_dict.get('facility_om', 1/100) # % of facility capital investment minus compressor capital investment

    def salt_cavern_capex(self):
        """
        Calculates the installed capital cost of salt cavern hydrogen storage
        Returns:
            - salt_cavern_capex_per_kg (float): the installed capital cost per kg h2 in 2018 [USD/kg]
            - installed_capex (float): the installed capital cost in 2018 [USD] (including compressor)
            - storage_compressor_capex (float): the installed capital cost in 2018 for the compressor [USD]
            - output_dict (dict):
                - salt_cavern_capex (float): installed capital cost in 2018 [USD]
        """

        if self.model == 'papadias':
            # Installed capital cost
            a = 0.092548
            b = 1.6432
            c = 10.161
            self.salt_cavern_storage_capex_per_kg = np.exp(a*(np.log(self.H2_storage_kg/1000))**2 - b*np.log(self.H2_storage_kg/1000) + c)  # 2019 [USD] from Papadias [2]
            self.installed_capex = self.salt_cavern_storage_capex_per_kg * self.H2_storage_kg
            cepci_overall = 1.29/1.30 # Convert from $2019 to $2018
            self.installed_capex = cepci_overall * self.installed_capex
            self.output_dict['salt_cavern_storage_capex'] = self.installed_capex

            outlet_pressure = 120 # Max outlet pressure of salt cavern in [1]
            n_compressors = 2
            storage_compressor = Compressor(outlet_pressure,self.system_flow_rate,n_compressors=n_compressors)
            storage_compressor.compressor_power()
            motor_rating, power = storage_compressor.compressor_system_power()
            if motor_rating > 1600:
                n_compressors += 1
                storage_compressor = Compressor(outlet_pressure,self.system_flow_rate,n_compressors=n_compressors)
                storage_compressor.compressor_power()
                motor_rating, power = storage_compressor.compressor_system_power()
            comp_capex,comp_OM = storage_compressor.compressor_costs()
            cepci = 1.36/1.29 # convert from $2016 to $2018
            self.comp_capex = comp_capex*cepci
        elif self.model == 'hdsam':
            raise NotImplementedError
        return self.salt_cavern_storage_capex_per_kg, self.installed_capex, self.comp_capex

    def salt_cavern_opex(self):
        """
        Calculates the operation and maintenance costs excluding electricity costs for the salt cavern hydrogen storage
        - Returns:
            - total_opex (float): the OPEX (annual, fixed) in 2018 excluding electricity costs [USD/kg-yr]
            - output_dict (dict):
                - salt_cavern_storage_opex (float): OPEX (annual, fixed) in 2018  [USD/yr]
        """
        # Operations and Maintenace costs [3]
        # Labor 
        # Base case is 1 operator, 24 hours a day, 7 days a week for a 100,000 kg/day average capacity facility.  Scaling factor of 0.25 is used for other sized facilities
        annual_hours = 8760 * (self.system_flow_rate/100000)**0.25
        self.overhead = 0.5 
        labor = (annual_hours*self.labor_rate) * (1+self.overhead) # Burdened labor cost
        insurance = self.insurance * self.installed_capex
        property_taxes = self.property_taxes * self.installed_capex
        licensing_permits = self.licensing_permits * self.installed_capex
        comp_op_maint = self.comp_om * self.comp_capex 
        facility_op_maint = self.facility_om * (self.installed_capex - self.comp_capex)

        # O&M excludes electricity requirements
        total_om = labor+insurance+licensing_permits+property_taxes+comp_op_maint+facility_op_maint
        self.output_dict['salt_cavern_storage_opex'] = total_om
        return total_om