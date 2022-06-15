"""
Python Model of Hydrogen Pressure Vessel

PEM electrolyzer --> compressor --> pressure vessel storage

Hydrogen storage in underground Type-1, API 5L X52 pipes
TODO: limit max pressure to 104 bar.

TODO: Add different types of storage? Salt cavern, Underground lined-rock caverns, above ground storage.

Underground Type-1: 
- Pipe material is a major cost factor.
- CAPEX and OPEX decrease less than ~10% for 1-20 tonne H2
- More economical than geologic storage for <20-tonne H2

Salt Cavern:
- CAPEX and OPEX decrease with economy of scale

Lined-rock Cavern:
- More expensive than salt cavern
- CAPEX and OPEX decrease with economy of scale
"""
from audioop import add
from os import stat
import CoolProp.CoolProp as CP
import numpy as np
import math
import matplotlib.pyplot as plt

class PressureVessel():

    def __init__(self):

            # length of simulation
            self.Nt = 1
            
            # amount of H2 production by plant (kg/hr)
            self.H2_production = np.zeros(self.Nt)

            # amount of energy needed from the pressure vessel
            self.shortfall = np.zeros(self.Nt)

            # grid demand (kg/hr)
            self.grid_demand = np.zeros(self.Nt)

            # size of pressure vessel (kg)
            self.pressure_vessel_storage = 0

            # charge rate of the pressure vessel (kg/hr)
            self.charge_rate = 0
            self.discharge_rate = 0

    def dispatch(self):

         # storage module
        pressure_vessel_storage = self.pressure_vessel_storage  # kg
        charge_rate = self.charge_rate  # kg/hr
        discharge_rate = self.discharge_rate # kg/hr


        pressure_vessel_fill_level = np.zeros(self.Nt)
        pressure_vessel_used = np.zeros(self.Nt)
        excess_energy = np.zeros(self.Nt)

        for i in range(self.Nt):
            # should you charge
            if self.H2_production[i] > self.grid_demand[i]:
                if i == 0:
                    pressure_vessel_fill_level[i] = np.min(self.H2_production[i] - self.grid_demand[i],charge_rate)
                    amount_filled = pressure_vessel_fill_level[i]
                    excess_energy[i]=(self.H2_production[i] -self.grid_demand[i]) - amount_filled
                else:
                    if pressure_vessel_fill_level[i-1] < pressure_vessel_storage:
                        add_gen = np.min(self.H2_production[i] - self.grid_demand[i], charge_rate)
                        pressure_vessel_fill_level[i] = np.min([pressure_vessel_fill_level[i-1] + add_gen]\
                            ,pressure_vessel_storage)
                        amount_filled = pressure_vessel_fill_level[i] - pressure_vessel_fill_level[i-1]
                        excess_energy[i] = (self.H2_production[i] - self.grid_demand[i]) - amount_filled

            # should you discharge
            else:
                if i > 0:
                    if pressure_vessel_fill_level[i-1] > 0:

                        pressure_vessel_used[i] = np.min(self.grid_demand[i] - self.H2_production[i]\
                            ,pressure_vessel_fill_level[i-1], discharge_rate)
                        pressure_vessel_fill_level[i] = pressure_vessel_fill_level[i-1] - pressure_vessel_used[i]
                        
        return pressure_vessel_used, excess_energy, pressure_vessel_fill_level



    def static_pressure_vessel(H2_storage):
        """Nominal storage volume is 300 MWh (50 MW, 6 hours)
        https://www.nrel.gov/docs/fy10osti/48360.pdf
        https://www.energy.gov/sites/prod/files/2015/11/f27/fcto_fuel_cells_fact_sheet.pdf
        Assume 85% compressor efficiency
        Assume 60% Fuel cell conversion
        Assume 20 degrees celcius, isothermal compression
        Assume compressed to 100 bar
        H2 energy 16kWh (includes fuel cell conversion)
            300,000 kWh [storage] / 16kWh [H2]  = 
            18,750 kg H2 needed for nominal storage
        """
        
        if H2_storage:
            #Round trip storage (PEM -> Compressor -> Storage -> Fuel Cell): 18,750 kg H2
            storage_volume = 2404.78  #[m^3]
            compressor_work = 228.56  #[kWh]
            pressure_vessel_capex = 10500000    #[USD]
            pressure_vessel_opex = 1575000  #[USD/yr]
            print("H2 Storage volume: ",storage_volume, "[m^3]",\
                "Compressor Work: ", compressor_work, "[kWh]", \
                    "Pressure Vessel Capex: ", pressure_vessel_capex, "[USD]", \
                        "Pressure Vessel Opex: ", pressure_vessel_opex, "[USD/yr]")
        else:
            pass
        return storage_volume, compressor_work, pressure_vessel_capex, pressure_vessel_opex



    def pressure_vessel(amount_hydrogen,temperature, pressure):
        """Inputs: amount_hydrogen [kg], temperature [celcius], pressure [bar], 
        plant storage_capacity [kg]
        https://www.sciencedirect.com/science/article/pii/S0360319921005838#tbl2"""

        fluid = 'hydrogen'
        R = 8.314       # Ideal Gas Constant [J/mol/K]
        n = 0.00201588  # Molar mass of hydrogen [kg/mol]
        T_given = temperature + 273.15     #[K]
        P_given = pressure*10**5    #[Pa]

        #Calculate compressibility factor at given temp and pressure
        Z = CP.PropsSI('Z', 'P', P_given, 'T', T_given, fluid) 
    
        #Calculate volumetric storage density
        v = ((1/n)*Z*R*T_given) / P_given    #[m^3/kg]

        #Calculate required storage volume of hydrogen
        storage_volume = v * amount_hydrogen    #[m^3]

        #Compressor Work required
        """Assume 85% compressor efficiency. Initial state of hydrogen is 
        assumed to be 0 Pa and the same temperature after the 
        compression i.e. the input temperature."""

        comp_efficiency = 0.85  #Compressor efficiency

        #Hydrogen's enthalpy before compression [J/kg]
        P_initial = 2e6 #[Pa] Typical outlet pressure of PEM electrolyzer
        h_initial = CP.PropsSI('H','P', P_initial, 'T', T_given, fluid)  #[J/kg]
        
        #Hydrogen's enthaply after compression [J/kg]   
        h_final = CP.PropsSI('H','P', P_given, 'T', T_given, fluid) #[J/kg]

        w = (h_final - h_initial) / comp_efficiency #[J/kg] 

        compressor_work =  (amount_hydrogen * w) / (3.6e6)   #[kWh]

        """Financial metrics - Hydrogen storage in underground Type-1, API 5L X52 pipes
        https://www.sciencedirect.com/science/article/pii/S0360319921030834?via%3Dihub"""

        #Calculate Capex
        pressure_vessel_capex = 560 * amount_hydrogen #[USD]

        #Calculate OPEX
        pressure_vessel_opex = 84 * amount_hydrogen #[USD/yr]
        
        return storage_volume, compressor_work, pressure_vessel_capex, pressure_vessel_opex


if __name__ == '__main__':
    test=pressure_vessel(18750,20,100)
    print(test)

    static_pressure_vessel(True)
    
