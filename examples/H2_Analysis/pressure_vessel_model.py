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
from os import stat
import CoolProp.CoolProp as CP
import numpy as np
import math
import matplotlib.pyplot as plt

def static_pressure_vessel(H2_storage):
    """Nominal storage volume is 300 MWh (50 MW, 6 hours)
    https://www.nrel.gov/docs/fy10osti/48360.pdf
    https://www.energy.gov/sites/prod/files/2015/11/f27/fcto_fuel_cells_fact_sheet.pdf
    Assume 85% compressor efficiency
    Assume 60% Fuel cell conversion
    Assume 20 degrees celcius, isothermal compression
    Assume compressed to 100 bar
    """
    
    if H2_storage:
        #Round trip storage (PEM -> Compressor -> Storage -> Fuel Cell): 31250 kg H2
        storage_volume = 4007.9665  #[m^3]
        compressor_work = 380.9388  #[kWh]
        pressure_vessel_capex = 17500000    #[USD]
        pressure_vessel_opex = 2625000  #[USD/yr]
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
    
