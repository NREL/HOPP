## High-Pressure Reverse Osmosis Desalination Model
"""
Python model of High-Pressure Reverse Osmosis Desalination (HPRO).

Reverse Osmosis (RO) is a membrane separation process. No heating or phase change is necessary.
The majority of energy required is for pressurizing the feed water.

A typical RO system is made up of the following basic components:
Pre-treatment: Removes suspended solids and microorganisms through sterilization, fine filtration and adding chemicals to inhibit precipitation.
High-pressure pump: Supplies the pressure needed to enable the water to pass through the membrane (pressure ranges from 54 to 80 bar for seawater).
Membrane Modules: Membrane assembly consists of a pressure vessel and the membrane. Either sprial wound membranes or hollow fiber membranes are used.
Post-treatment: Consists of sterilization, stabilization, mineral enrichment and pH adjustment of product water.
Energy recovery system: A system where a portion of the pressure energy of the brine is recovered.
"""
from os import system
import sys
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

np.set_printoptions(threshold=sys.maxsize)


def RO_desal(net_power_supply_kW, desal_sys_size, \
    water_recovery_ratio = 0.40, energy_conversion_factor = 2.928, \
    high_pressure_pump_efficency = 0.85, pump_pressure_kPa = 6370,
    energy_recovery = 0.40):

    """This function calculates the fresh water flow rate (m^3/hr) as 
    a function of supplied power (kW) in reverse osmosis desalination.
    Also calculats CAPEX (USD) and OPEX (USD/yr) based on system's 
    rated capacity (m^3/hr).

    :param net_power_supply_kW: ``list``,
        hourly power input (kW)

    desal_sys_size: Fresh water flow rate [m^3/hr]

    SWRO: Sea water Reverse Osmosis, water >18,000 ppm 
    SWRO energy_conversion_factor range 2.5 to 4.0 kWh/m^3

    BWRO: Brakish water Reverse Osmosis, water < 18,000 ppm
    BWRO energy_conversion_factor range 1.0 to 1.5 kWh/m^3
    Source: https://www.sciencedirect.com/science/article/pii/S0011916417321057

    TODO: link fresh water produced by desal to fresh water needed by Electrolyzer 
    Make sure to not over or under produce water for electrolyzer.
    """
    net_power_supply_kW = np.array(net_power_supply_kW)
    
    desal_power_max = desal_sys_size * energy_conversion_factor #kW
    print("Max power allowed by system: ", desal_power_max, "kW")
    
    # Modify power to not exceed system's power maximum (100% rated power capacity) or
    # minimum (approx 50% rated power capacity --> affects filter fouling below this level)
    net_power_supply_kW = np.where(net_power_supply_kW >= desal_power_max, \
        desal_power_max, net_power_supply_kW)
    net_power_supply_kW = np.where(net_power_supply_kW < 0.5 * desal_power_max, \
         0, net_power_supply_kW)
    # print("Net power supply after checks: ",net_power_supply_kW, "kW")

    feed_water_flowrate = ((net_power_supply_kW * (1 + energy_recovery))\
        * high_pressure_pump_efficency) / pump_pressure_kPa * 3600 #m^3/hr
     
    fresh_water_flowrate = feed_water_flowrate * water_recovery_ratio  # m^3/hr
    # print("Fresh water flowrate: ", fresh_water_flowrate, "m^3/hr")



    """Values for CAPEX and OPEX given as $/(kg/s)
    Source: https://www.nrel.gov/docs/fy16osti/66073.pdf
    Assumed density of recovered water = 997 kg/m^3"""

    desal_capex = 32894 * (997 * desal_sys_size / 3600) # Output in USD
    # print("Desalination capex: ", desal_capex, " USD")

    desal_opex = 4841 * (997 * desal_sys_size / 3600) # Output in USD/yr
    # print("Desalination opex: ", desal_opex, " USD/yr")
    
    return fresh_water_flowrate, feed_water_flowrate, desal_capex, desal_opex

# Power = np.linspace(0, 100, 100)
# system_size = np.linspace(1,1000,1000)        #m^3/hr

# f = RO_desal(Power,system_size)

# plt.plot(system_size,f,color="C0")
# plt.xlabel("Desalination System Size [m^3/hr]")
# plt.ylabel("Desalination OPEX [USD/yr]")
# plt.show()

if __name__ == '__main__':
    Power = np.array([446,500,183,200,250,100])
    test = RO_desal(Power,100000)
    print(test)