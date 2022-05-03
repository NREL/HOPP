## High-Pressure Reverse Osmosis Desalination Model
"""
Python model of High-Pressure Reverse Osmosis Desalination (HPRO).

Reverse Osmosis (RO) is a membrane separation process. No heating or phase change is necessary.
The majority of energy required is for pressurizing the feed water.

A typical RO system is made up of the following basic components:
Pre-treatment: Removes suspended solids and microorganisms through sterilization, fine filtration and adding chemicals to inhibit precipitation.
High-pressure pump: Supplies he pressure needed to enable the water to pass through the membrane (pressure ranges from 54 to 80 bar for seawater).
Membrane Modules: Membrane assembly consists of a pressure vessel and the membrane. Either sprial wound membranes or hollow fiber membranes are used.
Post-treatment: Consists of sterilization, stabilization, mineral enrichment and pH adjustment of product water.
Energy recovery system: A system where a portion of the pressure energy of the brine is recovered.
"""
import sys
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

np.set_printoptions(threshold=sys.maxsize)


def RO_desal(fresh_water_quantity, feed_water_flowrate = 2500, \
    recovery_ratio = 0.45, energy_conversion_factor = 4.0):
    """This function calculates the required energy 
    to produce X amount of fresh water with reverse osmosis desalination.
    Also calculats CAPEX and OPEX for a generic desal system.

    Add units here for arguments:

    SWRO: Sea water Reverse Osmosis, water >18,000 ppm 
    SWRO energy_conversion_factor range 2.5 to 4.0 kWh/m^3

    BWRO: Brakish water Reverse Osmosis, water < 18,000 ppm
    BWRO energy_conversion_factor range 1.0 to 1.5 kWh/m^3

    TODO: link SWRO_fresh_water_quantity to fresh water needed by Electrolyzer

    Conservative conversion rate based on actual data from >20 SWRO 
    between 2005-2010. Energy conversion varies from 2.5 to 4.0 kWh/m^3.
    Source: https://www.sciencedirect.com/science/article/pii/S0011916417321057"""

    energy_required = fresh_water_quantity * energy_conversion_factor  # kWh
    print("Energy required to produce fresh water: ", energy_required, " kWh")
    
    """TODO: Add various feed_water_flowrates and associated power requirements.
    PureAqua was used in previous NREL investigation
    https://pureaqua.com/content/pdf/industrial-seawater-reverse-osmosis-desalination-systems.pdf"""

    # feed_water_flowrate = 2500  # m^3/hr
    # recovery_ratio = 0.45    # fresh_water_flow_rate / feed_water_flow_rate

    fresh_water_flowrate = feed_water_flowrate * recovery_ratio  # m^3/hr

    desal_power_required = energy_required / (fresh_water_quantity / fresh_water_flowrate) # kW
    print("Power required to produce freshwater: ", desal_power_required, "kW")

# Specific energy
# Use to predict the required energy for generic desalination system
#E = (P_f*Q_f*(E_pump)**(-1) - (P_r*Q_r*E_ERD))/Q_p

# E = specific energy consumption (kWh)
# P_f = Feed water pressure (Pa)
# Q_f = Feed flow rate (m^3/day)
# E_pump = Pump energy consumption (kWh)
# P_r = Rejected pressure (Pa)
# E_ERD = Turbine energy/Energy recovery device (kWh)
# Q_p = Permeate flow rate (m^3/day)

    """Source: https://www.nrel.gov/docs/fy16osti/66073.pdf
    Assumed density of recovered water = 997 kg/m^3"""

    desal_capex = 32894 * (997 * fresh_water_flowrate / 3600) # Output in USD
    print("Desalination capex: ", desal_capex, " USD")

    """Source: https://www.nrel.gov/docs/fy16osti/66073.pdf
    Assumed density of recovered water = 997 kg/m^3"""

    desal_opex = 4841 * (997 * fresh_water_flowrate / 3600) # Output in USD/yr
    print("Desalination opex: ", desal_opex, " USD/yr")
    return

RO_desal(20000)


