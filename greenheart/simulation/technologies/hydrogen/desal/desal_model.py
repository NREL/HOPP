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
import sys
import numpy as np
from greenheart.to_organize.H2_Analysis.simple_cash_annuals import simple_cash_annuals

def RO_desal(net_power_supply_kW, desal_sys_size, useful_life, plant_life, \
    water_recovery_ratio = 0.30, energy_conversion_factor = 4.2, \
    high_pressure_pump_efficency = 0.70, pump_pressure_kPa = 5366,
    energy_recovery = 0.40):

    """
    Calculates the fresh water flow rate (m^3/hr) as
    a function of supplied power (kW) in RO desal.
    Calculats CAPEX (USD), OPEX (USD/yr), annual cash flows
    based on system's rated capacity (m^3/hr).

    param: net_power_supply_kW: (list), hourly power input [kW]

    param: desal_sys_size: Given as desired fresh water flow rate [m^3/hr]

    param: useful_life: useful life of desal system [years]

    param: plant_life: years of plant operation [years]

    Assumed values:
    Common set points from:
    https://www.sciencedirect.com/science/article/abs/pii/S0011916409008443
    water_recovery_ratio = 0.30
    energy_conversion_factor = 4.2
    high_pressure_pump_efficency = 0.70
    pump_pressure_kPa = 5366    (kept static for simplicity. TODO: Modify pressure through RO process)
    energy_recovery = 0.40
    Assumed energy savings by energy recovery device to be 40% of total energy
    https://www.sciencedirect.com/science/article/pii/S0360544210005578?casa_token=aEz_d_LiSgYAAAAA:88Xa6uHMTZee-djvJIF9KkhpuZmwZCLPHNiThmcwv9k9RC3H17JuSoRWI-l92rrTl_E3kO4oOA


    TODO: modify water recovery to vary based on salinity
    SWRO: Sea water Reverse Osmosis, water >18,000 ppm
    SWRO energy_conversion_factor range 2.5 to 4.2 kWh/m^3

    BWRO: Brakish water Reverse Osmosis, water < 18,000 ppm
    BWRO energy_conversion_factor range 1.0 to 1.5 kWh/m^3
    Source: https://www.sciencedirect.com/science/article/pii/S0011916417321057
    """
    # net_power_supply_kW = np.array(net_power_supply_kW)

    desal_power_max = desal_sys_size * energy_conversion_factor #kW

    # Modify power to not exceed system's power maximum (100% rated power capacity) or
    # minimum (approx 50% rated power capacity --> affects filter fouling below this level)
    net_power_for_desal = list()
    operational_flags = list()
    feed_water_flowrate = list()
    fresh_water_flowrate = list()
    for i, power_at_time_step in enumerate(net_power_supply_kW):
        if power_at_time_step > desal_power_max:
            current_net_power_available = desal_power_max
            operational_flag = 2
        elif (0.5 * desal_power_max) <= power_at_time_step <= desal_power_max:
            current_net_power_available = power_at_time_step
            operational_flag = 1
        elif power_at_time_step <= 0.5 * desal_power_max:
            current_net_power_available = 0
            operational_flag = 0

        # Append Operational Flags to a list
        operational_flags.append(operational_flag)
        # Create list of net power available for desal at each timestep
        net_power_for_desal.append(current_net_power_available)

        # Create list of feedwater flowrates based on net power available for desal
        # https://www.sciencedirect.com/science/article/abs/pii/S0011916409008443
        instantaneous_feed_water_flowrate = ((current_net_power_available * (1 + energy_recovery))\
        * high_pressure_pump_efficency) / pump_pressure_kPa * 3600 #m^3/hr

        instantaneous_fresh_water_flowrate = instantaneous_feed_water_flowrate * water_recovery_ratio  # m^3/hr

        feed_water_flowrate.append(instantaneous_feed_water_flowrate)
        fresh_water_flowrate.append(instantaneous_fresh_water_flowrate)

    # print("Fresh water flowrate: ", fresh_water_flowrate, "m^3/hr")
    # print(net_power_for_desal)
    # net_power_supply_kW = np.where(net_power_supply_kW >= desal_power_max, \
    #     desal_power_max, net_power_supply_kW)
    # net_power_supply_kW = np.where(net_power_supply_kW < 0.5 * desal_power_max, \
    #      0, net_power_supply_kW)
    # print("Net power supply after checks: ",net_power_supply_kW, "kW")


    """Values for CAPEX and OPEX given as $/(kg/s)
    Source: https://www.nrel.gov/docs/fy16osti/66073.pdf
    Assumed density of recovered water = 997 kg/m^3"""

    desal_capex = 32894 * (997 * desal_sys_size / 3600) # Output in USD
    # print("Desalination capex: ", desal_capex, " USD")

    desal_opex = 4841 * (997 * desal_sys_size / 3600) # Output in USD/yr
    # print("Desalination opex: ", desal_opex, " USD/yr")

    """
    Assumed useful life = payment period for capital expenditure.
    compressor amortization interest = 3%
    """
    desal_annuals = simple_cash_annuals(plant_life, useful_life,\
            desal_capex,desal_opex, 0.03)
    # a = 0.03
    # desal_annuals = [0] * useful_life

    # desal_amortization = desal_capex * \
    #     ((a*(1+a)**useful_life)/((1+a)**useful_life - 1))

    # for i in range(len(desal_annuals)):
    #     if desal_annuals[i] == 0:
    #         desal_annuals[i] = desal_amortization + desal_opex
    #     return desal_annuals        #[USD]

    return fresh_water_flowrate, feed_water_flowrate, operational_flags, desal_capex, desal_opex, desal_annuals

# Power = np.linspace(0, 100, 100)
# system_size = np.linspace(1,1000,1000)        #m^3/hr

# f = RO_desal(Power,system_size)

# plt.plot(system_size,f,color="C0")
# plt.xlabel("Desalination System Size [m^3/hr]")
# plt.ylabel("Desalination OPEX [USD/yr]")
# plt.show()

if __name__ == '__main__':
    Power = np.array([446,500,183,200,250,100])
    test = RO_desal(Power,300,30,30)
    print(test)