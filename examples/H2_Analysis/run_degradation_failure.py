from pathlib import Path
import matplotlib.pyplot as plt
from hybrid.sites import SiteInfo, flatirons_site
from hybrid.hybrid_simulation import HybridSimulation
from examples.H2_Analysis.degradation import Degradation
from examples.H2_Analysis.failure import Failure
import numpy as np
from hybrid.log import hybrid_logger as logger
from hybrid.keys import set_nrel_key_dot_env
from tools.analysis.bos.cost_calculator import create_cost_calculator

examples_dir = Path(__file__).resolve().parents[1]


# Set API key
set_nrel_key_dot_env()

# Set wind, solar, and interconnection capacities (in MW)
solar_size_mw = 50
wind_size_mw = 50
interconnection_size_mw = 50        #Required by HybridSimulation() not currently being used for calculations.
battery_capacity_mw = 20
battery_capacity_mwh = battery_capacity_mw * 4 
electrolyzer_capacity_mw = 40
useful_life = 30
load = [electrolyzer_capacity_mw*1000] * useful_life * 8760

technologies = {'pv': {
                'system_capacity_kw': solar_size_mw * 1000
            },
            'wind': {
                'num_turbines': 10,
                'turbine_rating_kw': 2000},
            'battery': {
                'system_capacity_kwh': battery_capacity_mwh * 1000,
                'system_capacity_kw': battery_capacity_mw * 1000
                }
            }

# Get resource
lat = flatirons_site['lat']
lon = flatirons_site['lon']
site = SiteInfo(flatirons_site)

# Create model
hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_mw * 1000)


hybrid_plant.pv.system_capacity_kw = solar_size_mw * 1000
hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)
hybrid_plant.ppa_price = 0.1
# hybrid_plant.pv.dc_degradation = [0] * 25
# hybrid_plant.wind._system_model.value("env_degrad_loss", 20)
hybrid_plant.simulate(useful_life)

# Save the outputs
generation_profile = hybrid_plant.generation_profile

# Instantiate degradation
hybrid_degradation = Degradation(technologies, True, electrolyzer_capacity_mw, useful_life, generation_profile, load)

# Simulate wind and pv degradation
hybrid_degradation.simulate_generation_degradation()

# Assign output from generation degradation to be fed into failure class
pv_deg = [x for x in hybrid_degradation.pv_degraded_generation]
wind_deg = [x for x in hybrid_degradation.wind_degraded_generation]

degraded_generation = dict()
degraded_generation['pv'] = pv_deg
degraded_generation['wind'] = wind_deg


# Instantiate failure 
# Use degraded generation profiles
hybrid_failure = Failure(technologies,False, useful_life,degraded_generation,load, False)

# # Add failures to generation technology
hybrid_failure.simulate_generation_failure()

# hybrid_degradation.hybrid_degraded_generation = [0]*8760*useful_life
# hybrid_degradation.pv_degraded_generation = [0]*8760*useful_life
# hybrid_degradation.wind_degraded_generation = [0]*8760*useful_life
# hybrid_degradation.simulate_battery_degradation()
# hybrid_degradation.simulate_electrolyzer_degradation()
# print("Number of battery repairs: ", hybrid_degradation.battery_repair)
# print("Number of electrolyzer repairs: ", hybrid_degradation.electrolyzer_repair)
print("Number of pv repairs: ", hybrid_failure.pv_repair)
print("Number of wind repairs: ", hybrid_failure.wind_repair)

print("Non-degraded lifetime pv power generation: ", np.sum(hybrid_plant.pv.generation_profile)/1000, "[MW]")
print("Degraded lifetime pv power generation: ", np.sum(hybrid_degradation.pv_degraded_generation)/1000, "[MW]")
print('Failed pv power generation: ', np.sum(hybrid_failure.pv_failed_generation)/1000, "[MW]")

print("Non-degraded lifetime wind power generation: ", np.sum(hybrid_plant.wind.generation_profile)/1000, "[MW]")
print("Degraded lifetime wind power generation: ", np.sum(hybrid_degradation.wind_degraded_generation)/1000, "[MW]")
print('Failed wind power generation: ', np.sum(hybrid_failure.wind_failed_generation)/1000, "[MW]")

# print("Battery used over lifetime: ", np.sum(hybrid_degradation.battery_used)/1000, "[MW]")
# print("Life-time Hydrogen production: ", np.sum(hybrid_degradation.hydrogen_hourly_production), "[kg]")

plot_degradation = False

if plot_degradation:
    plt.figure(figsize=(10,6))
    plt.subplot(311)
    plt.title("Max power generation vs degraded power generation")
    plt.plot(hybrid_degradation.wind_degraded_generation[175200:175344],label="degraded wind")
    plt.plot(hybrid_plant.wind.generation_profile[175200:175344],label="max generation")
    plt.ylabel("Power Production (kW)")
    plt.legend()
    
    plt.subplot(312)
    plt.plot(hybrid_degradation.pv_degraded_generation[175200:175344],label="degraded pv")
    plt.plot(hybrid_plant.pv.generation_profile[175200:175344],label="max generation")
    plt.ylabel("Power Production (kW)")
    plt.legend()

    plt.subplot(313)
    plt.plot(hybrid_degradation.hybrid_degraded_generation[175200:175344], label="degraded hybrid generation")
    plt.plot(load[175200:175344], label = "load profile")
    plt.ylabel("Power Production (kW)")
    plt.xlabel("Time (hour)")
    plt.legend()
    plt.show()

