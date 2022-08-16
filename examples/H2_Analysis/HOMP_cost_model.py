import json
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
from hybrid.sites import SiteInfo, flatirons_site
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.log import hybrid_logger as logger
from hybrid.keys import set_nrel_key_dot_env
from examples.H2_Analysis.simple_dispatch import SimpleDispatch
from examples.H2_Analysis.degradation import Degradation
from lcoe.lcoe import lcoe as lcoe_calc

examples_dir = Path(__file__).resolve().parents[1]
plot_degradation = True

# Set API key
set_nrel_key_dot_env()

# Set wind, solar, and interconnection capacities (in MW)
solar_size_mw = 50
wind_size_mw = 50
interconnection_size_mw = 50        #Required by HybridSimulation() not currently being used for calculations.
battery_capacity_mw = 20
battery_capacity_mwh = battery_capacity_mw * 4 
useful_life = 30
load = [40000] * useful_life * 8760
discount_rate = 0.07

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

# Save relevant HybridSimulation outputs
generation_profile = hybrid_plant.generation_profile
pv_capex = hybrid_plant.pv.total_installed_cost
wind_capex = hybrid_plant.wind.total_installed_cost
battery_capex = hybrid_plant.battery.total_installed_cost
electro_capex = 86160284            # from PEM electrolysis spreadsheet model
total_installed_cost = pv_capex + wind_capex + battery_capex + electro_capex

# Set base O&M costs
pv_base_om = 23 * solar_size_mw * 1000            # $/kW, accordin to 2022 ATB; includes asset management, insurance products, site security, cleaning, vegetation removal, and component failure
wind_base_om = 43 * wind_size_mw * 1000           # $/kW, according to 2022 ATB for landbased wind
battery_base_om = 0.025 * battery_capex     # 2.5% of CAPEX, according to 2022 ATB
electro_base_om = 0.05 * electro_capex      # According to PEM electrolysis spreadsheet model. Specifically 4796493 for ~86mil capex

# Set replacement costs
inverter_replace_cost = 0.25 * 250  # $0.25/kW for string inverters ($0.14/kW for central inverters); largest string inverters are 250kW and 350kW
wind_replace_cost = 300000
battery_replace_cost = 8000         # Battery replacement cost can be $5k-$11k
electro_replace_cost = 0.15 * electro_capex

# Run degradation
hybrid_degradation = Degradation(technologies, False, useful_life, generation_profile, load)
hybrid_degradation.simulate_degradation()

### Run failure (later) ###


# Calculate O&M costs from degradation
annualized_pv_om = pv_base_om + ((sum(hybrid_degradation.pv_repair)*inverter_replace_cost)/useful_life)    # Estimated PV O&M is $14-16/kW
print("Total PV O&M /kW: ", annualized_pv_om/(solar_size_mw * 1000))
annualized_wind_om = wind_base_om + ((sum(hybrid_degradation.wind_repair)*wind_replace_cost)/useful_life)
print("Total Wind O&M /kW: ", annualized_wind_om/(wind_size_mw * 1000))
annualized_battery_om = battery_base_om + ((sum(hybrid_degradation.battery_repair)*battery_replace_cost)/useful_life)
print("Total Battery O&M /kW: ", annualized_battery_om/(battery_capacity_mw * 1000))
# annualized_electro_om = electro_base_om + (sum(hybrid_degradation.electro_repair)*electro_replace_cost)/useful_life
annualized_electro_om = electro_base_om + (((useful_life // 7) * electro_replace_cost)/useful_life)
print("Total Electrolyzer O&M: ", annualized_electro_om)

# # Add O&M costs from failure
# annualized_pv_om += ((sum(hybrid_failure.pv_fail)*___)/(useful_life*solar_size_mw*1000)
# annualized_wind_om += ((sum(hybrid_failure.wind_fail)*3000)/(useful_life*wind_size_mw*1000)
# annualized_battery_om += ((sum(hybrid_failure.battery_fail)*8000)/useful_life
# annualized_electro_om += ((sum(hybrid_failure.electro_fail)*___)/useful_life


# Calculate adjusted financial parameters manually
print("Lifetime Electricity Generation: ", sum(hybrid_degradation.hybrid_degraded_generation))
aep = (sum(hybrid_degradation.hybrid_degraded_generation)/useful_life) * 8760                   # Input in kWh
total_annualized_om = annualized_pv_om + annualized_wind_om + annualized_battery_om + annualized_electro_om
annual_h2_prod = aep / 55.5          # Based on kWh/kg H2 from PEM electrolysis spreadsheet model
lcoe_homp = lcoe_calc(aep, total_installed_cost, total_annualized_om, discount_rate, useful_life)
lcoh_homp = lcoe_calc(annual_h2_prod, total_installed_cost, total_annualized_om, discount_rate, useful_life)

print("LCOE: ", lcoe_homp)
print("LCOH: ", lcoh_homp)