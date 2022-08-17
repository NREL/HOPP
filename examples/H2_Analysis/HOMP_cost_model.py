## HOMP cost model calcuations. This file is for reference for the calculation method
## File does not run b/c it does not have the degradatio and failure hierarchy set up.
## A running version with degradation and failure incorporated is in run_degradation_failure.py

import json
from pathlib import Path
import numpy as np
import numpy_financial as npf

import matplotlib.pyplot as plt
from hybrid.sites import SiteInfo, flatirons_site
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.log import hybrid_logger as logger
from hybrid.keys import set_nrel_key_dot_env
from examples.H2_Analysis.simple_dispatch import SimpleDispatch
from examples.H2_Analysis.degradation import Degradation
from lcoe.lcoe import lcoe as lcoe_calc
from examples.H2_Analysis.simple_cash_annuals import simple_cash_annuals


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
electrolyzer_capacity_mw = 1000
useful_life = 30
load = [40000] * useful_life * 8760
discount_rate = 0.07
amortization_interest = 0.03

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
electro_capex = 460 * electrolyzer_capacity_mw * 1000           # From HOMP table
total_installed_cost = pv_capex + wind_capex + battery_capex + electro_capex

# Set base O&M costs
pv_base_om = 23 * solar_size_mw * 1000            # $/kW, accordin to 2022 ATB; includes asset management, insurance products, site security, cleaning, vegetation removal, and component failure
wind_base_om = 43 * wind_size_mw * 1000           # $/kW, according to 2022 ATB for landbased wind
battery_base_om = 0.025 * battery_capex           # 2.5% of CAPEX, according to 2022 ATB
electro_base_om = 0.05 * electro_capex            # According to PEM electrolysis spreadsheet model. Specifically 4796493 for ~86mil capex

# Set replacement costs
inverter_replace_cost = 0.25 * 250                # $0.25/kW for string inverters ($0.14/kW for central inverters); largest string inverters are 250kW and 350kW
wind_replace_cost = 300000
battery_replace_cost = 8000                       # Battery replacement cost can be $5k-$11k
electro_replace_cost = 0.15 * electro_capex

# Run degradation
hybrid_degradation = Degradation(technologies, False, useful_life, generation_profile, load)
hybrid_degradation.simulate_degradation()

### Run failure (doesn't occur here; does occur in run_degradation_failure.py) ###


# Calculate full cashflows (no grid connection)
pv_cf_no_replace = simple_cash_annuals(useful_life, useful_life, pv_capex, pv_base_om, amortization_interest)
wind_cf_no_replace = simple_cash_annuals(useful_life, useful_life, wind_capex, wind_base_om, amortization_interest)
battery_cf_no_replace = simple_cash_annuals(useful_life, useful_life, battery_capex, battery_base_om, amortization_interest)
h2_cf_no_replace = simple_cash_annuals(useful_life, useful_life, electro_capex, electro_base_om, amortization_interest)

# Add repalcement cashflows
# Relevant for integration into runfile; doesn't work in this file b/c failure isn't incorporated
pv_cf = np.add((hybrid_failure.pv_repair * inverter_replace_cost), pv_cf_no_replace)
wind_cf = np.add((hybrid_failure.wind_repair * wind_replace_cost), wind_cf_no_replace)
battery_cf = np.add((hybrid_failure.battery_repair_failure * battery_replace_cost), battery_cf_no_replace)
electrolyzer_cf = np.add((hybrid_failure.electrolyzer_repair_failure * electro_replace_cost), h2_cf_no_replace)

# Calculate NPVs
pv_npv = npf.npv(discount_rate, pv_cf)
wind_npv = npf.npv(discount_rate, wind_cf)
battery_npv = npf.npv(discount_rate, battery_cf)
h2_npv = npf.npv(discount_rate, electrolyzer_cf)
total_npv = pv_npv + wind_npv + battery_npv + h2_npv

# Calculate LCOE
LCOH_cf_method = -total_npv / (H2_Results['hydrogen_annual_output'] * useful_life)                          # Relevant for integration into run file; doesn't work within this file
LCOE_cf_method = -total_npv / ((sum(hybrid_degradation.hybrid_degraded_generation)/useful_life) * 8760)

# Print results
print("LCOH: ", LCOH_cf_method)
print("LCOE: ", LCOE_cf_method)
print("Total NPV: ", total_npv)
print("PV NPV: ", pv_npv)
print("Wind NPV: ", wind_npv)
print("Battery NPV: ", battery_npv)
print("Electrolyzer NPV: ", h2_npv)