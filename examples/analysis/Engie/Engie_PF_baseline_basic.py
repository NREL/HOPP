import os
from dotenv import load_dotenv

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.log import hybrid_logger as logger
from hybrid.keys import set_developer_nrel_gov_key
from tools.analysis import create_cost_calculator
import json

# Set API key
load_dotenv()
NREL_API_KEY = os.getenv("NREL_API_KEY")
set_developer_nrel_gov_key(NREL_API_KEY)  # Set this key manually here if you are not setting it using the .env

# TODO for Engie Setup
# Get Longi 550 Curve data and incorporate in inputs:
# cec_adjust
# cec_alpha_sc
# cec_area
# cec_beta_oc
# cec_gamma_r
# cec_i_l_ref
# cec_i_mp_ref
# cec_i_o_ref
# cec_i_sc_ref
# cec_module_length
# cec_n_s
# cec_r_s
# cec_r_sh_ref
# cec_t_noct
# cec_v_mp_ref
# cec_v_oc_ref
# cp_system_nameplate

# Get Metmast data into hourly SAM format
# Open up variables for desired analysis:
# NPV, Row spacing min: 4ft, Row Spacing max: 9ft, Min DC:AC: 0.95, MAX DC:AC: 1.45
# Interconnection: 220MW / 230KV, gen-tie length 1.24miles
# Solar System Design:
#  Modules per string: 30
#  Max string circuit loss: 2.5%, Max DC Feeder Circuit Loss (central inverters): 4%
#  Max LC AC Feeder Circuit Loss (String Inverters): 4%
#  Max Combiner Boxes: 36
#  Avg Annual Soiling: 0.2% per month
#  Energy Availability: 98.75%
#  Non-Ohmic DC Losses: 4%
# Set wind, solar, and interconnection capacities (in MW)

solar_size_mw = 200
wind_size_mw = 200
interconnection_size_mw = 220

technologies = {'solar': solar_size_mw,  # mw system capacity
                'wind': wind_size_mw,  # mw system capacity
                'grid': interconnection_size_mw,
                'collection_system': True}

# Get resource
lat = flatirons_site['lat']
lon = flatirons_site['lon']
site = SiteInfo(flatirons_site)

# Create Hybrid model
hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_mw * 1000)

# Set up cost model
hybrid_plant.setup_cost_calculator(create_cost_calculator(interconnection_size_mw))

# Set up other project parameters
hybrid_plant.solar.system_capacity_kw = solar_size_mw * 1000
hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)
hybrid_plant.ppa_price = 0.1


# Import power + thrust curves
turbine_choice = 'SG170 6MW'
powercurve_filename = 'powercurve_{}.json'.format(turbine_choice)
powercurve_file = open(powercurve_filename)
powercurve_data = json.load(powercurve_file)
powercurve_file.close()

thrustcurve_filename = 'thrustcurve_{}.json'.format(turbine_choice)
thrustcurve_file = open(thrustcurve_filename)
thrustcurve_data = json.load(thrustcurve_file)
powercurve_file.close()
hybrid_plant.wind.system_model.Turbine.wind_turbine_powercurve_windspeeds = \
    powercurve_data['wind_speed_ms']
hybrid_plant.wind.system_model.Turbine.wind_turbine_powercurve_powerout = \
    powercurve_data['turbine_power_output']

# Run HOPP Analysis
hybrid_plant.simulate(35)

# Save the outputs
annual_energies = hybrid_plant.annual_energies
wind_plus_solar_npv = hybrid_plant.net_present_values.wind + hybrid_plant.net_present_values.solar
npvs = hybrid_plant.net_present_values

wind_installed_cost = hybrid_plant.wind.financial_model.SystemCosts.total_installed_cost
solar_installed_cost = hybrid_plant.solar.financial_model.SystemCosts.total_installed_cost
hybrid_installed_cost = hybrid_plant.grid.financial_model.SystemCosts.total_installed_cost

print("Wind Installed Cost: {}".format(wind_installed_cost))
print("Solar Installed Cost: {}".format(solar_installed_cost))
print("Hybrid Installed Cost: {}".format(hybrid_installed_cost))
print("Wind NPV: {}".format(hybrid_plant.net_present_values.wind))
print("Solar NPV: {}".format(hybrid_plant.net_present_values.solar))
print("Hybrid NPV: {}".format(hybrid_plant.net_present_values.hybrid))
print("Wind + Solar Expected NPV: {}".format(wind_plus_solar_npv))


print(annual_energies)
print(npvs)
