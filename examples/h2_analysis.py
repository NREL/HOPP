import os
from dotenv import load_dotenv
from math import sin, pi
from hybrid.reopt import REopt
from hybrid.solar_source import SolarPlant
from hybrid.wind_source import WindPlant
import PySAM.Singleowner as so

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.log import hybrid_logger as logger
from hybrid.keys import set_developer_nrel_gov_key
from tools.analysis import create_cost_calculator


# Set API key
load_dotenv()
NREL_API_KEY = os.getenv("NREL_API_KEY")
set_developer_nrel_gov_key(NREL_API_KEY)  # Set this key manually here if you are not setting it using the .env

# Get resource
lat = flatirons_site['lat']
lon = flatirons_site['lon']
site = SiteInfo(flatirons_site)

# Run ReOpt
MTC02_yr = 10640
MTC02_yr_to_kw_continuous_conversion = 0.2779135
kw_continuous = MTC02_yr * MTC02_yr_to_kw_continuous_conversion
load = [kw_continuous * 1000 * (sin(x) + pi) for x in range(0, 8760)]
urdb_label = "5ca4d1175457a39b23b3d45e"  # https://openei.org/apps/IURDB/rate/view/5ca3d45ab718b30e03405898

solar_model = SolarPlant(site, 20000)
wind_model = WindPlant(site, 20000)
wind_model.system_model.Resource.wind_resource_filename = os.path.join(
    "data", "39.7555_-105.2211_windtoolkit_2012_60min_60m.srw")
fin_model = so.default("GenericSystemSingleOwner")
filepath = os.path.dirname(os.path.abspath(__file__))
fileout = os.path.join(filepath, "data", "REoptResultsNoExportAboveLoad.json")
print(fileout)
reopt = REopt(lat=lat,
              lon=lon,
              load_profile=load,
              urdb_label=urdb_label,
              solar_model=solar_model,
              wind_model=wind_model,
              fin_model=fin_model,
              interconnection_limit_kw=20000,
              fileout=os.path.join(filepath, "data", "REoptResultsNoExportAboveLoad.json"))

reopt.set_rate_path(os.path.join(filepath, 'data'))
result = reopt.get_reopt_results()

reopt_site = reopt.post['Scenario']['Site']
pv = reopt_site['PV']
wind = reopt_site['Wind']

# Set wind, solar, and interconnection capacities (in MW)
solar_size_mw = 50
wind_size_mw = 50
interconnection_size_mw = 50

technologies = {'solar': solar_size_mw,  # mw system capacity
                'wind': wind_size_mw,  # mw system capacity
                'grid': interconnection_size_mw,
                'collection_system': True}

# Create model
hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_mw * 1000)

# Setup cost model
hybrid_plant.setup_cost_calculator(create_cost_calculator(interconnection_size_mw))
hybrid_plant.solar.system_capacity_kw = solar_size_mw * 1000
hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)
hybrid_plant.ppa_price = 0.1
hybrid_plant.simulate(25)

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
