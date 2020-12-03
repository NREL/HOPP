import os
from dotenv import load_dotenv

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.log import hybrid_logger as logger
from hybrid.keys import set_developer_nrel_gov_key
from tools.analysis import create_cost_calculator


# Set API key
load_dotenv()
NREL_API_KEY = os.getenv("NREL_API_KEY")
set_developer_nrel_gov_key(NREL_API_KEY)  # Set this key manually here if you are not setting it using the .env

# Set wind, solar, and interconnection capacities (in MW)
solar_size_mw = 20
wind_size_mw = 80
interconnection_size_mw = 100

technologies = {'solar': solar_size_mw,  # mw system capacity
                'wind': wind_size_mw,  # mw system capacity
                'grid': interconnection_size_mw}

# Get resource
lat = flatirons_site['lat']
lon = flatirons_site['lon']
site = SiteInfo(flatirons_site)

# Create model
hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_mw * 1000)

# Setup cost model
hybrid_plant.setup_cost_calculator(create_cost_calculator(interconnection_size_mw))
hybrid_plant.solar.system_capacity_kw = solar_size_mw * 1000
hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)
hybrid_plant.ppa_price = 0.10
hybrid_plant.simulate(25)

# Save the outputs
annual_energies = hybrid_plant.annual_energies
npvs = hybrid_plant.net_present_values

print(annual_energies)
print(npvs)
