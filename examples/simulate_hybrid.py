from hybrid.sites import SiteInfo, flatirons_site
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.log import hybrid_logger as logger
from tools.analysis import create_cost_calculator


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

hybrid_plant.ppa_price = 0.03
hybrid_plant.simulate(25)

# Save the outputs
annual_energies = hybrid_plant.annual_energies
npvs = hybrid_plant.net_present_values

print(annual_energies)
print(npvs)
