from pathlib import Path
from hybrid.sites import SiteInfo, flatirons_site
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.log import hybrid_logger as logger
from tools.analysis import create_cost_calculator

from hybrid.plot_tools import plot_battery_output, plot_battery_dispatch_error, plot_generation_profile


from hybrid.keys import set_developer_nrel_gov_key

set_developer_nrel_gov_key('')


solar_size_mw = 50 #20
wind_size_mw = 50 #80
battery_capacity_mwh = 200 #30
interconnection_size_mw = 50 #100

technologies = {'solar': solar_size_mw,  # mw system capacity
                'wind': wind_size_mw,  # mw system capacity
                'battery': battery_capacity_mwh,
                'grid': interconnection_size_mw}  # TODO: why is this specified twice?

# Get resource
lat = flatirons_site['lat']
lon = flatirons_site['lon']
prices_file = Path(__file__).parent.parent / "resource_files" / "grid" / "pricing-data-2019-IronMtn-002_factors.csv"
site = SiteInfo(flatirons_site,
                grid_resource_file=prices_file)
# Create model
hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_mw * 1000)

# Setup cost model
hybrid_plant.setup_cost_calculator(create_cost_calculator(interconnection_size_mw))

hybrid_plant.solar.system_capacity_kw = solar_size_mw * 1000

hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)

hybrid_plant.ppa_price = 0.06   # [$/kWh]
hybrid_plant.simulate(25, is_simple_battery_dispatch=True, is_test=False)

file = 'figures/'
tag = 'simple2_'
plot_battery_dispatch_error(hybrid_plant, plot_filename=file+tag+'battery_dispatch_error.png')
'''
for d in range(0, 360, 5):
    plot_battery_output(hybrid_plant, start_day=d, plot_filename=file+tag+'day'+str(d)+'_battery_gen.png')
    plot_generation_profile(hybrid_plant, start_day=d, plot_filename=file+tag+'day'+str(d)+'_system_gen.png')
'''
plot_battery_output(hybrid_plant)
plot_generation_profile(hybrid_plant)
#plot_battery_dispatch_error(hybrid_plant, plot_filename=tag+'battery_dispatch_error.png')

# Save the outputs
annual_energies = hybrid_plant.annual_energies
npvs = hybrid_plant.net_present_values

print(annual_energies)
print(npvs)
