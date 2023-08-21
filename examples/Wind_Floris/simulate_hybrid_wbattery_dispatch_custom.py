import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.hybrid_simulation import HybridSimulation

from hybrid.dispatch.plot_tools import plot_battery_output, plot_battery_dispatch_error, plot_generation_profile


from hybrid.keys import set_developer_nrel_gov_key
import yaml

set_developer_nrel_gov_key('')

# ADD CUSTOM WIND MODULE
# download FLORIS at www.github.com/NREL/FLORIS
# pip install -e floris
with open(Path(__file__).absolute().parent / "floris_input.yaml", 'r') as f:
    floris_config = yaml.load(f, yaml.SafeLoader)

# properties from floris
nTurbs = len(floris_config['farm']['properties']['layout_x'])

solar_size_mw = 50 #20
wind_size_mw = 50 #80
battery_capacity_mwh = 200 #30
interconnection_size_mw = 50 #100

technologies = {'pv': {
                    'system_capacity_kw': solar_size_mw * 1000,
                },
                'wind': {
                    'num_turbines': 25,
                    'turbine_rating_kw': 2000,
                    'model_name': 'floris',
                    'timestep': [0,8759],
                    'floris_config': floris_config # if not specified, use default SAM models
                },
                'battery': {
                    'system_capacity_kwh': 20 * 1000,
                    'system_capacity_kw': 5 * 1000
                },
                'grid': {
                    'interconnect_kw': interconnection_size_mw * 1000
                }}

# Get resource
lat = flatirons_site['lat']
lon = flatirons_site['lon']
prices_file = Path(__file__).parent.absolute().parent.parent / 'resource_files' / 'grid' / 'pricing-data-2015-IronMtn-002_factors.csv'
site = SiteInfo(flatirons_site, grid_resource_file=prices_file)

# Create model
hybrid_plant = HybridSimulation(technologies, site)

hybrid_plant.pv.system_capacity_kw = solar_size_mw * 1000
hybrid_plant.pv.dc_degradation = [0] * 25
hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)

hybrid_plant.ppa_price = 0.06   # [$/kWh]
hybrid_plant.simulate(2)

file = 'figures/'
tag = 'simple2_'
#plot_battery_dispatch_error(hybrid_plant, plot_filename=file+tag+'battery_dispatch_error.png')
'''
for d in range(0, 360, 5):
    plot_battery_output(hybrid_plant, start_day=d, plot_filename=file+tag+'day'+str(d)+'_battery_gen.png')
    plot_generation_profile(hybrid_plant, start_day=d, plot_filename=file+tag+'day'+str(d)+'_system_gen.png')
'''
plot_battery_dispatch_error(hybrid_plant)
plot_battery_output(hybrid_plant)
plot_generation_profile(hybrid_plant)
#plot_battery_dispatch_error(hybrid_plant, plot_filename=tag+'battery_dispatch_error.png')

# Save the outputs
annual_energies = hybrid_plant.annual_energies
npvs = hybrid_plant.net_present_values

print(annual_energies)
print(npvs)
