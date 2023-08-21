import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

import yaml
from pathlib import Path

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.keys import set_nrel_key_dot_env

# ADD CUSTOM WIND MODULE
# download FLORIS at www.github.com/NREL/FLORIS
# pip install -e floris
with open(Path(__file__).absolute().parent / "floris_input.yaml", 'r') as f:
    floris_config = yaml.load(f, yaml.SafeLoader)

# properties from floris
nTurbs = len(floris_config['farm']['layout_x'])

# Set API key
set_nrel_key_dot_env()

# Set wind, solar, and interconnection capacities (in MW)
solar_size_mw = 20

# TODO: couple this to the custom model - floris
wind_size_mw = 20
interconnection_size_mw = 20

# Get resource
lat = flatirons_site['lat']
lon = flatirons_site['lon']
prices_file = Path(__file__).parent.absolute().parent.parent / 'resource_files' / 'grid' / 'pricing-data-2015-IronMtn-002_factors.csv'
site = SiteInfo(flatirons_site, grid_resource_file=prices_file)

# initialize custom model
rated_power = 5000 # NREL 5MW turbine  # TODO: generate rated power from input file

technologies = {'pv': {
                    'system_capacity_kw': solar_size_mw * 1000
                },
                'wind': {
                    'num_turbines': nTurbs,
                    'turbine_rating_kw': rated_power,
                    'model_name': 'floris',
                    'timestep': [0,8759],
                    'floris_config': floris_config # if not specified, use default SAM models
                }}

# Create model
hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=interconnection_size_mw * 1000)

hybrid_plant.pv.system_capacity_kw = solar_size_mw * 1000
hybrid_plant.pv.dc_degradation = [0] * 25
hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)
hybrid_plant.ppa_price = 0.1
hybrid_plant.simulate(2)

# Save the outputs
annual_energies = hybrid_plant.annual_energies
wind_plus_solar_npv = hybrid_plant.net_present_values.wind + hybrid_plant.net_present_values.pv
npvs = hybrid_plant.net_present_values

wind_installed_cost = hybrid_plant.wind.total_installed_cost
solar_installed_cost = hybrid_plant.pv.total_installed_cost
hybrid_installed_cost = hybrid_plant.grid.total_installed_cost

print("Wind Installed Cost: {}".format(wind_installed_cost))
print("Solar Installed Cost: {}".format(solar_installed_cost))
print("Hybrid Installed Cost: {}".format(hybrid_installed_cost))
print("Wind NPV: {}".format(hybrid_plant.net_present_values.wind))
print("Solar NPV: {}".format(hybrid_plant.net_present_values.pv))
print("Hybrid NPV: {}".format(hybrid_plant.net_present_values.hybrid))
print("Wind + Solar Expected NPV: {}".format(wind_plus_solar_npv))


print(annual_energies)
print(npvs)
