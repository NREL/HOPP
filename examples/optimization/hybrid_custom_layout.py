import sys
sys.path.append('/Users/jannoni/Desktop/Desktop/Repos/HOPP_FLORIS/HOPP/')

from pathlib import Path
import matplotlib.pyplot as plt
from hybrid.sites import make_circular_site, make_irregular_site, SiteInfo, locations
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.layout.wind_layout import WindCustomParameters
from hybrid.layout.solar_layout import SolarGridParameters
from hybrid.layout.flicker_mismatch import FlickerMismatch
from hybrid.layout.flicker_data.plot_flicker import plot_contour


# ADD CUSTOM WIND MODULE
# download FLORIS at www.github.com/NREL/FLORIS
# pip install -e floris
import json
with open("/Users/jannoni/Desktop/Desktop/Repos/HOPP_FLORIS/floris/examples/example_input.json", 'r') as f:
    floris_config = json.load(f)

# parameters from floris
nTurbs = len(floris_config['farm']['properties']['layout_x'])
rated_power = 5000 # NREL 5MW in kW

site = 'irregular'
location = locations[1]
site_data = None

if site == 'circular':
    site_data = make_circular_site(lat=location[0], lon=location[1], elev=location[2])
elif site == 'irregular':
    site_data = make_irregular_site(lat=location[0], lon=location[1], elev=location[2])
else:
    raise Exception("Unknown site '" + site + "'")

# g_file = Path(__file__).parent.parent.parent / "resource_files" / "grid" / "pricing-data-2015-IronMtn-002_factors.csv"
g_file = '../../resource_files/grid/pricing-data-2015-IronMtn-002_factors.csv'
site_info = SiteInfo(site_data, grid_resource_file=g_file)

# set up hybrid simulation with all the required parameters
solar_size_mw = 100
interconnection_size_mw = 150
wind_size_mw = (rated_power * nTurbs) / 1000

technologies = {'solar': {
                    'system_capacity_kw': solar_size_mw * 1000,
                    'layout_params': SolarGridParameters(x_position=0.5,
                                                         y_position=0.5,
                                                         aspect_power=0,
                                                         gcr=0.5,
                                                         s_buffer=2,
                                                         x_buffer=2)
                },
                'wind': {
                    'num_turbines': nTurbs,
                    'turbine_rating_kw': rated_power,
                    'layout_mode': 'custom',
                    'timestep': [0,8759],
                    'model_name': 'floris',
                    'floris_config': floris_config # if not specified, use default SAM models

                },
                'grid': interconnection_size_mw}

# Create model
hybrid_plant = HybridSimulation(technologies, site_info, interconnect_kw=interconnection_size_mw * 1000)

# plot the layout
hybrid_plant.plot_layout()
plt.show()

# simulate the hybrid
hybrid_plant.solar.system_capacity_kw = solar_size_mw * 1000
hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)
hybrid_plant.ppa_price = 0.1
hybrid_plant.simulate(25)


