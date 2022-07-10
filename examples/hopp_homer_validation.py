from pathlib import Path
from numpy import genfromtxt
import pandas as pd
from hybrid.sites import SiteInfo, flatirons_site
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.dispatch.plot_tools import plot_battery_output, plot_battery_dispatch_error, plot_generation_profile
from hybrid.keys import set_nrel_key_dot_env
# Set API key
set_nrel_key_dot_env()

examples_dir = Path(__file__).parent.absolute()

solar_size_mw = 0.430
wind_size_mw = 1.5
battery_capacity_mw = 1
battery_capacity_mwh = 1
interconnection_size_mw = 2
hub_height = 80
rotor_diameter = 77
curve_data = pd.read_csv(examples_dir.parent / "examples"/"H2_Analysis" / "NREL_Reference_1.5MW_Turbine.csv")
wind_speed = curve_data['Wind Speed [m/s]'].values.tolist() 
curve_power = curve_data['Power [kW]']

technologies = {
    'pv': {
        'system_capacity_kw': solar_size_mw * 1000,
    }, 
    'wind': {
        'num_turbines': 1,
        'turbine_rating_kw': wind_size_mw * 1000,
        'hub_height': hub_height,
        'rotor_diameter': rotor_diameter
    },
    'battery': {
        'system_capacity_kwh': battery_capacity_mw * 1000,
        'system_capacity_kw': battery_capacity_mwh  * 1000
    }
}

# Get resource
lat = flatirons_site['lat']
lon = flatirons_site['lon']
flatirons_site['year'] = 2020
price_file = examples_dir.parent / "resource_files" / "grid" / "constant_nom_prices.csv"
wind_resource_file = examples_dir.parent / "resource_files" / "wind" / "hopp_validation_wind_data_hourly.srw"
solar_resource_file = examples_dir.parent / "resource_files" / "solar" / "564277_35.21_-101.94_2020_hopp_validation.csv"
load_profile = genfromtxt(examples_dir.parent / "resource_files" / "grid" / "hopp_validation_load_hourly_MW.csv", delimiter=",")

site = SiteInfo(flatirons_site,
                solar_resource_file = solar_resource_file,
                wind_resource_file = wind_resource_file, 
                hub_height = 80,
                grid_resource_file = price_file, 
                desired_schedule = load_profile)


# Create base model
hybrid_plant = HybridSimulation(technologies,
                                 site, 
                                 interconnect_kw=interconnection_size_mw * 1000,
                                 dispatch_options={
                                    'is_test_start_year' : True,
                                    'solver': 'cbc',
                                    'n_look_ahead_periods': 48, #hrs
                                    'grid_charging': False,
                                    'pv_charging_only': False,
                                    'include_lifecycle_count': False})

# PV
hybrid_plant.pv.system_capacity_kw = solar_size_mw * 1000
hybrid_plant.pv.dc_degradation = (0,)             # year over year degradation

# Wind
hybrid_plant.wind.wake_model = 3                # constant wake loss, layout-independent
hybrid_plant.wind.value("wake_int_loss", 1)     # percent wake loss
hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)
hybrid_plant.wind._system_model.Turbine.wind_turbine_powercurve_windspeeds = wind_speed
hybrid_plant.wind._system_model.Turbine.wind_turbine_powercurve_powerout = curve_power
hybrid_plant.wind._system_model.Turbine.wind_resource_shear = 0.3                           #TODO: validate that this is equivalent to HOMER "wind shear: logarithmic w/ roughness length = 0.3m"

# Battery
hybrid_plant.battery._system_model.value("minimum_SOC", 20.0)
hybrid_plant.battery._system_model.value("maximum_SOC", 100.0)
hybrid_plant.battery._system_model.value("initial_SOC", 90.0)

'''
Financial modifications to change cost minimizing objective function into purely a load following objective function
Will attempt to meet load at all costs
Sets O&M of technologies and lifecycle cost for battery to 0
Objective funcion uses price file and ppa price which in this case are set to 1, thus all values are representative of
technologies ability to meet desired load
aka difference in generation vs load
'''
# prices_file are unitless dispatch factors, so add $/kwh here
hybrid_plant.ppa_price = 1  #Set as 1 for objective function cost minimization (price becomes negligible and only a load vs generation calculation)
hybrid_plant.pv.value('om_capacity', (0.0,))
hybrid_plant.wind.value('om_capacity', (0.0,))
hybrid_plant.battery.value('om_capacity', (0.0,))

# use single year for now, multiple years with battery not implemented yet
hybrid_plant.simulate(project_life=1)

print("output after losses over gross output",
      hybrid_plant.wind.value("annual_energy") / hybrid_plant.wind.value("annual_gross_energy"))

# Save the outputs
annual_energies = hybrid_plant.annual_energies
npvs = hybrid_plant.net_present_values
revs = hybrid_plant.total_revenues
load = hybrid_plant.site.desired_schedule
print(annual_energies)
print(npvs)
print(revs)


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
plot_generation_profile(hybrid_plant, 150, 2)
#plot_battery_dispatch_error(hybrid_plant, plot_filename=tag+'battery_dispatch_error.png')
