from pathlib import Path
from numpy import genfromtxt
import pandas as pd
from hybrid.sites import SiteInfo, flatirons_site
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.dispatch.plot_tools import plot_battery_output, plot_battery_dispatch_error, plot_generation_profile
from hybrid.keys import set_nrel_key_dot_env 
from examples.yaw_pitch_comp import yaw_pitch_comp
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
curve_data = pd.read_csv(examples_dir.parent / "examples"/"H2_Analysis" / "NREL_Reference_1.5MW_Turbine_Sea_Level.csv")
wind_speed = curve_data['Wind Speed [m/s]'].values.tolist() 
curve_power = curve_data['Power [kW]']
battery_costs = False

technologies = {
    'pv': {
        'system_capacity_kw': solar_size_mw * 1000,
    }, 
    'wind': {
        'num_turbines': 1,
        'turbine_rating_kw': wind_size_mw * 1000,
        'hub_height': hub_height,
        'rotor_diameter': rotor_diameter
    # },
    # 'battery': {
    #     'system_capacity_kwh': battery_capacity_mw * 1000,
    #     'system_capacity_kw': battery_capacity_mwh  * 1000
    }
}

# Get resource
lat = 39.91#flatirons_site['lat']
lon = -105.22#flatirons_site['lon']
flatirons_site['elev'] = 1855
flatirons_site['year'] = 2020
price_file = examples_dir.parent / "resource_files" / "grid" / "constant_nom_prices.csv"
wind_resource_file = examples_dir.parent / "examples" / "resource_files" / "yearlong_hopp_validation_wind.srw"
solar_resource_file = examples_dir.parent / "examples" / "resource_files" / "yearlong_hopp_validation_solar.csv"
load_profile = genfromtxt(examples_dir.parent / "resource_files" / "grid" / "yearlong_hopp_validation_load.csv", delimiter=",",skip_header=1)

site = SiteInfo(flatirons_site,
                solar_resource_file = solar_resource_file,
                wind_resource_file = wind_resource_file, 
                hub_height = 80,
                grid_resource_file = price_file, 
                desired_schedule = load_profile)

# # Uncomment for simple pitch/yaw correction - doesn't appear to being doing much good right now 

# minutely_wind_dir_filepath = str(examples_dir) + '/resource_files/' + 'yearlong_hopp_wind_dir_deg.csv'
# hourly_yaw_pitch_filepath = str(examples_dir) + '/resource_files/' + 'yearlong_hopp_GE15_pitch_yaw_deg.csv'
# site, pitch, yaw_misalignment = yaw_pitch_comp(site, minutely_wind_dir_filepath, hourly_yaw_pitch_filepath)
# pitch_yaw_misalignment = pd.DataFrame([pitch,yaw_misalignment],index=['Pitch','Yaw Misalignment'])
# pitch_yaw_misalignment.to_json(str(examples_dir) + '/results/' + 'yearlong_wind_misalignment.json')

# Create base model
hybrid_plant = HybridSimulation(technologies,
                                 site, 
                                 interconnect_kw=interconnection_size_mw * 1000,
                                 dispatch_options={
                                    # 'is_test_start_year' : True,
                                    'solver': 'cbc',
                                    'n_look_ahead_periods': 48, #hrs
                                    # 'grid_charging': True,
                                    'pv_charging_only': False,
                                    'include_lifecycle_count': False})

# PV
hybrid_plant.pv.system_capacity_kw = solar_size_mw * 1000
hybrid_plant.pv.dc_degradation = (0,)             # year over year degradation
hybrid_plant.pv.value('array_type',0)
hybrid_plant.pv.value('tilt',25)
hybrid_plant.pv.value('dc_ac_ratio',1.149)
hybrid_plant.pv.value('losses',1.006)

# Wind
hybrid_plant.wind.wake_model = 3                # constant wake loss, layout-independent
hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)
hybrid_plant.wind._system_model.Turbine.wind_turbine_powercurve_windspeeds = wind_speed
hybrid_plant.wind._system_model.Turbine.wind_turbine_powercurve_powerout = curve_power
hybrid_plant.wind._system_model.Turbine.wind_resource_shear = 0.15 #Wind Shear Exponent https://www.engineeringtoolbox.com/wind-shear-d_1215.html
hybrid_plant.wind.value("avail_bop_loss", 0)
hybrid_plant.wind.value("avail_grid_loss", 0)
hybrid_plant.wind.value("avail_turb_loss", 0)
hybrid_plant.wind.value("elec_eff_loss", 0)
hybrid_plant.wind.value("elec_parasitic_loss", 0)
hybrid_plant.wind.value("env_degrad_loss", 0)
hybrid_plant.wind.value("env_env_loss", 0)
hybrid_plant.wind.value("env_icing_loss", 0)
hybrid_plant.wind.value("ops_env_loss", 0)
hybrid_plant.wind.value("ops_grid_loss", 0)
hybrid_plant.wind.value("ops_load_loss", 0)
hybrid_plant.wind.value("turb_generic_loss", 0)
hybrid_plant.wind.value("turb_hysteresis_loss", 0)
hybrid_plant.wind.value("turb_perf_loss", 0)
hybrid_plant.wind.value("turb_specific_loss", 9.886)
hybrid_plant.wind.value("wake_ext_loss", 0)  

# # Battery
# hybrid_plant.battery._system_model.value("minimum_SOC", 20.0)
# hybrid_plant.battery._system_model.value("maximum_SOC", 100.0)
# hybrid_plant.battery._system_model.value("initial_SOC", 90.0)

'''
Financial modifications to change cost minimizing objective function into purely a load following objective function
Will attempt to meet load at all costs
Sets O&M of technologies and lifecycle cost for battery to 0
Objective funcion uses price file and ppa price which in this case are set to 1, thus all values are representative of
technologies ability to meet desired load
aka difference in generation vs load
'''
# prices_file are unitless dispatch factors, so add $/kwh here
#Set as 1 for objective function cost minimization (price becomes negligible and only a load vs generation calculation)
# Set as $0.1/kwh
hybrid_plant.ppa_price = .1  
hybrid_plant.pv.value('om_capacity', (0.0,))
hybrid_plant.wind.value('om_capacity', (0.0,))
# if battery_costs:
#     hybrid_plant.battery.value('om_batt_capacity_cost', (17.0,)) # Capacity-based O&M amount [$/kWcap]
# else:
#     hybrid_plant.battery.value('om_batt_capacity_cost', (0.0,))



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
# plot_battery_dispatch_error(hybrid_plant)
# plot_battery_output(hybrid_plant)
# plot_generation_profile(hybrid_plant)
# plot_generation_profile(hybrid_plant, 150, 2)
#plot_battery_dispatch_error(hybrid_plant, plot_filename=tag+'battery_dispatch_error.png')

load_kw = [p * 1000 for p in list(load)]
power_scale = 1000
# discharge = [(p > 0) * p*power_scale for p in hybrid_plant.battery.Outputs.dispatch_P]
# charge = [(p < 0) * p*power_scale for p in hybrid_plant.battery.Outputs.dispatch_P]

original_gen = [(w+s) for w, s in zip(list(hybrid_plant.wind.generation_profile),
                                                        list(hybrid_plant.pv.generation_profile))]
gen = [p for p in list(hybrid_plant.grid.generation_profile)]

# grid_supplied = [load - dispatch for (load, dispatch) in zip(list(load_kw),
#                                                                 list(hybrid_plant.grid.generation_profile))]

outputs = pd.DataFrame(
            {'pv generation (kW)': hybrid_plant.pv.generation_profile,
            'wind generation (kW)': hybrid_plant.wind.generation_profile,
            # 'dispatch battery SOC (%)': hybrid_plant.battery.Outputs.dispatch_SOC,
            # 'original battery SOC (%)': hybrid_plant.battery.Outputs.SOC,
            'load (kW)': load_kw,
            # 'battery charge (kW)': charge,
            # 'battery discharge (kW)': discharge,
            'original hybrid generation (kW)': original_gen,
            # 'optimized dispatch (kW)': gen,
            # 'grid supplied load (kW)': grid_supplied
            # 'plant curtailment (kW)': combined_pv_wind_curtailment_hopp,
            # 'plant shortfall (kW)': energy_shortfall_hopp
            })

outputs.to_csv(str(examples_dir) + '/results/' + 'yearlong_outputs_no_batt_m2.csv')
outputs.to_json(str(examples_dir) + '/results/' + 'yearlong_outputs_no_batt_m2.json')