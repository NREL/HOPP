import pandas as pd
import csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from numpy import genfromtxt
from hybrid.sites import SiteInfo, flatirons_site
from hybrid.hybrid_simulation import HybridSimulation
from examples.H2_Analysis.hopp_for_h2 import hopp_for_h2
from examples.H2_Analysis.simple_dispatch import SimpleDispatch
from hybrid.dispatch.plot_tools import plot_battery_output, plot_battery_dispatch_error, plot_generation_profile
from hybrid.keys import set_nrel_key_dot_env

# Set API key
set_nrel_key_dot_env()
examples_dir = Path(__file__).parent.absolute()

plot_power_production = True
plot_battery = True
plot_hopp_homer_validation = True

# Set technologies
solar_size_mw = 0.430

wind_size_mw = 1.5
hub_height = 80
rotor_diameter = 77
custom_powercurve = True
curve_data = 'NREL_Reference_1.5MW_Turbine.csv'

battery_capacity_mw = 1
battery_capacity_mwh = 1
battery_duration_hrs = 1
interconnection_size_mw = 2

#Set scenario
scenario = dict()
atb_year = 2025
ptc_avail = 'no'
itc_avail = 'no'
discount_rate = 0.07
forced_sizes = True
force_electrolyzer_cost = True
electrolyzer_size = 2
kw_continuous = electrolyzer_size * 1000
wind_cost_kw = 9999
solar_cost_kw = 9999
storage_cost_kw = 250
storage_cost_kwh = 240
debt_equity_split = 60
scenario['Useful Life'] = 1
scenario['PTC Available'] = 'no'
scenario['ITC Available'] = 'no'
scenario['Debt Equity'] = 60
scenario['Discount Rate'] = 0.07
scenario['Tower Height'] = 80
scenario['Powercurve File'] = curve_data


# Add load data
load = genfromtxt(examples_dir.parent / "resource_files" / "grid" / "yearlong_hopp_validation_load.csv", delimiter=",",skip_header=1) *1000

technologies = {'pv':
                    {'system_capacity_kw': solar_size_mw * 1000},
                'wind':
                    {'num_turbines': 1,
                        'turbine_rating_kw': wind_size_mw*1000,
                        'hub_height': hub_height,
                        'rotor_diameter': rotor_diameter},
                'battery': {
                    'system_capacity_kwh': battery_capacity_mwh * 1000,
                    'system_capacity_kw': battery_capacity_mw * 1000
                    }
                }
# Set site
lat = flatirons_site['lat']
lon = flatirons_site['lon']
flatirons_site['elev'] = 1855
flatirons_site['year'] = 2021
wind_resource_file = examples_dir.parent / "resource_files" / "wind" / "yearlong_hopp_validation_wind_winddir.srw"
solar_resource_file = examples_dir.parent / "resource_files" / "solar" / "yearlong_hopp_validation_solar.csv"

site = SiteInfo(flatirons_site, 
                solar_resource_file = solar_resource_file, 
                wind_resource_file= wind_resource_file, 
                hub_height=80)

# Ability to sell/purchase electricity to/from the grid. Price defined in $/kWh
sell_price = False
buy_price = False

# Run HOPP
hybrid_plant, combined_pv_wind_power_production_hopp, combined_pv_wind_curtailment_hopp,\
energy_shortfall_hopp, gen_profile, annual_energies, wind_plus_solar_npv, npvs, lcoe =  \
    hopp_for_h2(site, scenario, technologies,
                wind_size_mw, solar_size_mw, battery_capacity_mw, battery_capacity_mwh, battery_duration_hrs,
    wind_cost_kw, solar_cost_kw, storage_cost_kw, storage_cost_kwh,
    kw_continuous, load,
    custom_powercurve,
    electrolyzer_size, grid_connected_hopp=True)


if plot_power_production:
    plt.figure(figsize=(10,4))
    plt.title("HOPP power production")
    plt.plot(combined_pv_wind_power_production_hopp[6816:6960],label="wind + pv")
    plt.plot(energy_shortfall_hopp[6816:6960],label="shortfall")
    plt.plot(combined_pv_wind_curtailment_hopp[6816:6960],label="curtailment")
    plt.plot(load[6816:6960],label="load")
    plt.xlabel("Time (hour)")
    plt.ylabel("Power Production (kW)")
    # plt.ylim(0,250000)
    plt.legend()
    plt.tight_layout()
    plt.show()

bat_model = SimpleDispatch()
bat_model.Nt = len(energy_shortfall_hopp)
bat_model.curtailment = np.divide(combined_pv_wind_curtailment_hopp, 1000)
bat_model.shortfall = np.divide(energy_shortfall_hopp, 1000)
bat_model.battery_storage = battery_capacity_mwh
bat_model.charge_rate = battery_capacity_mw
bat_model.discharge_rate = battery_capacity_mw
bat_model.initial_SOC = .2
bat_model.max_SOC = 1
bat_model.min_SOC = .2

battery_used, excess_energy, battery_SOC = bat_model.run()
print('Annual Energy w/o battery dispatch [kWh]: ', annual_energies)
print('Annual battery use [kWh]: ', np.sum(battery_used*1000))
combined_pv_wind_storage_power_production_hopp = combined_pv_wind_power_production_hopp + (battery_used*1000)
battery_used_kW = battery_used *1000
print(np.sum(combined_pv_wind_power_production_hopp), np.sum(combined_pv_wind_storage_power_production_hopp))
print('Minimum Battery SOC: ', np.min(battery_SOC))
pv_generation = gen_profile.pv
wind_generation = gen_profile.wind

if plot_battery:
    plt.figure(figsize=(9,6))
    plt.subplot(311)
    plt.plot(combined_pv_wind_curtailment_hopp[6816:6960],label="curtailment")
    plt.plot(energy_shortfall_hopp[6816:6960],label="shortfall")
    plt.ylabel('Power (kW)')
    plt.title('Energy Curtailment and Shortfall')
    plt.legend()

    plt.subplot(312)
    plt.plot(combined_pv_wind_storage_power_production_hopp[6816:6960],label="wind+pv+storage")
    plt.plot(combined_pv_wind_power_production_hopp[6816:6960],"--",label="wind+pv")
    plt.plot(load[6816:6960],"--",label="load")
    plt.legend()
    plt.ylabel('Power (kW)')
    plt.title("Hybrid Plant Power Flows with and without storage")
    plt.tight_layout()
    
    plt.subplot(313)
    plt.plot(battery_SOC[6816:6960],label="state of charge")
    plt.plot(battery_used[6816:6960],"--",label="battery used")
    plt.title('Battery State')
    plt.legend()
    # plt.savefig(os.path.join(results_dir,'HOPP Full Power Flows_{}_{}_{}'.format(site_name,atb_year,ptc_avail)),bbox_inches='tight')
    plt.show()

if plot_hopp_homer_validation:
    plt.figure(figsize=(12,6))
    plt.plot(pv_generation[0:120], label="pv")
    plt.plot(wind_generation[0:120], label = "wind")
    plt.plot(battery_used_kW[0:120],"--",label="battery used")
    plt.plot(load[0:120],label="load")
    plt.xlabel("Time (hour)")
    plt.ylabel("Power Production (kW)")
    plt.legend()
    plt.show()

plt.figure(figsize=(10,4))
plt.plot(site.solar_resource.data['gh'][0:72], label = "GHI")
plt.plot(site.solar_resource.data['df'][0:72], label = "DHI")
plt.plot(site.solar_resource.data['dn'][0:72], label = "DNI")
plt.xlabel("Time (hour)")
plt.ylabel("Irradiance (w/m^2)")
plt.title("Solar Resource Data (NSRDB)")
plt.legend()
plt.show()

outputs = pd.DataFrame(
            {'pv generation (kW)': pv_generation,
            'wind generation (kW)': wind_generation,
            'battery used (kW)': battery_used_kW,
            'battery SOC (%)': battery_SOC,
            'load (kW)': load,
            'plant curtailment (kW)': combined_pv_wind_curtailment_hopp,
            'plant shortfall (kW)': energy_shortfall_hopp})

