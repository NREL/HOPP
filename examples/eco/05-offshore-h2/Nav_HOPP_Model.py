"""
Created on Thu Jul  5 11:39:19 2023

@author: nkodanda
"""
#Tasks
#1. Load Met data - solar, wind and wave
#2. Load device and costs data
#3. Calculate total cost
#4. Calculate total energy output- Solar+wind+wave
#5. Use Eco model to calculate H2 requirements, costs and Desal and H2 outputs
#6. Use run_capex & opex functions to calculate LCOE& LCOH

#Use single owner models

"""
#Read Met Data and power curves
"""
#Technologits
data={}
data['lat']=40.498
data['lon']=-70.1
data['year']=2020

#Send 2 locations


#Path
# function: SiteInfo(data,solar_resource_file="",wind_resource_file="", wave_resource_file="", grid_resource_file="",hub_height=97,capacity_hours=[],desired_schedule=[]):)
data['resource']={} #path for resource files

data['resource']['solar']="C:/Users/nkodanda/Downloads/Files/HOPP/examples/eco/05-offshore-h2/input/Resources/blythe_ca_33.617773_-114.588261_psmv3_60_tmy.csv" #solar_resource_file
data['resource']['wave']="C:/Users/nkodanda/Downloads/Files/HOPP/examples/eco/05-offshore-h2/input/Resources/Wave_resource_timeseries_HI.csv" #wave_resource_file
data['resource']['wind']="C:/Users/nkodanda/Downloads/Files/HOPP/examples/eco/05-offshore-h2/input/Resources/28.96_-98_windtoolkit_2013_60min_160m_200m.srw"

import PySAM.PySSC as pssc
ssc = pssc.PySSC()
data2 = ssc.data_create()
ssc.data_set_matrix_from_csv( data2, b'wave_power_matrix', b'C:/Users/nkodanda/Downloads/Files/HOPP/examples/eco/05-offshore-h2/input/Resources/wave_power_matrix.csv');
data['powermatrix']={} #Path for Power curves, matrices
data['powermatrix']['wave']=ssc.data_get_matrix(data2, b'wave_power_matrix')

# Modified offshore data load inputs as needed
import yaml
from yamlinclude import YamlIncludeConstructor 
YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir='C:/Users/nkodanda/Downloads/Files/HOPP/examples/eco/05-offshore-h2/input/floris/')
YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir='C:/Users/nkodanda/Downloads/Files/HOPP/examples/eco/05-offshore-h2/input/turbines/')
#turbine_model="osw_18MW"
#filename_floris_config = "C:/Users/nkodanda/Downloads/Files/HOPP/examples/eco/05-offshore-h2/input/floris/floris_input_iea_18MW_osw.yaml"    #Power Curve
##filename_floris_config2 = "C:/Users/nkodanda/Downloads/Files/HOPP/examples/eco/05-offshore-h2/input/floris/floris_input_iea_18MW_osw.yaml"    #Power Curve

turbine_model="osw_12MW"
filename_floris_config = "C:/Users/nkodanda/Downloads/Files/HOPP/examples/eco/05-offshore-h2/input/floris/floris_input_iea_osw_12MW.yaml"    #Power Curve


"""
#Read device and costs
"""
##Wind farm Layout
#filename_turbine_config = "C:/Users/nkodanda/Downloads/Files/HOPP/examples/eco/05-offshore-h2/input/turbines/"+turbine_model+".yaml" 
filename_turbine_config = "C:/Users/nkodanda/Downloads/Files/HOPP/examples/eco/05-offshore-h2/input/turbines/osw_12MW.yaml" 

#economic parameters for wind turbine & other parameters for H2
filename_orbit_config= "C:/Users/nkodanda/Downloads/Files/HOPP/examples/eco/05-offshore-h2/input/plant/orbit-config-osw_12MW_02.yaml"  


#Wave
#filename_turbine_config2 = "C:/Users/nkodanda/Downloads/Files/HOPP/examples/eco/05-offshore-h2/input/turbines/osw_12MW.yaml" 
#filename_orbit_config2= "C:/Users/nkodanda/Downloads/Files/HOPP/examples/eco/05-offshore-h2/input/plant/orbit-config-osw_12MW_02.yaml"  

#read files
#from PySAM.ResourceTools import SRW_to_wind_data
wind_resource_file={}
wind_resource_file=data['resource']['wind']

import sys
sys.path.insert(0,'C:/Users/nkodanda/Downloads/Files/HOPP')
import hopp.eco.utilities as he_util
# ORBIT imports 
import os
from ORBIT.core.library import initialize_library
initialize_library('C:/Users/nkodanda/Downloads/Files/HOPP/examples/eco/05-offshore-h2/input/')
# Use NREL keys if data is not available
#from hopp.keys import set_developer_nrel_gov_key, get_developer_nrel_gov_key
#global NREL_API_KEY
#NREL_API_KEY = os.getenv("NREL_API_KEY")
#set_developer_nrel_gov_key(NREL_API_KEY)  # Set this key manually here if you are not setting it using the .env or with an env var


verbose=False
show_plots=False
save_plots=True
#Wind
plant_config, turbine_config, wind_resource, floris_config= he_util.get_inputs(filename_orbit_config, filename_turbine_config, wind_resource_file, filename_floris_config, verbose=verbose, show_plots=show_plots, save_plots=save_plots)
#Wave
#plant_config2, turbine_config2, wind_resource2, floris_config2= he_util.get_inputs(filename_orbit_config2, filename_turbine_config2, wind_resource_file, filename_floris_config2, verbose=verbose, show_plots=show_plots, save_plots=save_plots)


# run orbit for wind plant construction and other costs
import hopp.eco.finance as he_fin
orbit_project = he_fin.run_orbit(plant_config, weather=None, verbose=verbose)
#TBD Note: similar aproach can be used for wave device costs using floating wind model
#orbit_project2 = he_fin.run_orbit(plant_config2, weather=None, verbose=verbose)

#Wave costs $/MW
if plant_config['wave']['flag']:
    mhk_config=plant_config['wave']
    cost_model_inputs=plant_config['WaveCost']

    from hopp.mhk_wave_source import MHKCosts
    mhk_cost_results=MHKCosts(mhk_config, cost_model_inputs)
    mhk_cost_results.simulate_costs()
    WEC_Cost={}
    WEC_Cost['capex'] = (mhk_cost_results.cost_outputs['structural_assembly_cost_modeled'] + mhk_cost_results.cost_outputs['power_takeoff_system_cost_modeled'] \
        + mhk_cost_results.cost_outputs['mooring_found_substruc_cost_modeled'])
    WEC_Cost['bos'] = (mhk_cost_results.cost_outputs['development_cost_modeled'] + mhk_cost_results.cost_outputs['eng_and_mgmt_cost_modeled'] \
        + mhk_cost_results.cost_outputs['plant_commissioning_cost_modeled'] +mhk_cost_results.cost_outputs['site_access_port_staging_cost_modeled'] \
            + mhk_cost_results.cost_outputs['assembly_and_install_cost_modeled'] + mhk_cost_results.cost_outputs['other_infrastructure_cost_modeled'])
    WEC_Cost['elec_infrastruc_costs'] = (mhk_cost_results.cost_outputs['array_cable_system_cost_modeled'] + mhk_cost_results.cost_outputs['export_cable_system_cost_modeled'] \
        + mhk_cost_results.cost_outputs['onshore_substation_cost_modeled'] + mhk_cost_results.cost_outputs['offshore_substation_cost_modeled'] \
            + mhk_cost_results.cost_outputs['other_elec_infra_cost_modeled'])
    WEC_Cost['financial_costs'] = (mhk_cost_results.cost_outputs['project_contingency'] + mhk_cost_results.cost_outputs['insurance_during_construction'] \
        + mhk_cost_results.cost_outputs['reserve_accounts'])
        
    WEC_Cost['opex'] = (mhk_cost_results.cost_outputs['maintenance_cost'] + mhk_cost_results.cost_outputs['operations_cost'])
    
    WEC_Cost['total_installed_cost'] = WEC_Cost['capex'] + WEC_Cost['bos'] + WEC_Cost['elec_infrastruc_costs'] 
else:
    WEC_Cost['total_installed_cost'] = 0
    WEC_Cost['opex']= 0
    WEC_Cost['bos'] = 0


# setup HOPP model
import hopp.eco.hopp_mgmt as he_hopp
hopp_site, hopp_technologies, hopp_scenario, hopp_h2_args = he_hopp.setup_hopp(plant_config, turbine_config, wind_resource, orbit_project, floris_config, WEC_Cost, solar_resource_file=data['resource']['solar'],wave_resource_file=data['resource']['wave'],Wave_PowerMatrix=data['powermatrix']['wave'],show_plots=show_plots, save_plots=save_plots)

##### BOS to be added & O&M for wave
hopp_results = he_hopp.run_hopp(hopp_site, hopp_technologies, hopp_scenario, hopp_h2_args, verbose=verbose)





import numpy as np
HRly_Op=hopp_results["combined_pv_wind_power_production_hopp"]

HRly_Op2=np.asarray(hopp_results["combined_pv_wind_power_production_hopp"]) #Combined power
# Hydrogen Model calculate outputs given power input


import hopp.eco.electrolyzer as he_elec
import hopp.eco.finance as he_fin
import hopp.eco.hopp_mgmt as he_hopp
import hopp.eco.hydrogen_mgmt as he_h2
import matplotlib.pyplot as plt 




storage_types = ["pressure_vessel"]
scenarios = [0]
verbose=False
show_plots=False
save_plots=True
use_profast=True
incentive_option=1
electrolyzer_rating=None
plant_size=None
storage_type=None
incentive_option=1
plant_design_scenario=0
output_level=1
grid_connection=None


if electrolyzer_rating != None:
    plant_config["electrolyzer"]["rating"] = electrolyzer_rating

if grid_connection != None:
    plant_config["project_parameters"]["grid_connection"] = grid_connection

if storage_type != None:
    plant_config["h2_storage"]["type"] = storage_type

if plant_size != None:
    plant_config["plant"]["capacity"] = plant_size
    plant_config["plant"]["num_turbines"] = int(plant_size/turbine_config["turbine_rating"])
    print(plant_config["plant"]["num_turbines"])

design_scenario = plant_config["plant_design"]["scenario%s" %(plant_design_scenario)]
design_scenario["id"] = plant_design_scenario



def energy_internals(HRly_Op=HRly_Op, hopp_results=hopp_results, hopp_scenario=hopp_scenario, 
                     hopp_h2_args=hopp_h2_args, orbit_project=orbit_project, 
                     design_scenario=design_scenario, plant_config=plant_config, 
                     turbine_config=turbine_config, wind_resource=wind_resource, 
                     electrolyzer_rating=electrolyzer_rating, verbose=verbose, 
                     show_plots=show_plots, save_plots=save_plots, 
                     use_profast=use_profast, solver=True, 
                     power_for_peripherals_kw_in=0.0, breakdown=False):

    #hopp_results_internal = dict(hopp_results)
    HRly_Op3=HRly_Op

    # set energy input profile
    ### subtract peripheral power from supply to get what is left for electrolyzer
    remaining_power_profile_in = np.zeros_like(HRly_Op)

    high_count = sum(np.asarray(HRly_Op) >= power_for_peripherals_kw_in)
    total_peripheral_energy = power_for_peripherals_kw_in*365*24
    distributed_peripheral_power = total_peripheral_energy/high_count
    for i in range(len(HRly_Op)):
        r = HRly_Op[i] - distributed_peripheral_power
        if r > 0:
            remaining_power_profile_in[i] = r

    HRly_Op3 = tuple(remaining_power_profile_in)
    
    # run electrolyzer physics model
    electrolyzer_physics_results = he_elec.run_electrolyzer_physics(hopp_results, HRly_Op3, hopp_scenario, hopp_h2_args, plant_config, wind_resource, design_scenario, show_plots=show_plots, save_plots=save_plots, verbose=verbose)

    # run electrolyzer cost model
    electrolyzer_cost_results = he_elec.run_electrolyzer_cost(electrolyzer_physics_results, hopp_scenario, plant_config, design_scenario, verbose=verbose)
    
    desal_results = he_elec.run_desal(plant_config, electrolyzer_physics_results, design_scenario, verbose)

    # run array system model
    h2_pipe_array_results = he_h2.run_h2_pipe_array(plant_config, orbit_project, electrolyzer_physics_results, design_scenario, verbose)

    # compressor #TODO size correctly
    h2_transport_compressor, h2_transport_compressor_results = he_h2.run_h2_transport_compressor(plant_config, electrolyzer_physics_results, design_scenario, verbose=verbose)

    # transport pipeline
    h2_transport_pipe_results = he_h2.run_h2_transport_pipe(plant_config, electrolyzer_physics_results, design_scenario, verbose=verbose)

    # pressure vessel storage
    pipe_storage, h2_storage_results = he_h2.run_h2_storage(plant_config, turbine_config, electrolyzer_physics_results, design_scenario, verbose=verbose)
    
    total_energy_available = np.sum(HRly_Op)
    
    ### get all energy non-electrolyzer usage in kw
    desal_power_kw = desal_results["power_for_desal_kw"]

    h2_transport_compressor_power_kw = h2_transport_compressor_results["compressor_power"] # kW

    h2_storage_energy_kwh = h2_storage_results["storage_energy"] 
    h2_storage_power_kw = h2_storage_energy_kwh*(1.0/(365*24))
    
    # if transport is not HVDC and h2 storage is on shore, then power the storage from the grid
    if (design_scenario["transportation"] == "pipeline") and (design_scenario["h2_storage_location"] == "onshore"):
        total_accessory_power_renewable_kw = desal_power_kw + h2_transport_compressor_power_kw
        total_accessory_power_grid_kw = h2_storage_power_kw
    else:
        total_accessory_power_renewable_kw = desal_power_kw + h2_transport_compressor_power_kw + h2_storage_power_kw
        total_accessory_power_grid_kw = 0.0

    ### subtract peripheral power from supply to get what is left for electrolyzer and also get grid power
    remaining_power_profile = np.zeros_like(HRly_Op)
    grid_power_profile = np.zeros_like(HRly_Op)
    for i in range(len(HRly_Op)):
        r = HRly_Op[i] - total_accessory_power_renewable_kw
        grid_power_profile[i] = total_accessory_power_grid_kw
        if r > 0:
            remaining_power_profile[i] = r

    if verbose and not solver:
        print("\nEnergy/Power Results:")
        print("Supply (MWh): ", total_energy_available)
        print("Desal (kW): ", desal_power_kw)
        print("Transport compressor (kW): ", h2_transport_compressor_power_kw)
        print("Storage compression, refrigeration, etc (kW): ", h2_storage_power_kw)

    if (show_plots or save_plots) and not solver:
        fig, ax = plt.subplots(1)
        plt.plot(np.asarray(HRly_Op)*1E-6, label="Total Energy Available")
        plt.plot(remaining_power_profile*1E-6, label="Energy Available for Electrolysis")
        plt.xlabel("Hour")
        plt.ylabel("Power (GW)")
        plt.tight_layout()
        if save_plots:
            savepath = "figures/power_series/"
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            plt.savefig(savepath+"power_%i.png" %(design_scenario["id"]), transparent=True)
        if show_plots:
            plt.show()
    if solver:
        if breakdown:
            return total_accessory_power_renewable_kw, total_accessory_power_grid_kw, desal_power_kw, h2_transport_compressor_power_kw, h2_storage_power_kw
        else:
            return total_accessory_power_renewable_kw
    else:
        return electrolyzer_physics_results, electrolyzer_cost_results, desal_results, h2_pipe_array_results, h2_transport_compressor, h2_transport_compressor_results, h2_transport_pipe_results, pipe_storage, h2_storage_results, total_accessory_power_renewable_kw, total_accessory_power_grid_kw


def energy_residual_function(power_for_peripherals_kw_in):

    # get results for current design
    # print("power peri in: ", power_for_peripherals_kw_in)
    power_for_peripherals_kw_out = energy_internals(power_for_peripherals_kw_in=power_for_peripherals_kw_in, solver=True, verbose=False)

    # collect residual
    power_residual = power_for_peripherals_kw_out - power_for_peripherals_kw_in
    # print("\nresidual: ", power_residual)

    return power_residual



def simple_solver(initial_guess=0.0):

    # get results for current design
    total_accessory_power_renewable_kw, total_accessory_power_grid_kw, desal_power_kw, h2_transport_compressor_power_kw, h2_storage_power_kw = energy_internals(power_for_peripherals_kw_in=initial_guess, solver=True, verbose=False, breakdown=True)
    
    return total_accessory_power_renewable_kw, total_accessory_power_grid_kw, desal_power_kw, h2_transport_compressor_power_kw, h2_storage_power_kw


solver_results = simple_solver(0)
solver_result = solver_results[0]
# get results for final design
electrolyzer_physics_results, electrolyzer_cost_results, desal_results, h2_pipe_array_results, h2_transport_compressor, h2_transport_compressor_results, h2_transport_pipe_results, pipe_storage, h2_storage_results, total_accessory_power_renewable_kw, total_accessory_power_grid_kw \
    = energy_internals(solver=False, power_for_peripherals_kw_in=solver_result)

# Cost models

platform_results = he_h2.run_equipment_platform(plant_config, design_scenario, electrolyzer_physics_results, h2_storage_results, desal_results, verbose=verbose)
capex, capex_breakdown = he_fin.run_capex(hopp_results, orbit_project, electrolyzer_cost_results, h2_pipe_array_results, h2_transport_compressor_results, h2_transport_pipe_results, h2_storage_results, plant_config, design_scenario, desal_results, platform_results, verbose=verbose)
opex_annual, opex_breakdown_annual = he_fin.run_opex(hopp_results, orbit_project, electrolyzer_cost_results, h2_pipe_array_results, h2_transport_compressor_results, h2_transport_pipe_results, h2_storage_results, plant_config, desal_results, platform_results, verbose=verbose, total_export_system_cost=capex_breakdown["electrical_export_system"])


if use_profast:
    lcoe, pf_lcoe = he_fin.run_profast_lcoe(plant_config, orbit_project, capex_breakdown, opex_breakdown_annual, hopp_results, design_scenario, verbose=verbose, show_plots=show_plots, save_plots=save_plots)    
    lcoh_grid_only, pf_grid_only = he_fin.run_profast_grid_only(plant_config, orbit_project, electrolyzer_physics_results, capex_breakdown, opex_breakdown_annual, hopp_results, design_scenario, total_accessory_power_renewable_kw, total_accessory_power_grid_kw, verbose=verbose, show_plots=show_plots, save_plots=save_plots)
    lcoh, pf_lcoh = he_fin.run_profast_full_plant_model(plant_config, orbit_project, electrolyzer_physics_results, capex_breakdown, opex_breakdown_annual, hopp_results, incentive_option, design_scenario, total_accessory_power_renewable_kw, total_accessory_power_grid_kw, verbose=verbose, show_plots=show_plots, save_plots=save_plots)

power_breakdown = he_util.post_process_simulation(lcoe, lcoh, pf_lcoh, pf_lcoe, hopp_results, electrolyzer_physics_results, plant_config, h2_storage_results, capex_breakdown, opex_breakdown_annual, orbit_project, platform_results, desal_results, design_scenario, plant_design_scenario, incentive_option, solver_results=solver_results, show_plots=show_plots, save_plots=save_plots)#, lcoe, lcoh, lcoh_with_grid, lcoh_grid_only)
