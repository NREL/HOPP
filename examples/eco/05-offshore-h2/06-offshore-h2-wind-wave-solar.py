# Tasks
# 1. Load data - solar, wind and wave
# 2. Load device and costs data
# 3. Calculate total cost
# 4. Calculate total energy output- Solar+wind+wave
# 5. Use Eco model to calculate H2 requirements, costs and Desal and H2 outputs
# 6. Use run_capex & opex functions to calculate LCOE& LCOH
# General financial tools in main branch which is not available in current branch.
# Use single owner models
import PySAM.PySSC as pssc
import yaml
from yamlinclude import YamlIncludeConstructor 
import numpy as np
import pandas as pd
import csv 
import os
import sys
sys.path.insert(0,'C:/Users/nkodanda/Downloads/Files/HOPP')
import hopp.eco.utilities as he_util
import hopp.eco.finance as he_fin
import hopp.eco.electrolyzer as he_elec
import hopp.eco.finance as he_fin
import hopp.eco.hopp_mgmt as he_hopp
import hopp.eco.hydrogen_mgmt as he_h2
import hopp.eco.hopp_mgmt as he_hopp
import matplotlib.pyplot as plt 

from ORBIT.core.library import initialize_library

"""
#Read Met Data
"""
#Path
res_path="C:/Users/nkodanda/Downloads/Files/HOPP/examples/eco/05-offshore-h2/input/Resources/Oahu_West/" #Oahu_North, Oahu_South,Oahu_West
#res_path=os.path.dirname(os.path.abspath(__file__))+"/input/Resources/Oahu_West/" #Oahu_North, Oahu_South,Oahu_West

data={}
data['resource']={} #path for resource files
data['resource']['solar']=res_path+"solar_resource.csv" #solar_resource_file
data['resource']['wave']=res_path+"wave_resource.csv"#"C:/Users/nkodanda/Downloads/Files/HOPP/examples/eco/05-offshore-h2/input/Resources/Wave_resource_timeseries_HI.csv" #wave_resource_file
data['resource']['wind']=res_path+"wind_resource.srw"#"C:/Users/nkodanda/Downloads/Files/HOPP/examples/eco/05-offshore-h2/input/Resources/28.96_-98_windtoolkit_2013_60min_160m_200m.srw"

"""
#Read power curves or matrices
"""
# wave
ssc = pssc.PySSC()
data2 = ssc.data_create()
ssc.data_set_matrix_from_csv( data2, b'wave_power_matrix', os.path.abspath(os.path.join(res_path,".."))+'/RM3_wave_power_matrix.csv');
data['powermatrix']={} #Path for Power curves, matrices
data['powermatrix']['wave']=ssc.data_get_matrix(data2, b'wave_power_matrix')

# Modified offshore data load inputs as needed
YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=res_path+'floris/')
YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=res_path+'turbines/')
turbine_model="osw_12MW"
filename_floris_config = res_path+"floris/floris_input_iea_osw_12MW.yaml"    #Power Curve
#Wind farm Layout
filename_turbine_config = res_path+"turbines/osw_12MW.yaml" 
#economic parameters for wind turbine & other parameters for H2
filename_orbit_config= res_path+"plant/orbit-config-osw_12MW_02.yaml"  
wind_resource_file={}
wind_resource_file=data['resource']['wind']

"""
#Configure wind farm and get inputs for H2, wave, solar
"""
initialize_library(res_path)
verbose=False
show_plots=True
save_plots=True

#Wind
plant_config, turbine_config, wind_resource, floris_config= he_util.get_inputs(filename_orbit_config, filename_turbine_config, wind_resource_file, filename_floris_config, verbose=verbose, show_plots=show_plots, save_plots=save_plots)
# run orbit for wind plant construction and other costs
orbit_project = he_fin.run_orbit(plant_config, weather=None, verbose=verbose)

#Wave
WEC_Cost={}
#Wave costs $/MW
if plant_config['wave']['flag']:
    
    plant_config['WaveCost']['water_depth']=plant_config['site']['depth'] 
    mhk_config=plant_config['wave']
    cost_model_inputs=plant_config['WaveCost']

    from hopp.mhk_wave_source import MHKCosts
    mhk_cost_results=MHKCosts(mhk_config, cost_model_inputs)
    mhk_cost_results.simulate_costs()
    
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
    WEC_Cost['elec_infrastruc_costs'] = 0
    WEC_Cost['opex']= 0
    WEC_Cost['bos'] = 0
    WEC_Cost['capex'] = 0
    WEC_Cost['financial_costs'] = 0

PV_Cost={}
if plant_config['wave']['flag']:
    PV_Cost['capex']=plant_config["PVCosts"]["Module"]+plant_config["PVCosts"]["Inverter"]+plant_config["PVCosts"]["StructuralBOS"]
    PV_Cost['bos']=plant_config["PVCosts"]["SiteStaging"]+plant_config["PVCosts"]["InstallationandLabour"]+plant_config["PVCosts"]["EPCoverhead"]+plant_config["PVCosts"]["permitingInspectionandinterconnection"]+plant_config["PVCosts"]["shippingHandeling"]+plant_config["PVCosts"]["salestax"]+plant_config["PVCosts"]["Contingency"]+plant_config["PVCosts"]["Developeroverhead"]+plant_config["PVCosts"]["EPCDevProfit"]
    PV_Cost['elec_infrastruc_costs']=plant_config["PVCosts"]["ElectricalBOS"]
    plant_config["pv"]["solar_cost_kw"] = PV_Cost['capex'] + PV_Cost['bos'] + PV_Cost['elec_infrastruc_costs'] 
else:
    plant_config["pv"]["solar_cost_kw"] = 0
    
"""
#Setup & run HOPP
"""
hopp_site, hopp_technologies, hopp_scenario, hopp_h2_args = he_hopp.setup_hopp(plant_config, turbine_config, wind_resource, orbit_project, floris_config, WEC_Cost, solar_resource_file=data['resource']['solar'],wave_resource_file=data['resource']['wave'],Wave_PowerMatrix=data['powermatrix']['wave'],show_plots=show_plots, save_plots=save_plots)
hopp_results = he_hopp.run_hopp(hopp_site, hopp_technologies, hopp_scenario, hopp_h2_args, verbose=verbose)
#Results
HRly_Op=hopp_results["combined_power_production_hopp"] #Combined Power Output
HRly_Op2=np.asarray(hopp_results["combined_power_production_hopp"]) #Combined power for visulization
    
"""
#Load Grid profile data
"""
if plant_config["project_parameters"]["grid_connection"]:
    #read load_profile data
    load_profile_path= os.path.abspath(os.path.join(res_path,".."))+'\Oahu_load_profile.csv'
    #load_profile=tuple(list(map(float, list(line))) for line in csv.reader(open(load_profile_path)))
        
    csvFile = pd.read_csv(load_profile_path)
    HRly_Op4=HRly_Op-(csvFile.Load_Profile*0.25)
else:
    HRly_Op4=HRly_Op
"""
#Setup and Run H2 model
"""
# Hydrogen Model calculate outputs given power input
verbose=False
use_profast=True
incentive_option=1
electrolyzer_rating=None
plant_design_scenario=0

design_scenario = plant_config["plant_design"]["scenario%s" %(plant_design_scenario)]
design_scenario["id"]=plant_design_scenario

#Modify overall plant capacity
Plant_Capacity = plant_config["plant"]["capacity"]
if plant_config['pv']['flag']:
    Plant_Capacity += plant_config['pv']['system_capacity_kw']/1000
if plant_config['wave']['flag']:
    Plant_Capacity += plant_config['wave']['device_rating_kw']*plant_config['wave']['num_devices']/1000
# Need to modify to clculate Capacity factor and this was not done earlier because this is used to verify in Orbit which is wind only
plant_config["plant"]["capacity"] =Plant_Capacity

#Modify electrolyzer size
if plant_config['electrolyzer']['override_rating']:
    if electrolyzer_rating != None:
        plant_config["electrolyzer"]["rating"] = electrolyzer_rating
    else:
        electrolyzer_rating=Plant_Capacity

def energy_internals(HRly_Op=HRly_Op4, hopp_results=hopp_results, hopp_scenario=hopp_scenario, 
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
    
"""
# Run cost models
"""
platform_results = he_h2.run_equipment_platform(plant_config, design_scenario, electrolyzer_physics_results, h2_storage_results, desal_results, verbose=verbose)
capex, capex_breakdown = he_fin.run_capex(hopp_results, orbit_project, WEC_Cost, electrolyzer_cost_results, h2_pipe_array_results, h2_transport_compressor_results, h2_transport_pipe_results, h2_storage_results, plant_config, design_scenario, desal_results, platform_results, verbose=verbose)
opex_annual, opex_breakdown_annual = he_fin.run_opex(hopp_results, orbit_project, WEC_Cost, electrolyzer_cost_results, h2_pipe_array_results, h2_transport_compressor_results, h2_transport_pipe_results, h2_storage_results, plant_config, desal_results, platform_results, verbose=verbose, total_export_system_cost=capex_breakdown["electrical_export_system"])
    
if use_profast:
    lcoe, pf_lcoe, NPV, PI, IPP = he_fin.run_profast_lcoe(plant_config, orbit_project, capex_breakdown, opex_breakdown_annual, hopp_results, design_scenario, verbose=verbose, show_plots=show_plots, save_plots=save_plots)    
    lcoh_grid_only, pf_grid_only = he_fin.run_profast_grid_only(plant_config, orbit_project, electrolyzer_physics_results, capex_breakdown, opex_breakdown_annual, hopp_results, design_scenario, total_accessory_power_renewable_kw, total_accessory_power_grid_kw, verbose=verbose, show_plots=show_plots, save_plots=save_plots)
    lcoh, pf_lcoh = he_fin.run_profast_full_plant_model(plant_config, orbit_project, electrolyzer_physics_results, capex_breakdown, opex_breakdown_annual, hopp_results, incentive_option, design_scenario, total_accessory_power_renewable_kw, total_accessory_power_grid_kw, verbose=verbose, show_plots=show_plots, save_plots=save_plots)

print('Total plant capacity: ' + str(Plant_Capacity) + ' MW')
print('Avg output per hour: ' + str(sum(hopp_results['combined_power_production_hopp'])/1000/365/24) + ' MW')
print('AEP: ' + str(sum(hopp_results['combined_power_production_hopp'])/1000) + ' MW')
print('Total Power Curtailment: ' + str(sum(hopp_results['combined_curtailment_hopp'])))
print('Net Present Value: ' + str(NPV))
print('profit index: ' + str(PI))
print('investor payback period: ' + str(IPP))
print('Annual H2 Production: ' + str(electrolyzer_physics_results["H2_Results"]["hydrogen_annual_output"]) + ' kg')
print('Avg H2 Production per day: ' + str(electrolyzer_physics_results["H2_Results"]["hydrogen_annual_output"]/8760) + ' kg')
power_breakdown = he_util.post_process_simulation(lcoe, lcoh, pf_lcoh, pf_lcoe, hopp_results, electrolyzer_physics_results, plant_config, h2_storage_results, capex_breakdown, opex_breakdown_annual, orbit_project, platform_results, desal_results, design_scenario, plant_design_scenario, incentive_option, solver_results=solver_results, show_plots=show_plots, save_plots=save_plots)#, lcoe, lcoh, lcoh_with_grid, lcoh_grid_only)

#hopp_results['annual_energies']['pv']
#hopp_results['annual_energies']['wave']
#hopp_results['annual_energies']['wind']
