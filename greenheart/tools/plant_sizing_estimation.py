import greenheart.tools.eco.electrolysis as he_elec
from greenheart.simulation.technologies.steel import steel
from greenheart.simulation.technologies.ammonia import ammonia 
import warnings
from hopp.utilities import load_yaml
import os

def size_electrolyzer_for_end_use(greenheart_config):
    
    hybrid_electricity_estimated_cf = greenheart_config["project_parameters"]["hybrid_electricity_estimated_cf"]
    
    if "ammonia" in list(greenheart_config.keys()):
        feedstocks = ammonia.Feedstocks({'electricity_cost':0,
        'hydrogen_cost':0,'cooling_water_cost':0,
        'iron_based_catalyst_cost':0,'oxygen_cost':0})
        config = ammonia.AmmoniaCapacityModelConfig(
            input_capacity_factor_estimate = greenheart_config["ammonia"]["capacity"]['input_capacity_factor_estimate'],
            feedstocks = feedstocks,
            desired_ammonia_kgpy = greenheart_config["ammonia"]["capacity"]["annual_production_target"],
            )
        output = ammonia.run_size_ammonia_plant_capacity(config)
        
    if "steel" in list(greenheart_config.keys()):
        feedstocks = steel.Feedstocks(natural_gas_prices={})
        config = steel.SteelCapacityModelConfig(
            input_capacity_factor_estimate=greenheart_config["steel"]["capacity"]['input_capacity_factor_estimate'],
            feedstocks = feedstocks,
            desired_steel_mtpy = greenheart_config["steel"]["capacity"]["annual_production_target"],
        )
        output = steel.run_size_steel_plant_capacity(config)
        
    hydrogen_production_capacity_required_kgphr = output.hydrogen_amount_kgpy/(8760*hybrid_electricity_estimated_cf)
    
    deg_power_inc = greenheart_config["electrolyzer"]['eol_eff_percent_loss']/100
    bol_or_eol_sizing = greenheart_config["electrolyzer"]["sizing"]["size_for"]
    cluster_cap_mw = greenheart_config["electrolyzer"]["cluster_rating_MW"]
    electrolyzer_capacity_BOL_MW = he_elec.size_electrolyzer_for_hydrogen_demand(hydrogen_production_capacity_required_kgphr, size_for = bol_or_eol_sizing,electrolyzer_degradation_power_increase=deg_power_inc)
    electrolyzer_size_mw = he_elec.check_capacity_based_on_clusters(electrolyzer_capacity_BOL_MW,cluster_cap_mw)

    greenheart_config["electrolyzer"]["rating"] = electrolyzer_size_mw
    greenheart_config["electrolyzer"]["sizing"]["hydrogen_dmd"] = hydrogen_production_capacity_required_kgphr

    return greenheart_config

def run_resizing_estimation(greenheart_config):
    if greenheart_config["project_parameters"]["hybrid_electricity_estimated_cf"] > 1:
        raise(ValueError("hybrid plant capacity factor estimate (hybrid_electricity_estimated_cf) cannot exceed 1"))
    
    if greenheart_config["project_parameters"]["grid_connection"]:
        if greenheart_config["project_parameters"]["hybrid_electricity_estimated_cf"]<1:
            print("hybrid_electricity_estimated_cf reset to 1 for grid-connected cases")
            # warnings.warn("")
            greenheart_config["project_parameters"]["hybrid_electricity_estimated_cf"] = 1
    
    if greenheart_config["electrolyzer"]["sizing"]["resize_for_enduse"]:
        greenheart_config = size_electrolyzer_for_end_use(greenheart_config)

    return greenheart_config

