"""
Author:Charles Kiefer
Date: 4/11/2023
Institution: National Renewable Energy Lab 
Description: This file shall handle costing and sizing of offshore floating platforms deicated to hydrogen production.  It uses the 
             same foundation as fixed_platform.py.  Both have been modeled off of existing BOS cost/sizing calculations fond in ORBIT.
             It can be run as standalone functions or as appended ORBIT project phases.


             
Sources:
    - [1] ORBIT: https://github.com/WISDEM/ORBIT electrical_refactor branch & SemiTaut_mooring branch
Args:
    - tech_required_area: (float): area needed for combination of all tech (m^2), not including buffer or working space
    - tech_combined_mass: (float): mass of all tech being placed on the platform (kg or tonnes)year

   
    - depth: (float): bathometry at the platform location (m)
    - distance_to_port: (float): distance ships must travel from port to site location (km)
    
    Future arguments: (Not used at this time)
    - construction year  (int): 
    - lifetime (int): lifetime of the plant in years (may not be needed)

Returns:
    - platform_mass (float): Adjusted mass of platform + substructure
    - design_capex (float): capital expenditures (platform design + substructure fabrication)
    - installation_capex (float): capital expenditures (installation cost)
    - platform_opex (float): the OPEX (annual, fixed) in USD for the platform

"""
''' 
Notes:
    Thank you Jake Nunemaker's oswh2 repository and Rebecca Fuchs SemiTaut_mooring repository!!!
    pile_cost=0 $US/tonne for monopile construction. Not a bug, this # is consistent with the rest of ORBIT
'''

import os
import math
# 
from ORBIT import ProjectManager, load_config
from ORBIT.core import Vessel
from ORBIT.core.library import initialize_library
from ORBIT.phases.design import DesignPhase
from ORBIT.phases.install import InstallPhase
from scipy.interpolate import interp1d
import numpy as np


class FloatingPlatformDesign(DesignPhase):
    '''
    This is a modified class based on ORBIT's design phase
    '''

    #phase = "H2 Floating Platform Design"

    # Expected inputs from config yaml file
    expected_config = {
        "site": {
            "distance_m" : "int | float",
            "depth_m" : "int | float",
        }, 

        "equipment": {
            "tech_required_area_m^2" : "float", 
            "tech_combined_mass_t" : "float",
            "topside_design_cost_USD": "USD (optional, default:4.5e6)",
            "fabrication_cost_rate_USD": "USD/t (optional, default: 14500.)",
            "substructure_steel_rate_USD": "USD/t (optional, default: 3000.)",
        }

    }

    # Takes in arguments and initialize library files
    def __init__(self, config, **kwargs):
        
        self.phase = "H2 Floating Platform Design"

        config = self.initialize_library(config, **kwargs)
        self.config = self.validate_config(config)

        self._outputs = {}
            # Runs the design cost models 

    def run(self):
        
        #print("Floating Platform Design run() is working!!!")

        self.distance = self.config['site']['distance']     # km
        self.depth = self.config['site']['depth']           # m

        _platform = self.config.get('equipment',{})

        self.mass = _platform.get('tech_combined_mass',999)     # t
        self.area = _platform.get('tech_required_area', 1000)   # m**2

        design_cost = _platform.get('topside_design_cost', 4.5e6)   # USD
        fab_cost = _platform.get('fabrication_cost_rate', 14500.)   # USD/t
        steel_cost = _platform.get('substructure_steel_cost', 3000) # USD/t
        ##NEED updated version
        # Add individual calcs/functions in the run() method
        total_cost, total_mass = calc_substructure_mass_and_cost(self.mass, self.area, 
                        self.depth, fab_cost, design_cost, steel_cost
                        )

        # Create an ouput dict 
        self._outputs['floating_platform'] = {
            "mass_t" : total_mass, 
            "area_m^2" : self.area,
            "total_cost_USD" : total_cost
        }

    # A design object needs to have attribute design_result and detailed_output
    @property
    def design_result(self):

        return {
            "platform_design":{
                "mass_t" : self._outputs['floating_platform']['mass'],
                "area_m^2" : self._outputs['floating_platform']['area'],
                "total_cost_USD": self._outputs['floating_platform']['total_cost'],
            }
        }

    @property
    def detailed_output(self):

        return {}

class FloatingPlatformInstallation(InstallPhase):
    '''
    This is a modified class based on ORBIT's install phase 
    '''

    #phase = "H2 Floating Platform Installation"
    
    # Expected inputs from config yaml file
    expected_config = {
        "site": {
            "distance_m" : "int | float",
            "depth_m" : "int | float",
        }, 

        "equipment": {
            "tech_required_area_m^2" : "float", 
            "tech_combined_mass_t" : "float",
            "install_duration": "days (optional, default: 14)",
        },

        "oss_install_vessel" : "str | dict",
    }

    # Need to initialize arguments and weather files 
    def __init__(self, config, weather=None, **kwargs):
        
        super().__init__(weather, **kwargs)

        config = self.initialize_library(config, **kwargs)
        self.config = self.validate_config(config)

        self.initialize_port()
        self.setup_simulation(**kwargs)

    # Setup simulation seems to be the install phase's equivalent run() module
    def setup_simulation(self, **kwargs):

        #print("Floating Platform Install setup_sim() is working!!!")

        self.distance = self.config['site']['distance']
        self.depth = self.config['site']['depth']
        self.mass = self.config['equipment']['tech_combined_mass']
        self.area = self.config['equipment']['tech_required_area']

        _platform = self.config.get('equipment', {})
        design_cost = _platform.get('topside_design_cost', 4.5e6)   # USD
        fab_cost_rate = _platform.get('fabrication_cost_rate', 14500.)   # USD/t
        steel_cost = _platform.get('substructure_steel_cost', 3000) # USD/t
        
        install_duration = _platform.get("install_duration", 14)    # days
        
        # Initialize vessel 
        vessel_specs = self.config.get("oss_install_vessel", None)
        name = vessel_specs.get("name","Offshore Substation Install Vessel")

        vessel = Vessel(name, vessel_specs)
        self.env.register(vessel)

        vessel.initialize()
        self.install_vessel = vessel
        
        # Add in the mass of the substructure to total mass (may or may not impact the final install cost)
        _, substructure_mass = calc_substructure_mass_and_cost(self.mass, self.area, 
                        self.depth, fab_cost_rate, design_cost, steel_cost
                        )

        total_mass = self.mass + substructure_mass  # t

         # Call the install_platform function
        self.install_capex = install_platform(total_mass, self.area, self.distance, \
                                                   install_duration, self.install_vessel)

    # An install object needs to have attribute system_capex, installation_capex, and detailed output
    @property
    def system_capex(self):

        return {}

    @property 
    def installation_capex(self):
        
        return self.install_capex

    @property
    def detailed_output(self):

        return {}

# Define individual calculations and functions to use outside or with ORBIT
def calc_substructure_mass_and_cost(mass, area, depth, fab_cost_rate=14500., design_cost=4.5e6, sub_cost_rate=3000):
    '''
    calc_substructure_mass_and_cost returns the total mass including substructure, topside and equipment.  Also returns the cost of the substructure and topside
    Inputs: mass            | Mass of equipment on platform (tonnes)
            area            | Area needed for equipment (meter^2) (not necessary)
            depth           | Ocean depth at platform location (meters)
            fab_cost_rate   | Cost rate to fabricate topside (USD/tonne)
            design_cost     | Design cost to design structural components (USD) from ORBIT
            sub_cost_rate   | Steel cost rate (USD/tonne) from ORBIT'''

    '''
    Platform is substructure and topside combined
    All functions are based off NREL's ORBIT (oss_design)
    default values are specified in ORBIT
    '''
    topside_mass = mass
    topside_fab_cost_rate   =   fab_cost_rate   
    topside_design_cost     =   design_cost

    '''Topside Cost & Mass
    Topside Mass is the required Mass the platform will hold
    Topside Cost is a function of topside mass, fab cost and design cost'''
    topside_cost   =   topside_mass*topside_fab_cost_rate + topside_design_cost

    '''Substructure
    Substructure Mass is a function of the topside mass
    Substructure Cost is a function of of substructure mass pile mass and cost rates for each'''

    substructure_cost_rate  =   sub_cost_rate        # USD/t

    substructure_mass       =   0.4*topside_mass        # t
    substructure_cost       =   (substructure_mass*substructure_cost_rate)     # USD  
    substructure_total_mass =   substructure_mass       # t

    '''These arrays were pulled from Rebecca Fuchs SemitTaut_mooring cost models'''
    depths = np.array([0, 500, 750, 1000, 1250, 1500]) # [m]

    anchor_costs = np.array([28823, 112766, 125511, 148703, 204988, 246655]) # [USD]
    finterp_anchor_cost = interp1d(depths, anchor_costs)
    anchor_cost = finterp_anchor_cost(depth)

    total_line_costs = np.array([430315, 826598, 1221471, 1682208, 2380035, 3229700]) # [USD]
    finterp_total_line_cost = interp1d(depths, total_line_costs)
    line_cost = finterp_total_line_cost(depth)
    
    mooring_cost = line_cost + anchor_cost
    
    '''Total Platform capex = capex Topside + capex substructure'''
    total_capex = (topside_cost + substructure_cost + mooring_cost)
    platform_capex  = total_capex # USD
    platform_mass   = substructure_total_mass + topside_mass    # t 
    #mass of equipment and floating substructure for substation
    
    return platform_capex, platform_mass

#@process
def install_platform(mass, area, distance, install_duration=14, vessel=None):
    '''
    A simplified platform installation costing model. 
    Total Cost = install_cost * duration 
         Compares the mass and/or deck space of equipment to the vessel limits to determine 
         the number of trips. Add an additional "at sea" install duration
          
    '''
    # print("Install process worked!")
    # If no ORBIT vessel is defined set default values (based on ORBIT's floating_heavy_lift_vessel)
    if vessel == None:
        vessel_cargo_mass = 7999 # t
        vessel_deck_space = 3999 # m**2 
        vessel_day_rate = 500001 # USD/day 
        vessel_speed = 7 # km/hr 
    else:
        vessel_cargo_mass = vessel.storage.max_cargo_mass # t
        vessel_deck_space = vessel.storage.max_deck_space # m**2 
        vessel_day_rate = vessel.day_rate # USD/day 
        vessel_speed = vessel.transit_speed # km/hr 

    #print("Max Vessel Cargo and Mass:", vessel_cargo_mass, vessel_deck_space)

    # Get the # of trips based on ships cargo/space limits 
    num_of_trips = math.ceil(max((mass / vessel_cargo_mass), (area / vessel_deck_space)))
    #print("Number of trips:   ", num_of_trips)

    # Total duration = double the trips + install_duration
    duration = (2 * num_of_trips * distance) / (vessel_speed * 24) + install_duration # days
    #print("Duration (days):   %0.2f" % duration)

    # Final install cost is obtained by using the vessel's daily rate 
    install_cost = vessel_day_rate * duration   # USD

    return install_cost

def calc_platform_opex(capex, opex_rate=0.011):
    '''
    Simple opex calculation based on a capex
        https://www.acm.nl/sites/default/files/documents/study-on-estimation-method-for-additional-efficient-offshore-grid-opex.pdf
    
    Output in $USD/year
    '''
    
    opex = capex * opex_rate    # USD/year

    #print("OpEx of platform:", opex)
    
    return opex


# Standalone test sections 
if __name__ == '__main__':
    print("\n*** New FloatingPlatform Standalone test section ***\n")

    orbit_libpath = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir, 'ORBIT', 'library'))
    print(orbit_libpath)
    initialize_library(orbit_libpath)

    config_path = os.path.abspath(__file__)
    config_fname = load_config(os.path.join(config_path, os.pardir, "example_floating_project.yaml"))

    
    ProjectManager._design_phases.append(FloatingPlatformDesign)
    ProjectManager._install_phases.append(FloatingPlatformInstallation)

    platform = ProjectManager(config_fname)
    platform.run()

    design_capex = platform.design_results['platform_design']['total_cost']
    install_capex = platform.installation_capex

    #print("Project Params", h2platform.project_params.items())
    platform_opex = calc_platform_opex((design_capex + install_capex))

    print("ORBIT Phases: ", platform.phases.keys())
    print(f"\tH2 Platform Design Capex:    {design_capex:.0f} USD")
    print(f"\tH2 Platform Install Capex:  {install_capex:.0f} USD")
    print('')
    print(f"\tTotal H2 Platform Capex:   {(design_capex+install_capex)/1e6:.0f} mUSD")
    print(f"\tH2 Platform Opex: {platform_opex:.0f} USD/year")
