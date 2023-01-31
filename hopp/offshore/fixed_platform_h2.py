"""
Author:
Date:
Institution:
Description: This file should handles the cost and sizing of a centralized offshore platform dedicated to hydrogen production. It 
             has been modeled off of existing BOS cost/sizing calculations found in ORBIT (Thank you Jake Nunemaker). 
             It can be run as standalone functions or as appended ORBIT project phases. 

             
Sources:
    - [1] ORBIT: https://github.com/WISDEM/ORBIT electrical_refactor branch
Args:
    - tech_required_area: (float): area needed for combination of all tech (m^2), not including buffer or working space
    - tech_combined_mass: (float): mass of all tech being placed on the platform (kg or tonnes)year

   
    - depth: (float): bathometry at the platform location (m)
    - distance_to_port: (float): distance ships must travel from port to site location (km)
    
    Future arguments: (Not used at this time)
    - construction year  (int): 
    - lifetime (int): lifetime of the plant in years (may not be needed)

Returns:
    -  platform_mass (float): Adjusted mass of platform + substructure
    - platform_capex (float): capital expenditures (platform design, substructure fabrication, and installation costs)
    - platform_opex (float): the OPEX (annual, fixed) in USD for the platform

"""
''' 
Thank you Jake Nunemaker's oswh2 repository!!!
Able to follow his example and create additional Design phase classes and append to Project manager. 
Need to add the name of the class to the configuration yaml file under "design_phases"
'''

import os
import math
# 
from ORBIT import ProjectManager, load_config
from ORBIT.core import Vessel
from ORBIT.core.library import initialize_library
from ORBIT.phases.design import DesignPhase
from ORBIT.phases.install import InstallPhase

class FixedPlatformDesign(DesignPhase):
    '''
    This is a modified class based on ORBIT's design phase 
    '''
    
    #phase = "H2 Fixed Platform Design"
    
    # Expected inputs from config yaml file
    expected_config = {
        "site": {
            "distance" : "int | float",
            "depth" : "int | float",
        }, 

        "h2_platform": {
            "tech_required_area" : "float", 
            "tech_combined_mass" : "float",
            "fabrication_cost_rate": "USD/t (optional, default: 14500.)",
            "substructure_steel_rate": "USD/t (optional, default: 3000.)",
        }

    }
    
    # Takes in arguments and initialize library files
    def __init__(self, config, **kwargs):
        
        self.phase = "H2 Fixed Platform Design"

        config = self.initialize_library(config, **kwargs)
        self.config = self.validate_config(config)

        self._outputs = {}

    # Runs the design cost models 
    def run(self):
        
        #print("Fixed Platform Design run() is working!!!")

        self.distance = self.config['site']['distance']
        self.depth = self.config['site']['depth']

        _platform = self.config.get('h2_platform',{})

        self.mass = _platform.get('tech_combined_mass',999)
        self.area = _platform.get('tech_required_area', 1000)

        fab_cost = _platform.get('fabrication_cost_rate', 14500.)
        steel_cost = _platform.get('substructure_steel_cost')

        # Add individual calcs/functions in the run() method
        total_cost, total_mass = calc_substructure_mass_and_cost(self.mass, self.area, self.depth, fab_cost)

        # Create an ouput dict 
        self._outputs['fixed_platform_h2'] = {
            "mass" : total_mass, 
            "area" : self.area,
            "total_cost" : total_cost
        }

    # A design object needs to have attribute design_result and detailed_output
    @property
    def design_result(self):

        return {
            "h2_platform_design":{
                "mass" : self._outputs['fixed_platform_h2']['mass'],
                "area" : self._outputs['fixed_platform_h2']['area'],
                "total_cost": self._outputs['fixed_platform_h2']['total_cost'],
            }
        }

    @property
    def detailed_output(self):

        return {}

class FixedPlatformInstallation(InstallPhase):
    '''
    This is a modified class based on ORBIT's install phase 
    '''

    #phase = "H2 Fixed Platform Installation"
    
    # Expected inputs from config yaml file
    expected_config = {
        "site": {
            "distance" : "int | float",
            "depth" : "int | float",
        }, 

        "h2_platform": {
            "tech_required_area" : "float", 
            "tech_combined_mass" : "float",
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

        print("Fixed Platform Install setup_sim() is working!!!")

        self.distance = self.config['site']['distance']
        self.depth = self.config['site']['depth']
        self.mass = self.config['h2_platform']['tech_combined_mass']
        self.area = self.config['h2_platform']['tech_required_area']
        install_duration = self.config.get("install_duration", 14)
        
        # Initialize vessel 
        vessel_specs = self.config.get("oss_install_vessel", None)
        name = vessel_specs.get("name","Offshore Substation Install Vessel")

        vessel = Vessel(name, vessel_specs)
        self.env.register(vessel)

        vessel.initialize()
        self.install_vessel = vessel
        
        # Add in the mass of the substructure to total mass (may or may not impact the total install cost)
        _, substructure_mass = calc_substructure_mass_and_cost(self.mass, self.area, self.depth)

        # Call the install_h2_platform function
        total_mass = self.mass + substructure_mass
        self.install_capex = install_h2_platform(total_mass, self.area, self.distance, \
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
def calc_substructure_mass_and_cost(mass, area, depth, fab_cost=14500., design_cost=4.5e6, sub_cost=3000, pile_cost=0):
    '''
    Platform is substructure and topside combined
    All funstions are based off NREL's ORBIT (oss_design)
    default values are specified in ORBIT
    '''
    #Inputs needed
    topside_mass = mass
    topside_fab_cost_rate   =   fab_cost   
    topside_design_cost     =   design_cost

    '''Topside Cost & Mass
    Topside Mass is the required Mass the platform will hold
    Topside Cost is a function of topside mass, fab cost and design cost'''
    topside_cost   =   topside_mass   *topside_fab_cost_rate  +topside_design_cost

    '''Substructure
    Substructure Mass is a function of the topside mass
    Substructure Cost is a function of of substructure mass pile mass and cost rates for each'''

    #inputs needed
    substructure_cost_rate  =   sub_cost
    pile_cost_rate          =   pile_cost

    substructure_mass       =   0.4 *   topside_mass
    substructure_pile_mass  =   8   *   substructure_mass**0.5574
    substructure_cost  =   (substructure_mass  *substructure_cost_rate +
        substructure_pile_mass *pile_cost_rate) 
        
    substructure_total_mass  =   substructure_mass   +substructure_pile_mass

    '''Total Platform capex = capex Topside + capex substructure'''
    
    platform_capex  = substructure_cost + topside_cost
    platform_mass   = substructure_total_mass + topside_mass
    
    return platform_capex, platform_mass

#@process
def install_h2_platform(mass, area, distance, install_duration=14, vessel=None):
    '''
    A simplified platform installation costing model. 
    Total Cost = install_cost * duration 
         Compares the mass and/or deck space of equipment to the vessel limits to determine 
         the number of trips. Add an additional "at sea" install duration 
    '''
    # print("Install process worked!")
    # If no ORBIT vessel is defined set default values (based on ORBIT's example_heavy_lift_vessel)
    if vessel == None:
        vessel_cargo_mass = 7999 # tonnes 
        vessel_deck_space = 3999 # m**2 
        vessel_day_rate = 500001 # USD/day 
        vessel_speed = 5 # km/hr 
    else:
        vessel_cargo_mass = vessel.storage.max_cargo_mass # tonnes 
        vessel_deck_space = vessel.storage.max_deck_space # m**2 
        vessel_day_rate = vessel.day_rate # USD/day 
        vessel_speed = vessel.transit_speed # km/hr 

    #print("Max Vessel Cargo and Mass:", vessel_cargo_mass, vessel_deck_space)

    # Get the # of trips based on ships cargo/space limits 
    num_of_trips = math.ceil(max((mass / vessel_cargo_mass), (area / vessel_deck_space)))
    #print("Number of trips:   ", num_of_trips)

    # Total duration = double the trips + install_duration
    duration = (2 * num_of_trips * distance) / (vessel_speed * 24) + install_duration
    #print("Duration (days):   %0.2f" % duration)

    # Final install cost is obtained by using the vessel's daily rate 
    install_cost = vessel_day_rate * duration

    return install_cost

def calc_h2_platform_opex(lifetime, capacity, opex_rate=111):

    opex = opex_rate * capacity * lifetime

    #print("OpEx of platform:", opex)
    
    return opex


# Test sections 
if __name__ == '__main__':
    print("\n*** New FixedPlatform Testing section ***\n")

    orbit_libpath = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir, 'ORBIT', 'library'))
    print(orbit_libpath)
    initialize_library(orbit_libpath)

    config_path = os.path.abspath(__file__)
    config_fname = load_config(os.path.join(config_path, os.pardir, "example_fixed_project_h2.yaml"))

    
    ProjectManager._design_phases.append(FixedPlatformDesign)
    ProjectManager._install_phases.append(FixedPlatformInstallation)

    h2platform = ProjectManager(config_fname)
    h2platform.run()

    design_capex = h2platform.design_results['h2_platform_design']['total_cost']
    install_capex = h2platform.installation_capex

    #print("Project Params", h2platform.project_params.items())
    h2_opex = calc_h2_platform_opex(h2platform.project_params['project_lifetime'], \
                                       h2platform.config['plant']['capacity'],\
                                            h2platform.project_params['opex_rate'])

    print("ORBIT Phases: ", h2platform.phases.keys())
    print(f"\tH2 Platform Design Capex:    {design_capex:.0f} USD")
    print(f"\tH2 Platform Install Capex:  {install_capex:.0f} USD")
    print('')
    print(f"\tTotal H2 Platform Capex:   {(design_capex+install_capex)/1e6:.0f} mUSD")
    print(f"\tH2 Platform Lifetime Opex: {h2_opex/1e6:.0f} mUSD")
