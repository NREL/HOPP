"""
Author:
Date:
Institution:
Description: This file should handle the cost and sizing of offshore platforms, but can certainly use WISDEM (fixed_bottomse)
                or ORBIT for much of the modeling effort. 
Sources:
    - [1] ORBIT: https://github.com/WISDEM/ORBIT
    - [2] fixed_bottomse: https://github.com/WISDEM/WISDEM/tree/master/wisdem/fixed_bottomse
Args:
    - year (int): construction year
    - any/all ORBIT inputs are available as needed. Including, but not limited to:
        - depth (float): water depth at desired installation location
        - port_distance (float): distance from port
    - tech_required_area (float): area needed for combination of all tech (m^2), not including buffer or working space
    - tech_combined_mass (float): mass of all tech being placed on the platform (kg or tonnes)
    - lifetime (int): lifetime of the plant in years (may not be needed)
    - others may be added as needed
Returns:(can be from separate functions and/or methods as it makes sense):
    - capex (float): capital expenditures for building the platform, including material costs and installation
    - opex (float): the OPEX (annual, fixed) in USD for the platform
    - others may be added as needed
"""

import os
import math
# 
from ORBIT import ProjectManager, load_config
from ORBIT.core import Vessel
from ORBIT.core.library import initialize_library
from ORBIT.phases.design import DesignPhase # OffshoreSubstationDesign
from ORBIT.phases.install import InstallPhase # OffshoreSubstationInstallation
from marmot import process

''' 
Thank you Jake Nunemaker h2export repository!!!
Able to create additional Design phase classes and append to Project manager. 
Need to add the name of the class to the configuration yaml file under "design_phases"
'''

class FixedPlatformDesign(DesignPhase):

    phase = "H2 Fixed Platform Design"
    
    expected_config = {
        "site": {
            "distance" : "int | float",
            "depth" : "int | float",
        }, 

        "h2_platform": {
            "tech_required_area" : "float", 
            "tech_combined_mass" : "float",
            "fabrication_cost_rate": "USD/t (optional, default: 14500)",
            "substructure_steel_rate": "USD/t (optional, default: 3000)",
        }

    }
    
    # Needs to take in arguments 
    def __init__(self, config, **kwargs):
        
        self.phase = "H2 Fixed Platform Design"

        config = self.initialize_library(config, **kwargs)
        self.config = self.validate_config(config)

        self._outputs = {}

    # Needs a run method
    def run(self):
        
        print("Fixed Platform Design run() is working!!!")

        self.distance = self.config['site']['distance']
        self.mass = self.config['h2_platform']['tech_combined_mass']
        self.area = self.config['h2_platform']['tech_required_area']

        # Add individual calcs/functions in the run() method
        #self.calc_platform_capex()
        total_cost = calc_substructure_mass_and_cost(self)

        self._outputs['fixed_platform_h2'] = {
            "mass" : self.mass, 
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

    phase = "H2 Fixed Platform Installation"
    
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

    # Need to initiate some stuff with weather 

    def __init__(self, config, weather=None, **kwargs):
        
        super().__init__(weather, **kwargs)

        config = self.initialize_library(config, **kwargs)
        self.config = self.validate_config(config)

        self.initialize_port()
        self.setup_simulation(**kwargs)

    # Cant initiate without setup_simulation 
    def setup_simulation(self, **kwargs):

        print("Fixed Platform Install setup_sim() is working!!!")

        self.distance = self.config['site']['distance']
        self.mass = self.config['h2_platform']['tech_combined_mass']
        self.area = self.config['h2_platform']['tech_required_area']
        self.install_duration = self.config.get("install_duration", 14)
        
        # Initialize vessel 
        vessel_specs = self.config.get("oss_install_vessel", None)
        name = vessel_specs.get("name","Offshore Substation Install Vessel")

        vessel = Vessel(name, vessel_specs)
        self.env.register(vessel)

        vessel.initialize()
        self.install_vessel = vessel
        

        self.distance = self.config['site']['distance']

        self.install_capex = install_h2_platform(self.mass, self.area, self.distance, self.install_duration, self.install_vessel)

    #@property
    def system_capex(self):

        return {}

    @property 
    def installation_capex(self):
        
        return self.install_capex

    # Cant initiate without abtract method detailed_output

    @property
    def detailed_output(self):

        return {}

# Define individual calcs/functions 
def calc_substructure_mass_and_cost(self):
    '''
    Copy this '''
    platform_mass = self.mass
    platform_capex = 987654321

    return platform_capex

#@process
def install_h2_platform(mass, area, distance, install_duration=14, vessel=None):
    '''
    A simplified platform installation costing model. 
    Total Cost = install_cost * duration 
         Compare the mass and deck space of equipment to the vessel limits to determine 
         the number of trips. Add an additional "at sea" install duration 
    '''
    print("Install process worked!")
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
