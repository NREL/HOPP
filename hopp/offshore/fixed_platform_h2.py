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
# 
from ORBIT import ProjectManager, load_config
from ORBIT.core import Vessel
from ORBIT.core.library import initialize_library
from ORBIT.phases.design import DesignPhase # OffshoreSubstationDesign
from ORBIT.phases.install import InstallPhase # OffshoreSubstationInstallation

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

        "tech": {
            "tech_required_area" : "float", 
            "tech_combined_mass" : "float",
        }

        # Optional config input   
        #"substation_design": {
        #    "mpt_cost_rate": "USD/MW (optional)",
        #}
    }
    
    # Needs to take in arguments 
    def __init__(self, config, **kwargs):
        
        self.phase = "H2 Fixed Platform Design"

        config = self.initialize_library(config, **kwargs)
        self.config = self.validate_config(config)

        self._outputs = {}

    # Needs a run method
    def run(self):
        
        self.distance = self.config['site']['distance']

        # Add individual calcs/functions in the run() method
        self.calc_platform_capex()

        self._outputs['fixed_platform_h2'] = {
            "capex" : self.platform_capex,
        }

        print("Fixed Platform Design run() is working!!!")
    
    # Define individual calcs/functions 
    def calc_platform_capex(self):

        self.platform_capex = 987654321

    # A design object needs to have attribute design_result and detailed_output

    @property
    def design_result(self):

        return {
            "h2_subsystem":{
                "platform": self.platform_capex
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

        "tech": {
            "tech_required_area" : "float", 
            "tech_combined_mass" : "float",
        }
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

        # Initialize vessel 
        vessel_specs = self.config.get("oss_install_vessel", None)
        name = vessel_specs.get("name","Offshore Substation Install Vessel")

        vessel = Vessel(name, vessel_specs)
        self.env.register(vessel)

        vessel.initialize()
        self.install_vessel = vessel

        self.distance = self.config['site']['distance']

    @property
    def system_capex(self):

        install_cost = 123456
        return install_cost

    # Cant initiate without abtract method detailed_output

    @property
    def detailed_output(self):

        return {}


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

    print("Project Phases:    ", h2platform.phases.keys())
#    print("Project Phases:    ", h2platform.design_results)

