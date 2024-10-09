"""
Author: Nick Riccobono and Charles Kiefer
Date: 1/31/2023
Institution: National Renewable Energy Lab 
Description: This file should handles the cost and sizing of a centralized offshore platform dedicated to hydrogen production. It 
             has been modeled off of existing BOS cost/sizing calculations found in ORBIT (Thank you Jake Nunemaker). 
             It can be run as standalone functions or as appended ORBIT project phases. 

             
Sources:
    - [1] ORBIT: https://github.com/WISDEM/ORBIT electrical_refactor branch
    - [2] J. Nunemaker, M. Shields, R. Hammond, and P. Duffy, 
          “ORBIT: Offshore Renewables Balance-of-System and Installation Tool,” 
          NREL/TP-5000-77081, 1660132, MainId:26027, Aug. 2020. doi: 10.2172/1660132.
    - [3] M. Maness, B. Maples, and A. Smith, 
          “NREL Offshore Balance-of-System Model,” 
          NREL/TP--6A20-66874, 1339522, Jan. 2017. doi: 10.2172/1339522.
Args:
    - tech_required_area: (float): area needed for combination of all tech (m^2), not including buffer or working space
    - tech_combined_mass: (float): mass of all tech being placed on the platform (kg or tonnes)year
   
    - depth: (float): bathometry at the platform location (m)
    - distance: (float): distance ships must travel from port to site location (km)
    
    Future arguments: (Not used at this time)
    - construction year  (int): 
    - lifetime (int): lifetime of the plant in years (may not be needed)
    - Assembly costs and construction on land

Returns:
    - platform_mass (float): Adjusted mass of platform + substructure
    - design_capex (float): capital expenditures (platform design + substructure fabrication)
    - installation_capex (float): capital expenditures (installation cost)
    - platform_opex (float): the OPEX (annual, fixed) in USD for the platform

"""
''' 
Notes:
    - Thank you Jake Nunemaker's oswh2 repository!!!
    - pile_cost=0 $US/tonne for monopile construction. Not a bug, this # is 
      consistent with the rest of ORBIT [1].
'''

import os
import math
# 
import ORBIT as orbit
from greenheart.simulation.technologies.offshore.all_platforms import calc_platform_opex, install_platform

class FixedPlatformDesign(orbit.phases.design.DesignPhase):
    '''
    This is a modified class based on ORBIT's [1] design phase. The implementation
    is discussed in [2], Section 2.5: Offshore Substation Design. Default values originate
    from [3], Appendix A: Inputs, Key Assumptions and Caveats. 
    '''
    
    #phase = "H2 Fixed Platform Design"
    
    # Expected inputs from config yaml file
    expected_config = {
        "site": {
            "distance" : "int | float",
            "depth" : "int | float",
        }, 

        "equipment": {
            "tech_required_area" : "float", 
            "tech_combined_mass" : "float",
            "topside_design_cost": "USD (optional, default:4.5e6)",
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

        self.distance = self.config['site']['distance']     # km
        self.depth = self.config['site']['depth']           # m

        _platform = self.config.get('equipment',{})

        self.mass = _platform.get('tech_combined_mass',999)     # t
        self.area = _platform.get('tech_required_area', 1000)   # m**2

        design_cost = _platform.get('topside_design_cost', 4.5e6)   # USD
        fab_cost = _platform.get('fabrication_cost_rate', 14500.)   # USD/t
        steel_cost = _platform.get('substructure_steel_rate', 3000) # USD/t

        # Add individual calcs/functions in the run() method
        total_cost, total_mass = calc_substructure_mass_and_cost(self.mass, self.area, 
                        self.depth, fab_cost, design_cost, steel_cost
                        )
        
        # Create an ouput dict 
        self._outputs['fixed_platform'] = {
            "mass" : total_mass, 
            "area" : self.area,
            "total_cost" : total_cost
        }

    # A design object needs to have attribute design_result and detailed_output
    @property
    def design_result(self):

        return {
            "platform_design":{
                "mass" : self._outputs['fixed_platform']['mass'],
                "area" : self._outputs['fixed_platform']['area'],
                "total_cost": self._outputs['fixed_platform']['total_cost'],
            }
        }

    @property
    def detailed_output(self):

        return {}

class FixedPlatformInstallation(orbit.phases.install.InstallPhase):
    '''
    This is a modified class based on ORBIT's [1] install phase. The implementation
    is duscussed in [2], Section 3.6: Offshore Substation Installation. Default values
    originate from [3], Appendix A: Inputs, Key Assumptions and Caveats.  
    '''

    #phase = "H2 Fixed Platform Installation"
    
    # Expected inputs from config yaml file
    expected_config = {
        "site": {
            "distance" : "int | float",
            "depth" : "int | float",
        }, 

        "equipment": {
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

        #print("Fixed Platform Install setup_sim() is working!!!")

        self.distance = self.config['site']['distance']
        self.depth = self.config['site']['depth']
        self.mass = self.config['equipment']['tech_combined_mass']
        self.area = self.config['equipment']['tech_required_area']

        _platform = self.config.get('equipment', {})
        design_cost = _platform.get('topside_design_cost', 4.5e6)   # USD
        fab_cost = _platform.get('fabrication_cost_rate', 14500.)   # USD/t
        steel_cost = _platform.get('substructure_steel_rate', 3000) # USD/t
        
        install_duration = _platform.get("install_duration", 14)    # days
        
        # Initialize vessel 
        vessel_specs = self.config.get("oss_install_vessel", None)
        name = vessel_specs.get("name","Offshore Substation Install Vessel")

        vessel = orbit.core.Vessel(name, vessel_specs)
        self.env.register(vessel)

        vessel.initialize()
        self.install_vessel = vessel
        
        # Add in the mass of the substructure to total mass (may or may not impact the final install cost)
        _, substructure_mass = calc_substructure_mass_and_cost(self.mass, self.area, 
                        self.depth, fab_cost, design_cost, steel_cost
                        )

        self.total_mass = substructure_mass  # t
         # Call the install_platform function
        self.install_capex = install_platform(self.total_mass, self.area, self.distance, \
                                                   install_duration, self.install_vessel, foundation="fixed")

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
    calc_substructure_mass_and_cost returns the total mass including substructure, topside and equipment.  Also returns the cost of the substructure and topside
    Inputs: mass            | Mass of equipment on platform (tonnes)
            area            | Area needed for equipment (meter^2) (not necessary)
            depth           | Ocean depth at platform location (meters) (not necessary)
            fab_cost_rate   | Cost rate to fabricate topside (USD/tonne)
            design_cost     | Design cost to design structural components (USD) from ORBIT
            sub_cost_rate   | Steel cost rate (USD/tonne) from ORBIT'''
    '''
    Platform is substructure and topside combined
    All functions are based off NREL's ORBIT [1] (oss_design.py)
    default values are specified in [3], 
    '''
    #Inputs needed
    topside_mass = mass
    topside_fab_cost_rate   =   fab_cost   
    topside_design_cost     =   design_cost

    '''Topside Cost & Mass (repurposed eq. 2.26 from [2])
    Topside Mass is the required Mass the platform will hold
    Topside Cost is a function of topside mass, fab cost and design cost'''
    topside_cost   =   topside_mass*topside_fab_cost_rate + topside_design_cost

    '''Substructure (repurposed eq. 2.31-2.33 from [2])
    Substructure Mass is a function of the topside mass
    Substructure Cost is a function of of substructure mass pile mass and cost rates for each'''

    #inputs needed
    substructure_cost_rate  =   sub_cost        # USD/t
    pile_cost_rate          =   pile_cost       # USD/t

    substructure_mass       =   0.4*topside_mass        # t
    substructure_pile_mass  =   8*substructure_mass**0.5574   # t
    substructure_cost  =   (substructure_mass*substructure_cost_rate +
        substructure_pile_mass*pile_cost_rate)     # USD
        
    substructure_total_mass  =   substructure_mass + substructure_pile_mass    # t

    '''Total Platform capex = capex Topside + capex substructure'''
    
    platform_capex  = substructure_cost + topside_cost  # USD
    platform_mass   = substructure_total_mass + topside_mass    # t
    
    return platform_capex, platform_mass

# Standalone test sections 
if __name__ == '__main__':
    print("\n*** New FixedPlatform Standalone test section ***\n")

    orbit_libpath = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir, 'ORBIT', 'library'))
    print(orbit_libpath)
    orbit.core.library.initialize_library(orbit_libpath)

    config_path = os.path.abspath(__file__)
    config_fname = orbit.load_config(os.path.join(config_path, os.pardir, "example_fixed_project.yaml"))

    orbit.ProjectManager.register_design_phase(FixedPlatformDesign)

    orbit.ProjectManager.register_install_phase(FixedPlatformInstallation)

    platform = orbit.ProjectManager(config_fname)
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
