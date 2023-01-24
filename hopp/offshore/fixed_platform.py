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
import sys 

import numpy as np
import pandas as pd
#from construction_finance_param import con_fin_params

#import ORBIT.ProjectManager, load_config
#upthree = '..\..\..\..'
#ORBIT_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), upthree))
#print(ORBIT_DIR)
#sys.path.append(os.path.dirname(ORBIT_DIR))

from ORBIT import ProjectManager, load_config
#import ....ORBIT.ORBIT as orbit
from ORBIT.phases.design import OffshoreSubstationDesign
from ORBIT.phases.install import OffshoreSubstationInstallation

class FixedPlatform: 

    def __init__(self, input_dict, output_dict):
        self.input_dict = input_dict 
        self.output_dict = output_dict

        self.config_fname = input_dict['config_fname']
        self.nturbines = input_dict['nturbines']

        self.site_depth = input_dict['site_depth']
        self.distance = input_dict['distance']

        self.tech_required_area = input_dict['tech_required_area']
        self.tech_combined_mass = input_dict['tech_combined_mass']

    def calc_platform_capex(self):

        # Load the configuration .yaml file
        fixed_config = load_config(self.config_fname)
        
        # Append any of the example_fixed_project.yaml parameters 
        fixed_config['plant']['num_turbines'] = self.nturbines
        fixed_config['site']['depth'] = self.site_depth

        project = ProjectManager(fixed_config)
        project.run()

        dct = project.capex_breakdown
        dct = {k: [v] for k, v in dct.items()}
        
        df = pd.DataFrame.from_dict(dct, orient="columns")

        print(df.T)

        print(project.phases['OffshoreSubstationDesign'])
        print(project.phases['OffshoreSubstationInstallation'])
        
        self.output_dict['vessel'] = fixed_config['OffshoreSubstationInstallation']['feeder']

        # Save the outputs to output_dict or return values
        self.output_dict['SubstationCapEx'] = project.capex_breakdown['Offshore Substation'] \
                                            + project.capex_breakdown['Offshore Substation Installation']
        #self.output_dict['SubstationOpEx'] = self.output_dict['SubstationCapEx'] * 0.05 
        #return project.capex_breakdown['Offshore Substation']

        
        # Test running just the offshore substation design component (missing installation costs)
        # Load the configuration .yaml file
        #fixed_config = load_config(self.config_fname)

        # Append any of the example_fixed_project.yaml parameters 
        #fixed_config["substation_design"]["num_substations"] = 1

        #ossdesign = OffshoreSubstationDesign(fixed_config)
        #ossdesign.run()

        #print("OSS Design Output: ", ossdesign.detailed_output)
        #print("OSS Total Cost: ", ossdesign.total_cost)

        # Save the outputs to output_dict or return values
        #self.output_dict['SubstationCapEx'] = ossdesign.total_cost
        #self.output_dict['SubstationOpEx'] = ossdesign.total_cost * 0.05 
        #return ossdesign.total_cost, ossdesign.total_cost * 0.05

    def calc_platform_opex(self):

        # OpEx calculator placeholder
        self.output_dict['SubstationOpEx'] = self.output_dict['SubstationCapEx'] * 0.05



# Test sections 
if __name__ == '__main__':
    print("\n*** FixedPlatform Testing section ***\n")

    in_dict = dict()
    
    # ORBIT configuration file
    in_dict['config_fname'] = 'example_fixed_project2.yaml'
    in_dict['nturbines'] = 50

    # ORBIT Specific inputs
    in_dict['site_depth'] = 45              # m  
    in_dict['distance'] = 50                # km
    
    # Additional H2 parameters for platform
    in_dict['tech_required_area'] = 300.     # m**2
    in_dict['tech_combined_mass'] = 100.     # tonnes
    

    out_dict = dict()

    fixedplatform_test = FixedPlatform(in_dict, out_dict)
    fixedplatform_test.calc_platform_capex()
    fixedplatform_test.calc_platform_opex()
    

    print("!!! Placeholder values !!!!")
    print("Platform Install Vessel:", out_dict['vessel'])
    print("CapEx of the Fixed Platform: {} USD".format(out_dict['SubstationCapEx']))
    print("OpEx of the Fixed Platform: {} USD".format(out_dict['SubstationOpEx']))


