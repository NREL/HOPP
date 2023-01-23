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

import numpy as np
import pandas as pd
#from construction_finance_param import con_fin_params

from ORBIT import ProjectManager, load_config

class FixedPlatform: 

    def __init__(self, input_dict, output_dict):
        self.input_dict = input_dict 
        self.output_dict = output_dict

        self.config_fname = input_dict['config_fname']

    def runOrbit(self):

        print("current directory:", os.getcwd())
        fixed_config = load_config(self.config_fname)

        self.output_dict['vessel'] = fixed_config['OffshoreSubstationInstallation']['feeder']
        
        #project = pm.ProjectManager(fixed_config)


# Test sections 
if __name__ == '__main__':
    print("FixedPlatform Testing section")
    in_dict = dict()
    in_dict['config_fname'] = 'example_fixed_project.yaml'

    out_dict = dict()

    fixedplatform_test = FixedPlatform(in_dict,out_dict)
    fixedplatform_test.runOrbit()

    
    print("Feeder vessel:", out_dict['vessel'])


