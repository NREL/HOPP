# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 20:08:09 2022
@author: ppeng

Revisions:
- 20221118:
    Author: Jared J. Thomas
    Description: 
        - Reformatted to be a class
"""

"""
Model Revision Needed: storage space and mass
Description: This file should handle physical size (footprint and mass) needed for pressure vessel storage
Sources:
    - [1] ./README.md and other elements in this directory
Args:
    - same as for the physics and cost model contained herein
    - others may be added as needed
Returns:(can be from separate functions and/or methods as it makes sense):
    - mass_empty (float): mass (approximate) for pressure vessel storage components ignoring stored H2
    - footprint (float): area required for pressure vessel storage
    - others may be added as needed
"""

# package imports
import os
import numpy as np

# local imports
# from Compressed_gas_function import CompressedGasFunction
from .Compressed_gas_function import CompressedGasFunction

class PressureVessel():
    def __init__(self, Wind_avai=80, H2_flow=200, cdratio=1, Energy_cost=0.07, cycle_number=1, parent_path=os.path.abspath(os.path.dirname(__file__)), spread_sheet_name="Tankinator.xlsx", verbose=False):

        ########Key inputs##########
        self.Wind_avai = Wind_avai  #Wind availability in %
        self.H2_flow = H2_flow  #Flow rate of steel plants in tonne/day
        self.cdratio = cdratio  #Charge/discharge ratio, for example 2 means the charging is 2x faster than discharge
        self.Energy_cost = Energy_cost  #Renewable energy cost in $/kWh

        #######Other inputs########
        self.cycle_number = cycle_number #Equivalent cycle number for a year, only affects operation (the higher the number is the less effect there will be), set as now as I am not sure how the maximum sotrage capacity is determined and how the storage will be cycled

        self.compressed_gas_function = CompressedGasFunction(path_tankinator=os.path.join(parent_path, spread_sheet_name))
        self.compressed_gas_function.verbose = verbose
        
    def run(self):
        #####Run calculation########
        self.compressed_gas_function.func(Wind_avai=self.Wind_avai, H2_flow=self.H2_flow, cdratio=self.cdratio, Energy_cost=self.Energy_cost, cycle_number=self.cycle_number)

        ########Outputs################

        ######Maximum equivalent storage capacity and duration
        self.capacity_max = self.compressed_gas_function.capacity_max   #This is the maximum equivalent H2 storage in kg
        self.t_discharge_hr_max = self.compressed_gas_function.t_discharge_hr_max   #This is tha maximum storage duration in kg

        ###Parameters for capital cost fitting for optimizing capital cost
        self.a_fit_capex = self.compressed_gas_function.a_cap_fit
        self.b_fit_capex = self.compressed_gas_function.b_cap_fit
        self.c_fit_capex = self.compressed_gas_function.c_cap_fit

        #Parameters for operational cost fitting for optimizing capital cost
        self.a_fit_opex = self.compressed_gas_function.a_op_fit
        self.b_fit_opex = self.compressed_gas_function.b_op_fit
        self.c_fit_opex = self.compressed_gas_function.c_op_fit

    def calculate_from_fit(self, capacity_kg):
        capex_per_kg = self.compressed_gas_function.exp_log_fit([self.a_fit_capex, self.b_fit_capex, self.c_fit_capex], capacity_kg) 
        opex_per_kg = self.compressed_gas_function.exp_log_fit([self.a_fit_opex, self.b_fit_opex, self.c_fit_opex], capacity_kg) 
        energy_per_kg_h2 = self.compressed_gas_function.energy_function(capacity_kg)/capacity_kg

        # NOTE ON ENERGY: the energy value returned here is the energy used to fill the 
        # tanks initially for the first fill and so can be used as an approximation for the energy used on a per kg basis. 
        # If cycle_number > 1, the energy model output is incorrect.

        capex = capex_per_kg*capacity_kg
        opex = opex_per_kg*capacity_kg
        return capex, opex, energy_per_kg_h2

    def get_tanks(self, capacity_kg):
        """ gets the number of tanks necessary """
        return np.ceil(capacity_kg/self.compressed_gas_function.m_H2_tank)

    def get_tank_footprint(self, capacity_kg,
                           upright : bool = True,
                           custom_packing : bool = False,
                           packing_ratio : float = None):
        """
        gets the footprint required for the H2 tanks

        assumes that packing is square (unless custom_packing is true)
        - diameter D upright tank occupies D^2
        - diameter D, length L tank occupies D*L

        parameters:
            - `upright`: place tanks vertically (default yes)?
            - `custom_packing`: pack tanks at an alternate packing fraction?
            - `packing_ratio`: ratio for custom packing, defaults to theoretical max (if known)
        returns:
            - `tank_footprint`: footprint of each tank in m^2
            - `array_footprint`: total footprint of all tanks in m^2
        """

        tank_radius= self.compressed_gas_function.Router/100
        tank_length= self.compressed_gas_function.Louter/100
        Ntank= self.get_tanks(capacity_kg= capacity_kg)

        if upright:
            tank_area= np.pi*tank_radius**2
            tank_footprint= 4*tank_radius**2
        else:
            tank_area= np.pi*tank_radius**2*((tank_length - 2*tank_radius)*(2*tank_radius))
            tank_footprint= tank_radius*tank_length

        if custom_packing:
            if upright:
                if packing_ratio is None: packing_ratio= np.pi*np.sqrt(3.)/6. # default to tight packing
                tank_footprint= tank_area*packing_ratio
            else:
                if packing_ratio is None:
                    raise NotImplementedError("tight packing ratio for cylinders isn't derived yet")
                tank_footprint= tank_area*packing_ratio

        return (tank_footprint, Ntank*tank_footprint)
    
    def get_tank_mass(self, capacity_kg):
        """
        gets the mass required for the H2 tanks

        returns
            - `tank_mass`: mass of each tank
            - `array_mass`: total mass of all tanks
        """

        tank_mass = self.compressed_gas_function.Mempty_tank
        Ntank = self.get_tanks(capacity_kg = capacity_kg)

        return (tank_mass, Ntank*tank_mass)

    def plot(self):
        self.compressed_gas_function.plot()

    def distributed_storage_vessels(self, capacity_total_tgt, N_sites):
        """
        compute modified pressure vessel storage requirements for distributed
        pressure vessels

        parameters:
            - capacity_total_tgt: target gaseous H2 capacity in kilograms
            - N_sites: number of sites (e.g. turbines) where pressure vessels will be placed

        returns:
            - 
        """

        # assume that the total target capacity is equally distributed across sites
        capacity_site_tgt= capacity_total_tgt/N_sites

        # capex_centralized_total, opex_centralized_total, energy_kg_centralized_total= self.calculate_from_fit(capacity_total_tgt)
        capex_site, opex_site, energy_kg_site= self.calculate_from_fit(capacity_site_tgt)

        # get the resulting capex & opex costs, incl. equivalent
        capex_distributed_total= N_sites*capex_site # the cost for the total distributed storage facilities
        opex_distributed_total= N_sites*opex_site # the cost for the total distributed storage facilities

        # get footprint stuff
        area_footprint_site= self.get_tank_footprint(capacity_site_tgt)[1]
        mass_tank_empty_site= self.get_tank_mass(capacity_site_tgt)[1]

        # return the outputs
        return capex_distributed_total, opex_distributed_total, energy_kg_site, \
                area_footprint_site, mass_tank_empty_site, capacity_site_tgt

if __name__ == "__main__":
    storage = PressureVessel()
    storage.run()

    capacity_req= 1e3
    print("tank type:", storage.compressed_gas_function.tank_type)
    print("tank mass:", storage.get_tank_mass(capacity_req)[0])
    print("tank radius:", storage.compressed_gas_function.Router)
    print("tank length:", storage.compressed_gas_function.Louter)
    print("tank footprint (upright):", storage.get_tank_footprint(capacity_req, upright= True)[0])
    print("tank footprint (flat):", storage.get_tank_footprint(capacity_req, upright= False)[0])
    
    print("\nnumber of tanks req'd:",
          storage.get_tanks(capacity_req))
    print("total footprint (upright):", storage.get_tank_footprint(capacity_req, upright= True)[1])
    print("total footprint (flat):", storage.get_tank_footprint(capacity_req, upright= False)[1])
    print("total mass:", storage.get_tank_mass(capacity_req)[1])
