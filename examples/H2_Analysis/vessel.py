## Basic Vessel Model referenced from ORBIT's installation vessels 

# Combine Floating Barges + Pressure Vessels + Towing vessels

import math
import numpy as np
from vessel import *

class VesselTransport(): 
    def __init__(self, input_dict, output_dict):
        self.input_dict = input_dict
        self.output_dict = output_dict

        # inputs
        self.annual_capacity_kg = input_dict['h2_output_kg_per_year']
            
        # assumptions
        self.vessel_capacity_factor = 0.70
        
        # Floating barge from ORBIT/library/vessel/floating_barge.yaml
        # Towing vessels from ORBIT/library/vessel/example_towing_vessel.yaml  
        self.vessel_speed = 6 # km/hr
        self.barge_daily_cost = 120000.0 # USD/day
        self.barge_capacity_ton = 8000.0 # tonnes 
        self.tug_daily_cost = 30000.0 # USD/day 
        self.tug_capacity_ton = 4000.0 # tonnes 
        
        # Portage fees 
        self.portage_connection_fee = 20000.0 # Guess, each time you dock 
        self.portage_docking_fee = 10000.0 # Guess, per day 

    def calcNumberTrips(self):
        tons_to_kg = 1000.0 # mton to kg
        rho = 0.08375 # STP H2 density, kg/m**3

        # Calculate vessel capacity in kg
        vessel_cap_kg = self.barge_capacity_ton * self.vessel_capacity_factor * tons_to_kg
    
        num_trips = math.ceil(self.annual_capacity_kg / vessel_cap_kg)
        #vessel_h2_cap_vol = vessel_h2_cap_kg / rho
        
        self.output_dict['number_of_trips'] = num_trips

        return num_trips 

    def capitalCost(self):
        vessel_cost = 359000000.0 # Frontier Suiso for LH2 shipping 
        
        self.output_dict['vessel_cost'] = vessel_cost 
        
        return vessel_cost 

    def operatingCosts(self):
        # At Sea section
        # Assume 1 trip takes all day 
        number_of_trips = self.calcNumberTrips()

        barge_total_cost = number_of_trips * self.barge_daily_cost 
        tug_total_cost = number_of_trips * self.tug_daily_cost

        total_vessel_cost = barge_total_cost + 2 * tug_total_cost 
        
        #self.output_dict['annual_operating_cost_vessels'] = total_vessel_cost

        # At Port section
        # Assume days docked = 365 - days at sea (1 trip)
        days_docked = 365 - number_of_trips

        total_docking_fees = self.portage_docking_fee * days_docked
        total_connection_fees = self.portage_connection_fee * number_of_trips 

        total_portage_cost = total_docking_fees + total_connection_fees 

        #self.output_dict['annual_operating_cost_portage'] = total_portage_cost
        total_operating_cost = total_vessel_cost + total_portage_cost 
        self.output_dict['annual_vessel_transport_cost'] = total_operating_cost

        return total_operating_cost

# Test section
if __name__ == '__main__': 
    print('### vessel test ###')
    
    in_dict = dict()
    in_dict['h2_output_kg_per_year'] = 57909304.0

    out_dict = dict()

    test = VesselTransport(in_dict, out_dict)
    test.calcNumberTrips()
    test.capitalCost()
    test.operatingCosts()

    print("{} trips to transport {} kg of H2".format(out_dict['number_of_trips'],in_dict['h2_output_kg_per_year']))
    print("Total Cost of Suiso LH2 Vessel ($USD): {}".format(out_dict['vessel_cost']))
    print("Total Operating Cost ($USD): {}".format(out_dict['annual_vessel_transport_cost']))

