import math

def calc_platform_opex(capex, opex_rate=0.011):
    '''
    Simple opex calculation based on a capex
        https://www.acm.nl/sites/default/files/documents/study-on-estimation-method-for-additional-efficient-offshore-grid-opex.pdf
    
    Output in $USD/year
    '''
    
    opex = capex * opex_rate    # USD/year
    
    return opex

def install_platform(mass, area, distance, install_duration=14, vessel=None, foundation="fixed"):
    '''
    A simplified platform installation costing model. 
    Total Cost = install_cost * duration 
         Compares the mass and/or deck space of equipment to the vessel limits to determine 
         the number of trips. Add an additional "at sea" install duration
          
    '''
    
    # If no ORBIT vessel is defined set default values (based on ORBIT's floating_heavy_lift_vessel)
    if vessel == None:
        if foundation=="fixed":
            # If no ORBIT vessel is defined set default values (based on ORBIT's example_heavy_lift_vessel)
            # Default values are from [3].
            vessel_cargo_mass = 7999 # t
            vessel_deck_space = 3999 # m**2 
            vessel_day_rate = 500001 # USD/day 
            vessel_speed = 5 # km/hr 
        elif foundation=="floating":
            # If no ORBIT vessel is defined set default values (based on ORBIT's floating_heavy_lift_vessel)
            vessel_cargo_mass = 7999 # t
            vessel_deck_space = 3999 # m**2 
            vessel_day_rate = 500001 # USD/day 
            vessel_speed = 7 # km/hr 
        else:
            raise(ValueError("Invalid offshore platform foundation type. Must be one of ['fixed', 'floating']"))
    else:
        vessel_cargo_mass = vessel.storage.max_cargo_mass # t
        vessel_deck_space = vessel.storage.max_deck_space # m**2 
        vessel_day_rate = vessel.day_rate # USD/day 
        vessel_speed = vessel.transit_speed # km/hr 

    # Get the # of trips based on ships cargo/space limits 
    num_of_trips = math.ceil(max((mass / vessel_cargo_mass), (area / vessel_deck_space)))

    # Total duration = double the trips + install_duration
    duration = (2 * num_of_trips * distance) / (vessel_speed * 24) + install_duration # days\

    # Final install cost is obtained by using the vessel's daily rate 
    install_cost = vessel_day_rate * duration   # USD

    return install_cost