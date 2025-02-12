from scipy.constants import R, g, convert_temperature
import numpy as np

def calculate_air_density_for_elevation(elevation_m:float):
    """calculate air density based on site elevation using the Barometric formula.
    this function is based on Equation 1 from: https://en.wikipedia.org/wiki/Barometric_formula#Density_equations
    
    Args:
        elevation_m (float): elevation of site in meters

    Returns:
        float: air density in kg/m3 at elevation of site
    """
    rho0 = 1.225 # air density at sea level (kg/m3)
    t_ref = 20 # standard air temperature (Celsius)
    elevation_sea_level = 0.0 #reference elevation at sea level (m)
    l = 0.0065 # temperature lapse null rate (K/m) for 0-11000m above sea level
    molar_mass_air = 28.96 # molar mass of air (g/mol)
    
    # convert temperature to Kelvin
    T_ref = convert_temperature([t_ref], "C", "K")[0] 
    
    # exponent value used in equation below
    e = g*(molar_mass_air/1e3)/(R*l) 
    # g: acceleration due to gravity (m/s2)
    # R: universal gas constant (J/mol-K)
    
    # calculate air density at site elevation
    rho = rho0*((T_ref - ((elevation_m-elevation_sea_level)*l))/T_ref)**(e - 1)
    return rho

def calculate_elevation_air_density_losses(elevation_m:float):
    """Calculate loss (%) from air density drop at site elevation.

    Args:
        elevation_m (float): site elevation in meters

    Returns:
        float: percentage loss associated with air density decrease at elevation.
    """

    if elevation_m<=0.0:
        loss_percent = 0.0
    else:
        rho0 = 1.225
        air_density = calculate_air_density_for_elevation(elevation_m)
        loss_ratio = 1 - (air_density/rho0)
        loss_percent = loss_ratio*100

    return loss_percent

def parse_resource_data(wind_resource):
    """parse wind resource data into floris-friendly format.
        average wind speed and wind direction if theres data for 
        2 resource heights. this method assumes that the turbine hub-height 
        is in-between two resource heights.

        in ``wind_resource.data['fields']``, values correspond to:
            - 3: Wind speed in meters per second (m/s)
            - 4: Wind direction in degrees east of north (degrees).

    Args:
        wind_resource (HPCWindData | WindResource): wind resource data object

    Returns:
        2-element tuple containing

        - **speeds** (:obj:`numpy.ndarray`): wind speed in m/s
        - **wind_dirs** (:obj:`numpy.ndarray`): wind direction in deg from North (clockwise)
    """

    data = np.array(wind_resource.data['data'])
    
    # get indices of wind speed data
    idx_ws = [ii for ii,field in enumerate(wind_resource.data['fields']) if field == 3]
    
    # get indices of wind direction data
    idx_wd = [ii for ii,field in enumerate(wind_resource.data['fields']) if field == 4]
    
    # if theres multiple hub-heights - average the data
    if len(idx_ws) > 1:
        speeds = data[:,idx_ws].mean(axis = 1)
        wind_dirs = data[:,idx_wd].mean(axis = 1)
    
    # if theres only one hub-height, grab speed and direction data
    else:
        speeds = data[:,idx_ws[0]]
        wind_dirs = data[:,idx_wd[0]]

    return speeds, wind_dirs

def weighted_parse_resource_data(wind_resource):
    """parse wind resource data into floris-friendly format.
        weighted average wind speed and wind direction if theres data for 
        2 resource heights. weight wind resource data based on resource-height 
        relative to turbine hub-height. 
        
        in ``wind_resource.data['fields']``, values correspond to:
            - 3: Wind speed in meters per second (m/s)
            - 4: Wind direction in degrees east of north (degrees).


    Args:
        wind_resource (HPCWindData | WindResource): wind resource data object

    Returns:
        2-element tuple containing

        - **speeds** (:obj:`numpy.ndarray`): wind speed in m/s
        - **wind_dirs** (:obj:`numpy.ndarray`): wind direction in deg from North (clockwise)
    """

    data = np.array(wind_resource.data['data'])
    # get indices of wind speed data
    idx_ws = [ii for ii,field in enumerate(wind_resource.data['fields']) if field == 3]
    # get indices of wind direction data
    idx_wd = [ii for ii,field in enumerate(wind_resource.data['fields']) if field == 4]
    
    # if theres multiple hub-heights - average the data
    if len(idx_ws)>1:
        # weights corresponding to difference of resource height and hub-height
        hh1,hh2 = np.unique(wind_resource.data['heights'])
        weight1 = np.abs(hh1 - wind_resource.hub_height_meters)
        weight2 = np.abs(hh2 - wind_resource.hub_height_meters)
        
        # wind speed data indices for each resource height
        idx_ws1 = [i for i in idx_ws if wind_resource.data['heights'][i] == hh1][0]
        idx_ws2 = [i for i in idx_ws if wind_resource.data['heights'][i] == hh2][0]
        
        # wind speeds at the two resource heights
        ws1 = data[:,idx_ws1]
        ws2 = data[:,idx_ws2]

        # weight wind speed data based on height relative to turbine hub-height
        speeds = np.round(((weight1 * ws1) + (weight2 * ws2)) / (weight1 + weight2),3)

        # wind direction data indices for each resource height
        idx_wd1 = [i for i in idx_wd if wind_resource.data['heights'][i] == hh1][0]
        idx_wd2 = [i for i in idx_wd if wind_resource.data['heights'][i] == hh2][0]
        
        # wind directions at the two resource heights
        wd1 = data[:,idx_wd1]
        wd2 = data[:,idx_wd2]
        
        # weight wind direction data based on height relative to turbine hub-height
        wind_dirs = np.round(((weight1 * wd1) + (weight2 * wd2)) / (weight1 + weight2),3)
    
    # if theres only one hub-height, grab speed and direction data
    else:
        speeds = data[:,idx_ws[0]]
        wind_dirs = data[:,idx_wd[0]]
        

    return speeds, wind_dirs