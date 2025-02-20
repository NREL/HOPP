from scipy.constants import R, g, convert_temperature
import numpy as np

def calculate_air_density_for_elevation(elevation_m: float) -> float:
    """
    Calculate air density based on site elevation using the Barometric formula.
    
    This function is based on Equation 1 from: https://en.wikipedia.org/wiki/Barometric_formula#Density_equations
    
    Args:
        elevation_m (float): Elevation of site in meters

    Returns:
        float: Air density in kg/m^3 at elevation of site
    """
    rho0 = 1.225  # Air density at sea level (kg/m3)
    t_ref = 20  # Standard air temperature (Celsius)
    elevation_sea_level = 0.0  # Reference elevation at sea level (m)
    l = 0.0065  # Temperature lapse rate (K/m) for 0-11000m above sea level
    molar_mass_air = 28.96  # Molar mass of air (g/mol)
    
    # Convert temperature to Kelvin
    T_ref = convert_temperature([t_ref], "C", "K")[0]
    
    # Exponent value used in equation below
    e = g * (molar_mass_air / 1e3) / (R * l)
    # g: acceleration due to gravity (m/s2)
    # R: universal gas constant (J/mol-K)
    
    # Calculate air density at site elevation
    rho = rho0 * ((T_ref - ((elevation_m - elevation_sea_level) * l)) / T_ref) ** (e - 1)
    return rho

def calculate_elevation_air_density_losses(elevation_m: float) -> float:
    """Calculate loss (%) from air density drop at site elevation.

    Args:
        elevation_m (float): site elevation in meters

    Returns:
        float: percentage loss associated with air density decrease at elevation.
    """
    if elevation_m <= 0.0:
        loss_percent = 0.0
    else:
        rho0 = 1.225
        air_density = calculate_air_density_for_elevation(elevation_m)
        loss_ratio = 1 - (air_density / rho0)
        loss_percent = loss_ratio * 100

    return loss_percent

def parse_resource_data(wind_resource):
    """Parse wind resource data into floris-friendly format.
    Average wind speed and wind direction if there's data for 
    2 resource heights. This method assumes that the turbine hub-height 
    is in-between two resource heights.

    In ``wind_resource.data['fields']``, values correspond to:
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
    
    # Get indices of wind speed data
    idx_ws = [ii for ii, field in enumerate(wind_resource.data['fields']) if field == 3]
    
    # Get indices of wind direction data
    idx_wd = [ii for ii, field in enumerate(wind_resource.data['fields']) if field == 4]
    
    # If there's multiple hub-heights - average the data
    if len(idx_ws) > 1:
        hh1, hh2 = np.unique(wind_resource.data['heights'])
        
        if hh1 == wind_resource.hub_height_meters:
            idx_ws1 = [i for i in idx_ws if wind_resource.data['heights'][i] == hh1][0]
            idx_wd1 = [i for i in idx_wd if wind_resource.data['heights'][i] == hh1][0]
            speeds = data[:, idx_ws1]
            wind_dirs = data[:, idx_wd1]

        elif hh2 == wind_resource.hub_height_meters:
            idx_ws2 = [i for i in idx_ws if wind_resource.data['heights'][i] == hh2][0]
            idx_wd2 = [i for i in idx_wd if wind_resource.data['heights'][i] == hh2][0]
            speeds = data[:, idx_ws2]
            wind_dirs = data[:, idx_wd2]
        
        else:
            speeds = data[:, idx_ws].mean(axis=1)
            wind_dirs = data[:, idx_wd].mean(axis=1)
    else:
        # If there's only one hub-height, grab speed and direction data
        speeds = data[:, idx_ws[0]]
        wind_dirs = data[:, idx_wd[0]]

    return speeds, wind_dirs

def weighted_parse_resource_data(wind_resource):
    """Parse wind resource data into floris-friendly format.
    Weighted average wind speed and wind direction if there's data for 
    2 resource heights. Weight wind resource data based on resource-height 
    relative to turbine hub-height. 

    In ``wind_resource.data['fields']``, values correspond to:
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
    
    # Get indices of wind speed data
    idx_ws = [ii for ii, field in enumerate(wind_resource.data['fields']) if field == 3]
    
    # Get indices of wind direction data
    idx_wd = [ii for ii, field in enumerate(wind_resource.data['fields']) if field == 4]
    
    # If there's multiple hub-heights - average the data
    if len(idx_ws) > 1:
        # Weights corresponding to difference of resource height and hub-height
        hh1, hh2 = np.unique(wind_resource.data['heights'])
        weight1 = np.abs(hh1 - wind_resource.hub_height_meters)
        weight2 = np.abs(hh2 - wind_resource.hub_height_meters)
        
        # Wind speed data indices for each resource height
        idx_ws1 = [i for i in idx_ws if wind_resource.data['heights'][i] == hh1][0]
        idx_ws2 = [i for i in idx_ws if wind_resource.data['heights'][i] == hh2][0]
        
        # Wind speeds at the two resource heights
        ws1 = data[:, idx_ws1]
        ws2 = data[:, idx_ws2]

        # Weight wind speed data based on height relative to turbine hub-height
        speeds = np.round(((weight1 * ws1) + (weight2 * ws2)) / (weight1 + weight2), 3)

        # Wind direction data indices for each resource height
        idx_wd1 = [i for i in idx_wd if wind_resource.data['heights'][i] == hh1][0]
        idx_wd2 = [i for i in idx_wd if wind_resource.data['heights'][i] == hh2][0]
        
        # Wind directions at the two resource heights
        wd1 = data[:, idx_wd1]
        wd2 = data[:, idx_wd2]
        
        # Weight wind direction data based on height relative to turbine hub-height
        wind_dirs = np.round(((weight1 * wd1) + (weight2 * wd2)) / (weight1 + weight2), 3)
    else:
        # If there's only one hub-height, grab speed and direction data
        speeds = data[:, idx_ws[0]]
        wind_dirs = data[:, idx_wd[0]]

    return speeds, wind_dirs
