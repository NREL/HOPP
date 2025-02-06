from scipy.constants import R, g, convert_temperature
from hopp.simulation.technologies.resource import HPCWindData, WindResource
from typing import Optional, Tuple, Union
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

    if elevation_m<0.0:
        loss_percent = 0.0
    else:
        rho0 = 1.225
        air_density = calculate_air_density_for_elevation(elevation_m)
        loss_ratio = 1 - (air_density/rho0)
        loss_percent = loss_ratio*100

    return loss_percent

def parse_resource_data(wind_resource):
    """parse wind resource data into floris-friendly format.

    Args:
        wind_resource (HPCWindData | WindResource): wind resource data object

    Returns:
        2-element tuple containing

        - **speeds** (:obj:`numpy.ndarray`): wind speed in m/s
        - **wind_dirs** (:obj:`numpy.ndarray`): wind direction in deg from North (clockwise)
    """

    speeds = np.zeros(len(wind_resource.data['data']))
    wind_dirs = np.zeros(len(wind_resource.data['data']))
    data_rows_total = 4
    # if theres multiple hub-heights - average the data
    # TODO: weight data entries based on height relative to turbine hub-height
    # this method assumes that the turbine hub-height is in-between two resource heights
    if np.shape(wind_resource.data['data'])[1] > data_rows_total:
        height_entries = int(np.round(np.shape(wind_resource.data['data'])[1]/data_rows_total))
        data_entries = np.empty((height_entries))
        for j in range(height_entries):
            data_entries[j] = int(j*data_rows_total)
        data_entries = data_entries.astype(int)
        for i in range((len(wind_resource.data['data']))):
            data_array = np.array(wind_resource.data['data'][i])
            speeds[i] = np.mean(data_array[2+data_entries])
            wind_dirs[i] = np.mean(data_array[3+data_entries])
    # if theres only one hub-height, grab speed and direction data
    else:
        for i in range((len(wind_resource.data['data']))):
            speeds[i] = wind_resource.data['data'][i][2]
            wind_dirs[i] = wind_resource.data['data'][i][3]

    return speeds, wind_dirs