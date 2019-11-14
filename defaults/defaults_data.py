import os
from collections import OrderedDict

from defaults.genericsystem_singleowner import genericsystem_genericsystemsingleowner, singleowner_genericsystemsingleowner
from defaults.geothermal_singleowner import geothermal_singleowner, singleowner_geothermal
from defaults.pv_singleowner import PV_pvsingleowner, Singleowner_pvsingleowner
from defaults.wind_singleowner import wind_windsingleowner, singleowner_windsingleowner
from defaults.grid_none import grid_none
from defaults.battery_singleowner import battery_singleowner
from hybrid.resource import SolarResource, WindResource

path_current = os.path.dirname(os.path.abspath(__file__))
path_resource = os.path.join(path_current, '..', 'resource_files')


def update_site_default(default, site):
    """
    Update site data from the default values

    Function to take in a default dictionary and site dictionary, and update the path names in the default dictionary
    and possibly download the resource data if not present.

    Parameters:
    -----------
    default : dict
        dictionary of default values for each technology being evaluated
    site: dict
        dictionary of site data

    Returns:
    --------
    default: dict
        updated defaults
    """
    if 'Solar' in default:
        sr = SolarResource(lat=site['lat'], lon=site['lon'], year=site['year'], download=True)
        default['Solar']['Pvsamv1']['SolarResource']['solar_resource_file'] = sr.filename
    if 'Wind' in default:
        height = default['Wind']['Windpower']['Turbine']['wind_turbine_hub_ht']
        file = default['Wind']['Windpower']['Resource']['wind_resource_filename']
        if not os.path.isfile(file):
            wr = WindResource(lat=site['lat'], lon=site['lon'], year=site['year'], wind_turbine_hub_ht=height, download=True)
            default['Wind']['Windpower']['Resource']['wind_resource_filename'] = wr.filename
    if 'Geothermal' in default:
        sr = SolarResource(lat=site['lat'], lon=site['lon'], year=site['year'], download=True)
        default['Geothermal']['Geothermal']['GeoHourly']['file_name'] = sr.filename
    return default

"""
Dictionary to manage default Site
"""
Site = {
    "lat": 35.2018863,
    "lon": -101.945027,
    "elev": 1099,
    "year": 2012,
    "tz": -6,
    'site_boundaries': {
        'verts': [[3.0599999999976717, 288.87000000011176],
                    [0.0, 1084.0300000002608],
                    [1784.0499999999884, 1084.2400000002235],
                    [1794.0900000000256, 999.6399999996647],
                    [1494.3400000000256, 950.9699999997392],
                    [712.640000000014, 262.79999999981374],
                    [1216.9800000000396, 272.3600000003353],
                    [1217.7600000000093, 151.62000000011176],
                    [708.140000000014, 0.0]],
        'verts_simple': [[3.0599999999976717, 288.87000000011176],
                        [0.0, 1084.0300000002608],
                        [1784.0499999999884, 1084.2400000002235],
                        [1794.0900000000256, 999.6399999996647],
                        [1216.9800000000396, 272.3600000003353],
                        [1217.7600000000093, 151.62000000011176],
                        [708.140000000014, 0.0]]
    }
}

"""
Dictionary to manage default data.
Note, the paths of resource data will be updated to correspond to the location in the "Site" dictionary
"""

defaults_all = {
    'Geothermal':{
        'Geothermal': geothermal_singleowner,
        'Singleowner': singleowner_geothermal
    },
    'Solar': {
        'Pvsamv1': PV_pvsingleowner,
        'Singleowner': Singleowner_pvsingleowner
    },
    'Wind': {
        'Windpower': wind_windsingleowner,
        'Singleowner': singleowner_windsingleowner
    },
    'Grid': {
        'Grid' : grid_none,
        'Singleowner': singleowner_genericsystemsingleowner
    },
    'Generic': {
        'GenericSystem': genericsystem_genericsystemsingleowner,
        'Singleowner': singleowner_genericsystemsingleowner
    },
    'Battery': {
        'StandAloneBattery': battery_singleowner,
        'Singleowner': singleowner_genericsystemsingleowner
    }
}

# Update defaults to correspond to specified site location
defaults_all = update_site_default(defaults_all, Site)

def get_default(technologies, defaults=None):
    """
    Return the defaults for each technology specified as well as the default Site

    Parameters
    ----------
    technologies : list
        list of technologies, e.g ['Wind', 'Solar', 'Generic', 'Grid']
    defaults: dict (optional)
        optional dictionary of current defaults to update

    Returns
    -------
    default: dict
        dictionary of updated defaults
    Site: dict
        dictionary of site parameters
    """
    if defaults is None:
        default = dict()
    for technology in technologies:
        dict_tech = OrderedDict()
        dict_tech[technology] = defaults_all[technology]
        default.update(dict_tech)

    return default, Site

def setup_defaults(technologies=None):
    """
    Return default configurations and the default Site given a list of technology options to consider


    Parameters:
    ----------
    technologies : list
      list of technologies to run, e.g ['Wind', 'Solar', 'Generic']

    Returns:
    --------
    technologies : list
        list of technologies selected, e.g ['Wind', 'Solar', 'Generic']
    defaults : dict
        nested dictionary of technology defaults, e.g {'Wind': {...}, 'Solar': {...}, 'Generic': {...}}
    site: dict
        dictionary of site related information, e.g {'lat': 39.0, 'lon': -104.34, ...}
    """

    if technologies is None:
        technologies = ['Solar', 'Wind', 'Geothermal', 'Generic', 'Battery', 'Grid']
    defaults_setup, site = get_default(technologies)

    return technologies, defaults_setup, site