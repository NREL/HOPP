"""
resource_tools.py
This is a collection of tools for handling resource data used in performing hybrid analysis
Functionality includes:
 Filtering sites by country, whether over land or not.
 Getting timezone offset required for SAM file header
 Converting solar and wind data files to SAM formatted dictionaries (and vice-versa)
 Extrapolating wind speeds
"""

from datetime import datetime
from pytz import timezone, utc
from timezonefinder import TimezoneFinder
from global_land_mask import globe
from shapely.geometry import shape
from shapely.prepared import prep
from shapely.geometry import Point
import requests
import pandas as pd


def get_country(lat, lon, geo_data):
    """
    Determine which country a point lies in
    """
    countries = {}
    for feature in geo_data["features"]:
        geom = feature["geometry"]
        country = feature["properties"]["ADMIN"]
        countries[country] = prep(shape(geom))

    point = Point(lon, lat)
    for country, geom in countries.items():
        if geom.contains(point):
            return country

    return "unknown"


def filter_sites(site_details, location='usa only'):
    """

    :param site_details: pandas dataframe
    :param location: 'on land only' or 'usa only'
    :return:
    """

    # Load GEOJson
    geo_data = requests.get(
        "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson").json()

    # Creates a new dataframe to contain only the selected sites
    #  Only sites on land (includes lakes)
    if location == 'on land only':
        on_land = []
        for site_index, site in site_details.iterrows():
            is_on_land = globe.is_land(site['Lat'], site['Lon'])
            on_land.append(is_on_land)
        site_details['on_land'] = on_land
        site_details_selected = site_details[site_details['on_land'] == True]

    #  Only sites in the Continental US
    if location == 'usa only':
        in_usa = []
        for site_index, site in site_details.iterrows():
            is_in_usa = (get_country(site['Lat'], site['Lon'], geo_data=geo_data) == 'United States of America')
            in_usa.append(is_in_usa)
        site_details['in_usa'] = in_usa 
        site_details_selected = site_details[site_details['in_usa'] == True]

    return site_details_selected


def get_offset(lat, long):
    """
    returns the timezone offset for a given lat/long
    :param lat:
    :param long:
    :return:
    """
    today = datetime.now()
    tf = TimezoneFinder()
    tz_target = timezone(tf.timezone_at(lng=long, lat=lat))
    if not tz_target:
        raise ValueError("tz_target error")
    today_target = tz_target.localize(today)
    today_utc = utc.localize(today)
    return int((today_utc - today_target).total_seconds() / 3600)


def extrapolate_wind_speed(height_in, height_out, wind_speed):
    """

    :param height_in: m, Height data was recorded at
    :param height_out: m, Height desired
    :param wind_speed: m/s, Array of wind speeds recorded at height_in
    :return:
    """
    shear_exponent = .144444
    extrapolated_wind_speed = [x * ((height_out/height_in) ** shear_exponent) for x in wind_speed]
    return extrapolated_wind_speed
