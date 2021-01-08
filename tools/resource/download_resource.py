from hybrid.resource import WindResource, SolarResource

# California Site
# lat = 33.907295
# lon = -116.555588
# year = 2012
# hubheight = 80

# Texas Site
# lat = 34.212257
# lon = -101.361160
# year = 2012
# hubheight = 91.5

# Maine Site
lat = 44.766171
lon = -68.157669
year = 2012
hubheight = 116.5


WindResource(lat=lat, lon=lon, year=year, wind_turbine_hub_ht=hubheight)

SolarResource(lat=lat, lon=lon, year=year)
