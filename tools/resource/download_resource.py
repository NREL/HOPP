from hybrid.resource import WindResource, SolarResource
import os
from dotenv import load_dotenv
from hybrid.keys import set_developer_nrel_gov_key

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

load_dotenv()
NREL_API_KEY = os.getenv("NREL_API_KEY")
set_developer_nrel_gov_key(NREL_API_KEY)  # Set this key manually here if you are not setting it using the .env


WindResource(lat=lat, lon=lon, year=year, wind_turbine_hub_ht=hubheight)

SolarResource(lat=lat, lon=lon, year=year)
