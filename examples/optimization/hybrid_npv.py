from pathlib import Path
from hybrid.sites import make_circular_site, make_irregular_site, SiteInfo, locations

site = 'irregular'
location = locations[1]
site_data = None

if site == 'circular':
    site_data = make_circular_site(lat=location[0], lon=location[1], elev=location[2])
elif site == 'irregular':
    site_data = make_irregular_site(lat=location[0], lon=location[1], elev=location[2])
else:
    raise Exception("Unknown site '" + site + "'")

g_file = Path(__file__).parent.parent.parent / "resource_files" / "grid" / "pricing-data-2015-IronMtn-002_factors.csv"

site_info = SiteInfo(site_data, grid_resource_file=g_file)

print(site_info.elec_prices.data)
