from pathlib import Path
import json
from itertools import product
import multiprocessing as mp
import sys
sys.path.append(str(Path(__file__).parent.parent))
from hybrid.sites import SiteInfo, flatirons_site
from hybrid.hybrid_simulation import HybridSimulation
# from hybrid.keys import set_nrel_key_dot_env
# Set API key
# set_nrel_key_dot_env()

examples_dir = Path(__file__).parent

fin_info = json.load(open(examples_dir / "default_financial_parameters.json", 'r'))
cost_info = fin_info['capex']

# Get resource
solar_file = examples_dir.parent / "resource_files" / "solar" / "Coya_Solargis_TEMP/SG-88910-2007-1-1_TMY_P90_SAM.csv"
wind_file = examples_dir.parent / "resource_files" / "wind" / "35.2018863_-101.945027_windtoolkit_2012_60min_80m.srw"
prices_file = examples_dir.parent / "resource_files" / "grid" / "cmg_typical_day.csv"
site = SiteInfo(flatirons_site,
                solar_resource_file=solar_file,
                wind_resource_file=wind_file,
                grid_resource_file=prices_file)

solar_sizes = range(50, 501, 50)
wind_sizes = range(50, 501, 50)
battery_sizes = range(50, 501, 50)


def simulate_hybrid(sizes):
    solar_mw, wind_mw, battery_mw = sizes
    ic_mw = solar_mw + wind_mw

    technologies = {'pv': {
                        'system_capacity_kw': solar_mw * 1000,
                    },
                    'wind': {
                        'num_turbines': wind_mw * 1000 // 2000,
                        'turbine_rating_kw': 2000
                    },
                    'battery': {
                        'system_capacity_kwh': battery_mw * 4 * 1000,
                        'system_capacity_kw': battery_mw * 1000
                    }}

    hybrid_plant = HybridSimulation(technologies, site, interconnect_kw=ic_mw * 1000)

    hybrid_plant.pv.degradation = (0, )             # year over year degradation
    hybrid_plant.wind.wake_model = 3                # constant wake loss, layout-independent
    hybrid_plant.wind.value("wake_int_loss", 1)     # percent wake loss

    # use single year for now, multiple years with battery not implemented yet
    hybrid_plant.simulate(project_life=20)

    # Save the outputs for JSON
    annual_energies = str(hybrid_plant.annual_energies)
    npvs = str(hybrid_plant.net_present_values)
    revs = str(hybrid_plant.total_revenue)

    res = {
        "sizes": sizes,
        "aeps": json.loads(annual_energies),
        "npvs": json.loads(npvs),
        "revs": json.loads(revs),
    }
    print(res)
    return sizes, annual_energies, npvs, revs


with mp.Pool(mp.cpu_count()) as p:
    results = p.map(simulate_hybrid, product(solar_sizes, wind_sizes, battery_sizes))
    with open(examples_dir / "hybrid_size_grid.json", "w") as f:
        json.dump(results, f)
