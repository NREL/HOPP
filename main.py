"""
main.py

The high level wrapper that runs the system optimization
This is a wrapper around the workflow of:
- Step 0: Defining technologies to evaluate and their constraints
- Step 1: Resource allocation: getting the optimal mix and size of technologies
- Step 2: Optimal design: using Python to optimize the hybrid system design, evaluating performance and financial metrics with SAM
- Step 3: Evaluation model: Performing uncertainty quantification
"""
import os

from hybrid.log import *
from defaults.flatirons_site import Site
from hybrid.site_info import SiteInfo
from hybrid.hybrid_system import HybridSystem

if __name__ == '__main__':
    """
    Example script to run the hybrid optimization performance platform (HOPP)
    Custom analysis should be placed a folder within the hybrid_analysis project, which includes HOPP as a submodule
    https://github.com/hopp/hybrid_analysis
    """

    # define hybrid system and site
    solar_mw = 10
    wind_mw = 10
    interconnect_mw = 20
    # size in mw
    technologies = {'Solar': solar_mw,          # mw system capacity
                    'Wind': wind_mw,            # mw system capacity
                    'Grid': interconnect_mw}    # mw interconnect

    # get resource and create model
    lat = 35.2018863
    lon = -101.945027
    site = SiteInfo(dict({'lat': lat, 'lon': lon}))
    hybrid_plant = HybridSystem(technologies, site, interconnect_kw=interconnect_mw * 1000)

    # hybrid_plant.size_from_reopt()

    # prepare results folder
    results_dir = os.path.join('results')

    # Create a range of solar powers to evaluate
    solar = list(range(0, 110, 10))

    # Create a dictionary of outputs to write to file
    save_outputs = dict()
    save_outputs['Solar (%)'] = list()
    save_outputs['Solar (MW)'] = list()
    save_outputs['Wind (MW)'] = list()
    save_outputs['AEP (GWh)'] = list()
    save_outputs['Capacity Factor'] = list()
    save_outputs['Capacity Factor of Interconnect'] = list()
    save_outputs['NPV ($-million)'] = list()
    save_outputs['IRR (%)'] = list()

    # Loop over solar sizes
    for solar_pct in solar:
        optimal = []

        solar_pct *= 0.01
        solar_size_mw = interconnect_mw * solar_pct
        wind_size_mw = interconnect_mw * (1 - solar_pct)

        hybrid_plant.solar.system_capacity_kw = solar_size_mw * 1000
        hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)

        actual_solar_pct = hybrid_plant.solar.system_capacity_kw / \
                           (hybrid_plant.solar.system_capacity_kw + hybrid_plant.wind.system_capacity_kw)

        logger.info("Run with solar percent {}".format(actual_solar_pct))
        print('Solar percent: ' + str(actual_solar_pct))

        hybrid_plant.simulate()

        # save the outputs
        annual_energies = hybrid_plant.annual_energies
        hybrid_aep_mw = annual_energies.Hybrid / 1000
        cf_interconnect = 100 * hybrid_aep_mw / (interconnect_mw * 8760)

        save_outputs['Solar (%)'].append(solar_pct * 100)
        save_outputs['Solar (MW)'].append(solar_size_mw)
        save_outputs['Wind (MW)'].append(wind_size_mw)
        save_outputs['AEP (GWh)'].append(hybrid_aep_mw / 1000)

        capacity_factors = hybrid_plant.capacity_factors
        save_outputs['Capacity Factor'].append(capacity_factors.Hybrid)
        save_outputs['Capacity Factor of Interconnect'].append(capacity_factors.Grid)

        npvs = hybrid_plant.net_present_values
        save_outputs['NPV ($-million)'].append(npvs.Hybrid / 1000000)

        irrs = hybrid_plant.internal_rate_of_returns
        save_outputs['IRR (%)'].append(irrs.Hybrid)

    print(save_outputs)


