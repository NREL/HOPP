"""
main.py

The high level wrapper that runs the system optimization
This is a wrapper around the workflow of:
- Step 0: Defining technologies to evaluate and their constraints
- Step 1: Resource allocation: getting the optimal mix and size of technologies
- Step 2: Optimal design: using Python to optimize the hybrid system design, evaluating performance and financial metrics with SAM
- Step 3: Evaluation model: Performing uncertainty quantification
"""
import copy
import math
import os

from defaults.defaults_data import setup_defaults
from hybrid.hopp import run_hopp


if __name__ == '__main__':
    """
    Example script to run the hybrid optimization performance platform (HOPP)
    Custom analysis should be placed a folder within the hybrid_analysis project, which includes HOPP as a submodule
    https://github.com/hopp/hybrid_analysis
    """

    # prepare results folder
    results_dir = os.path.join('results')

    # Create a range of solar powers to evaluate
    solar = list(range(0, 110, 10))
    nameplate_mw = 20 # MW

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
        print('Solar percent: ' + str(solar_pct))
        technologies_am, defaults_am, Site = setup_defaults()
        technologies_am = ['Wind', 'Solar', 'Generic']
        technologies = copy.deepcopy(technologies_am)
        defaults = copy.deepcopy(defaults_am)

        solar_pct *= 0.01
        solar_size_mw = nameplate_mw * solar_pct
        wind_size_mw = nameplate_mw * (1 - solar_pct)

        # update solar defaults
        defaults['Solar']['Singleowner']['SystemCosts']['total_installed_cost'] = solar_size_mw * 1000000 * 1.11
        defaults['Solar']['Pvsamv1']['SystemDesign']['system_capacity'] = solar_size_mw * 1000
        nstrings = defaults["Solar"]["Pvsamv1"]["SystemDesign"]["subarray1_nstrings"]
        nstrings = math.ceil(nstrings * solar_size_mw / nameplate_mw)
        ninverters = defaults["Solar"]["Pvsamv1"]["SystemDesign"]["inverter_count"]
        ninverters = round(ninverters * solar_size_mw / nameplate_mw, 0)
        defaults["Solar"]["Pvsamv1"]["SystemDesign"]["inverter_count"] = max(ninverters, 1)
        defaults["Solar"]["Pvsamv1"]["SystemDesign"]["subarray1_nstrings"] = max(nstrings, 1)

        # update wind defaults
        defaults['Wind']['Singleowner']['SystemCosts']['total_installed_cost'] = wind_size_mw * 1000 * 1454
        defaults["Wind"]["Windpower"]["Farm"]["system_capacity"] = wind_size_mw * 1000
        turbine_output_kw = max(defaults["Wind"]["Windpower"]["Turbine"]["wind_turbine_powercurve_powerout"])
        n_turbines = math.ceil(wind_size_mw * 1000 / turbine_output_kw)
        xCoords = defaults["Wind"]["Windpower"]["Farm"]['wind_farm_xCoordinates'][0:n_turbines]
        yCoords = defaults["Wind"]["Windpower"]["Farm"]['wind_farm_yCoordinates'][0:n_turbines]
        defaults["Wind"]["Windpower"]["Farm"]['wind_farm_xCoordinates'] = xCoords
        defaults["Wind"]["Windpower"]["Farm"]['wind_farm_yCoordinates'] = yCoords

        if wind_size_mw <= 0:
            technologies.remove('Wind')
        if solar_size_mw <= 0:
            technologies.remove('Solar')

        # run the model
        outputs = run_hopp(technologies=technologies, defaults=defaults, site=Site, run_reopt_optimization=False,
                           run_wind_layout_opt=False, run_system_optimization=False)

        # save the outputs
        cf_interconnect = 100 * (outputs['Generic']['annual_energy']/1000) / (nameplate_mw * 8760)

        save_outputs['Solar (%)'].append(solar_pct * 100)
        save_outputs['Solar (MW)'].append(solar_size_mw)
        save_outputs['Wind (MW)'].append(wind_size_mw)
        save_outputs['AEP (GWh)'].append(outputs['Generic']['annual_energy']/1000000)
        save_outputs['Capacity Factor'].append(outputs['Generic']['capacity_factor'])
        save_outputs['Capacity Factor of Interconnect'].append(cf_interconnect)
        save_outputs['NPV ($-million)'].append(outputs['Generic']['project_return_aftertax_npv']/1000000)
        save_outputs['IRR (%)'].append(outputs['Generic']['analysis_period_irr'])

        #df = pd.DataFrame(save_outputs)
        #df.to_csv(os.path.join(results_dir, 'wind_solar_tradeoff_' + str(Site['lat']) + '_' + str(Site['lon']) + '.csv'))






