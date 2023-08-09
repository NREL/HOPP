"""
multi_location.py

The high level wrapper that runs the system optimization
This is a wrapper around the workflow of:
- Step 0: Defining technologies to evaluate and their constraints
- Step 1: Resource allocation: getting the optimal mix and size of technologies
- Step 2: Optimal design: using Python to optimize the hybrid system design, evaluating performance and financial metrics with SAM
- Step 3: Evaluation model: Performing uncertainty quantification
"""

import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import multiprocessing
import operator
from pathlib import Path
from dotenv import load_dotenv
from itertools import repeat

from hybrid.keys import set_developer_nrel_gov_key
from hybrid.log import analysis_logger as logger
from hybrid.sites import SiteInfo
from hybrid.sites import flatirons_site as sample_site
from hybrid.hybrid_simulation import HybridSimulation
from tools.analysis import create_cost_calculator
from tools.resource import *
from tools.resource.resource_loader import site_details_creator

resource_dir = Path(__file__).parent.parent.parent / "resource_files"

pd.set_option("display.max_rows", None, "display.max_columns", None)

# Set API key
load_dotenv()
NREL_API_KEY = os.getenv("NREL_API_KEY")
set_developer_nrel_gov_key(NREL_API_KEY)  # Set this key manually here if you are not setting it using the .env


def establish_save_output_dict():
    """
    Establishes and returns the base 'save_outputs' dictionary
    for storing all analysis variables.
    """
    # Establishes and returns a 'save_outputs' dict for saving the relevant analysis variables.
    save_outputs = dict()
    save_outputs['Scenario Description'] = list()
    save_outputs['Solar (%)'] = list()
    save_outputs['Solar (MW)'] = list()
    save_outputs['Wind (MW)'] = list()
    save_outputs['AEP (GWh)'] = list()
    save_outputs['Solar AEP (GWh)'] = list()
    save_outputs['Wind AEP (GWh)'] = list()
    save_outputs['Solar Capacity Factor'] = list()
    save_outputs['Capacity Factor'] = list()
    save_outputs['Wind Capacity Factor'] = list()
    save_outputs['Capacity Factor of Interconnect'] = list()
    save_outputs['NPV ($-million)'] = list()
    save_outputs['LCOE - Nominal'] = list()
    save_outputs['LCOE - Real'] = list()
    save_outputs['IRR (%)'] = list()
    save_outputs['PPA Price Used'] = list()
    save_outputs['TOD Profile Used'] = list()
    save_outputs['Revenue (PPA)'] = list()
    save_outputs['Revenue (TOD)'] = list()
    save_outputs['BOS Cost'] = list()
    save_outputs['BOS Cost percent reduction'] = list()
    save_outputs['Cost / MWh Produced'] = list()
    save_outputs['Cost / MWh Produced percent reduction'] = list()
    save_outputs['Percentage Curtailment'] = list()
    save_outputs['Pearson R Wind V Solar'] = list()

    return save_outputs


def establish_save_outputs_resource_loop_dict():
    """
    Establishes and returns a 'save_outputs_resource_loop' dict
    for saving the relevant analysis variables for each site.
    """

    save_outputs_resource_loop = dict()
    save_outputs_resource_loop['Site Lat'] = list()
    save_outputs_resource_loop['Site Lon'] = list()
    save_outputs_resource_loop['PPA Price'] = list()
    save_outputs_resource_loop['Wind Size(MW)'] = list()
    save_outputs_resource_loop['Solar Size(MW)'] = list()
    save_outputs_resource_loop['Hybrid Size(MW)'] = list()
    save_outputs_resource_loop['Wind AEP (GWh)'] = list()
    save_outputs_resource_loop['Solar AEP (GWh)'] = list()
    save_outputs_resource_loop['Hybrid AEP (GWh)'] = list()
    save_outputs_resource_loop['Wind NPV ($-million)'] = list()
    save_outputs_resource_loop['Solar NPV ($-million)'] = list()
    save_outputs_resource_loop['Hybrid NPV ($-million)'] = list()
    save_outputs_resource_loop['Wind LCOE (real)'] = list()
    save_outputs_resource_loop['Solar LCOE (real)'] = list()
    save_outputs_resource_loop['Hybrid LCOE (real)'] = list()
    save_outputs_resource_loop['Wind Cost / MWh Produced'] = list()
    save_outputs_resource_loop['Solar Cost / MWh Produced'] = list()
    save_outputs_resource_loop['Hybrid Cost / MWh Produced'] = list()
    save_outputs_resource_loop['Solar Beats Wind AEP'] = list()
    save_outputs_resource_loop['Solar Beats Wind NPV'] = list()
    save_outputs_resource_loop['Solar Beats Wind Cost/MWh'] = list()
    save_outputs_resource_loop['Percentage Cost Reduction Solar to Match Wind'] = list()
    save_outputs_resource_loop['Max AEP Index'] = list()
    save_outputs_resource_loop['Max AEP Value'] = list()
    save_outputs_resource_loop['Max NPV Index'] = list()
    save_outputs_resource_loop['Max NPV Value'] = list()
    save_outputs_resource_loop['Cost / MWh Produced reduction (%)'] = list()
    save_outputs_resource_loop['LCOE(real) reduction (%)'] = list()
    save_outputs_resource_loop['LCOE(real) reduction (%) vs wind'] = list()
    save_outputs_resource_loop['NPV Benefit ($-million) Hybrid Vs. Wind'] = list()
    save_outputs_resource_loop['Wind BOS Cost'] = list()
    save_outputs_resource_loop['Solar BOS Cost'] = list()
    save_outputs_resource_loop['Hybrid BOS Cost'] = list()
    save_outputs_resource_loop['Wind Capacity Factor'] = list()
    save_outputs_resource_loop['Solar Capacity Factor'] = list()
    save_outputs_resource_loop['Interconnect Capacity Factor (Wind Case)'] = list()
    save_outputs_resource_loop['Interconnect Capacity Factor (Solar Case)'] = list()
    save_outputs_resource_loop['Interconnect Capacity Factor (Hybrid Case)'] = list()
    save_outputs_resource_loop['Percentage Curtailment (Wind)'] = list()
    save_outputs_resource_loop['Percentage Curtailment (Solar)'] = list()
    save_outputs_resource_loop['Percentage Curtailment (Hybrid)'] = list()
    save_outputs_resource_loop['Pearson R Wind V Solar'] = list()
    save_outputs_resource_loop['Solar File Used'] = list()
    save_outputs_resource_loop['Wind File Used'] = list()
    save_outputs_resource_loop['Time Zone (for solar)'] = list()

    return save_outputs_resource_loop


def run_hopp_calc(Site, scenario_description, bos_details, total_hybrid_plant_capacity_mw, solar_size_mw, wind_size_mw,
                    nameplate_mw, interconnection_size_mw, load_resource_from_file,
                    ppa_price, results_dir):
    """ run_hopp_calc Establishes sizing models, creates a wind or solar farm based on the desired sizes,
     and runs SAM model calculations for the specified inputs.
     save_outputs contains a dictionary of all results for the hopp calculation.

    :param scenario_description: Project scenario - 'greenfield' or 'solar addition'.
    :param bos_details: contains bos details including type of analysis to conduct (cost/mw, json lookup, HybridBOSSE).
    :param total_hybrid_plant_capacity_mw: capacity in MW of hybrid plant.
    :param solar_size_mw: capacity in MW of solar component of plant.
    :param wind_size_mw: capacity in MW of wind component of plant.
    :param nameplate_mw: nameplate capacity of total plant.
    :param interconnection_size_mw: interconnection size in MW.
    :param load_resource_from_file: flag determining whether resource is loaded directly from file or through
     interpolation routine.
    :param ppa_price: PPA price in USD($)
    :return: collection of outputs from SAM and hybrid-specific calculations (includes e.g. AEP, IRR, LCOE),
     plus wind and solar filenames used
    (save_outputs)
    """
    # Get resource data
    # sample_site['lat'] = Site['Lat']
    # sample_site['lon'] = Site['Lon']
    # # site = SiteInfo(sample_site, solar_resource_file=Site['resource_filename_solar'],
    # #                 wind_resource_file=Site['resource_filename_wind'])
    if load_resource_from_file:
        pass
    else:
        Site['resource_filename_solar'] = ""  # Unsetting resource filename to force API download of wind resource
        Site['resource_filename_wind'] = ""  # Unsetting resource filename to force API download of solar resource

    site = SiteInfo(Site)

    if 'roll_tz' in Site.keys():
        site.solar_resource.roll_timezone(Site['roll_tz'], Site['roll_tz'])

    # Set up technology and cost model info
    technologies = {'solar': solar_size_mw,          # mw system capacity
                    'wind': wind_size_mw,            # mw system capacity
                    'grid': {'interconnect_kw': interconnection_size_mw * 1000}
                    }

    # Create model
    hybrid_plant = HybridSimulation(technologies, site)

    # hybrid_plant.setup_cost_calculator(create_cost_calculator(bos_cost_source='boslookup', interconnection_mw=interconnection_size_mw))
    hybrid_plant.setup_cost_calculator(create_cost_calculator(bos_cost_source=bos_details['BOSSource'],
                                                                    interconnection_mw=interconnection_size_mw,
                                                              modify_costs=bos_details['Modify Costs'],
                                                              cost_reductions=bos_details,
                                                              wind_installed_cost_mw=1696000,
                                                              solar_installed_cost_mw=1088600,
                                                              storage_installed_cost_mw=0,
                                                              storage_installed_cost_mwh=0,
                                                              ))

    hybrid_plant.ppa_price = ppa_price
    hybrid_plant.discount_rate = 6.4
    hybrid_plant.solar.system_capacity_kw = solar_size_mw * 1000
    hybrid_plant.wind.rotor_diameter = 100
    # Modify Wind Turbine Coordinates
    Nx = 8
    Ny = 5
    spacing = hybrid_plant.wind.row_spacing
    turbine_x = np.linspace(0, (Nx * spacing) - spacing, Nx)
    turbine_y = np.linspace(0, (Ny * spacing) - spacing, Ny)
    turbX = np.zeros(Nx * Ny)
    turbY = np.zeros(Nx * Ny)
    count = 0
    for i in range(Nx):
        for j in range(Ny):
            turbX[count] = turbine_x[i]
            turbY[count] = turbine_y[j]
            count = count + 1
    hybrid_plant.wind.modify_coordinates(list(turbX), list(turbY))
    hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)
    actual_solar_pct = hybrid_plant.solar.system_capacity_kw / \
                       (hybrid_plant.solar.system_capacity_kw + hybrid_plant.wind.system_capacity_kw)

    logger.info("Run with solar percent {}".format(actual_solar_pct))
    hybrid_plant.simulate()
    outputs = hybrid_plant.hybrid_outputs()
    for k, v in outputs.items():
        outputs[k] = [v]

    return outputs, site.wind_resource.filename, site.solar_resource.filename


def run_hybrid_calc(year, site_num, scenario_descriptions, results_dir, load_resource_from_file,
                    resource_filename_wind, resource_filename_solar, site_lat, site_lon,
                    wind_size, solar_size, hybrid_size, interconnection_size,
                    bos_details, ppa_price, solar_tracking_mode, hub_height, correct_wind_speed_for_height):
    """
    run_hybrid_calc loads the specified resource for each site, and runs wind, solar, hybrid and solar addition
    scenarios by calling run_hopp_calc for each scenario. Returns a DataFrame of all results for the supplied site
    :param year: year for which analysis is conducted
    :param site_num: number representing the site studied. Generally 1 to num sites to be studied.
    :param scenario_descriptions: description of scenario type, e.g. wind only, solar only, hybrid.
    :param results_dir: path to results directory.
    :param load_resource_from_file: flag determining whether resource is loaded from file directly or other routine.
    :param resource_filename_wind: filename of wind resource file.
    :param resource_filename_solar: filename of solar resource file.
    :param site_lat: site latitude (degrees).
    :param site_lon: site longitude (degrees).
    :param wind_size: capacity in MW of wind component of plant.
    :param solar_size: capacity in MW of solar component of plant.
    :param hybrid_size: capacity in MW of hybrid plant.
    :param interconnection_size: capacity in MW of interconnection.
    :param bos_details: contains bos details including type of analysis to conduct (cost/mw,
     json lookup, HybridBOSSE).
    :param ppa_price: PPA price in USD($).
    :param solar_tracking_mode: solar tracking mode (e.g. fixed, single-axis, two-axis).
    :param hub_height: hub height in meters.
    :param correct_wind_speed_for_height: (boolean) flag determining whether wind speed is extrapolated
     to hub height.
    :return: save_outputs_resource_loop_dataframe <pandas dataframe> dataframe of all site outputs from hopp runs
    """
    # Set up hopp_outputs dictionary
    hopp_outputs = dict()

    # Site details
    Site = sample_site  # sample_site has been loaded from flatirons_site to provide sample site boundary information
    Site['Lat'] = site_lat
    Site['Lon'] = site_lon
    Site['lat'] = site_lat
    Site['lon'] = site_lon
    Site['site_num'] = site_num
    Site['resource_filename_solar'] = resource_filename_solar
    Site['resource_filename_wind'] = resource_filename_wind
    Site['year'] = int(year)
    print('Performing Analysis for Site Num: {} Lat: {} Lon: {}'.format(Site['site_num'], Site['Lat'], Site['Lon']))
    # Get the Timezone offset value based on the lat/lon of the site

    each_site_filename = 'Site_Result_{}_Lat_{}_Lon_WindSize_{}_MW_SolarSize_{}_MW_ppa_price_$' \
                       '{}_{}m_hub_height.csv' \
        .format(site_lat, site_lon, wind_size, solar_size, ppa_price,
                hub_height)

    if os.path.exists(os.path.join('results/', each_site_filename)):
        hopp_outputs_all_dataframe = pd.read_csv(os.path.join('results/',each_site_filename))
        print("Analysis already run for {}. Loading from file.",format(each_site_filename))
    else:


        try:
            location = {'lat': Site['Lat'], 'long': Site['Lon']}
            tz_val = get_offset(**location)
            Site['tz'] = (tz_val - 1)
        except:
            print('Timezone lookup failed for {}'.format(location))

        scenario_description = 'greenfield'

        # Case 1 - Wind
        solar_size_mw = 0
        wind_size_mw = wind_size
        total_hybrid_plant_capacity_mw = solar_size + wind_size
        nameplate_mw = total_hybrid_plant_capacity_mw
        interconnection_size_mw = interconnection_size
        hopp_outputs['Wind'], resource_filename_wind, resource_filename_solar = \
            run_hopp_calc(Site, scenario_description, bos_details, total_hybrid_plant_capacity_mw, solar_size_mw, wind_size_mw,
                        nameplate_mw, interconnection_size_mw, load_resource_from_file,
                        ppa_price, results_dir)

        # Case 2 - Solar
        solar_size_mw = solar_size
        wind_size_mw = 0
        total_hybrid_plant_capacity_mw = solar_size + wind_size
        nameplate_mw = total_hybrid_plant_capacity_mw
        interconnection_size_mw = interconnection_size
        hopp_outputs['Solar'], resource_filename_wind, resource_filename_solar\
            = run_hopp_calc(Site, scenario_description, bos_details, total_hybrid_plant_capacity_mw, solar_size_mw, wind_size_mw,
                        nameplate_mw, interconnection_size_mw, load_resource_from_file,
                        ppa_price, results_dir)

        # Case 3 - Hybrid Wind + Solar
        solar_size_mw = solar_size
        wind_size_mw = wind_size
        total_hybrid_plant_capacity_mw = solar_size + wind_size
        nameplate_mw = total_hybrid_plant_capacity_mw
        interconnection_size_mw = interconnection_size
        hopp_outputs['Hybrid'], resource_filename_wind, resource_filename_solar \
            = run_hopp_calc(Site, scenario_description, bos_details, total_hybrid_plant_capacity_mw, solar_size_mw, wind_size_mw,
                        nameplate_mw, interconnection_size_mw, load_resource_from_file,
                        ppa_price, results_dir)

        max_aep_index, max_aep_value = max(enumerate([hopp_outputs['Wind']['AEP (GWh)'][0],
                                                      hopp_outputs['Solar']['AEP (GWh)'][0],
                                                      hopp_outputs['Hybrid']['AEP (GWh)'][0]]),
                                           key=operator.itemgetter(1))
        max_npv_index, max_npv_value = max(enumerate([hopp_outputs['Wind']['NPV ($-million)'][0],
                                                      hopp_outputs['Solar']['NPV ($-million)'][0],
                                                      hopp_outputs['Hybrid']['NPV ($-million)'][0]]),
                                           key=operator.itemgetter(1))

        # Determine the differential between standalone wind + standalone solar vs. wind + adding solar
        hopp_outputs['Hybrid vs. Separate'] = establish_save_output_dict()
        hopp_outputs['Hybrid vs. Separate']['Scenario Description'].append(
            '(Combined Wind & Solar) - (Standalone Wind + Standalone Solar)')
        hopp_outputs['Hybrid vs. Separate']['Solar (%)'].append(float('nan'))
        hopp_outputs['Hybrid vs. Separate']['Solar (MW)'].append(float('nan'))
        hopp_outputs['Hybrid vs. Separate']['Wind (MW)'].append(float('nan'))
        hopp_outputs['Hybrid vs. Separate']['AEP (GWh)'].append((hopp_outputs['Hybrid']['AEP (GWh)'][0])
                                                               - (hopp_outputs['Wind']['AEP (GWh)'][0]
                                                                  + hopp_outputs['Solar']['AEP (GWh)'][0]))
        hopp_outputs['Hybrid vs. Separate']['Solar AEP (GWh)'].append(float('nan'))
        hopp_outputs['Hybrid vs. Separate']['Wind AEP (GWh)'].append(float('nan'))
        hopp_outputs['Hybrid vs. Separate']['Solar Capacity Factor'].append(float('nan'))
        hopp_outputs['Hybrid vs. Separate']['Capacity Factor'].append(float('nan'))
        hopp_outputs['Hybrid vs. Separate']['Wind Capacity Factor'].append(float('nan'))
        hopp_outputs['Hybrid vs. Separate']['Capacity Factor of Interconnect'].append(float('nan'))
        hopp_outputs['Hybrid vs. Separate']['Percentage Curtailment'].append(float('nan'))
        hopp_outputs['Hybrid vs. Separate']['NPV ($-million)'].append(float('nan'))
        hopp_outputs['Hybrid vs. Separate']['LCOE - Nominal'].append(float('nan'))
        hopp_outputs['Hybrid vs. Separate']['LCOE - Real'].append(float('nan'))
        hopp_outputs['Hybrid vs. Separate']['IRR (%)'].append(float('nan'))
        hopp_outputs['Hybrid vs. Separate']['PPA Price Used'].append(float('nan'))
        hopp_outputs['Hybrid vs. Separate']['TOD Profile Used'].append(float('nan'))
        hopp_outputs['Hybrid vs. Separate']['Revenue (PPA)'].append(float('nan'))
        hopp_outputs['Hybrid vs. Separate']['Revenue (TOD)'].append(float('nan'))
        hopp_outputs['Hybrid vs. Separate']['Pearson R Wind V Solar'].append(float('nan'))
        hopp_outputs['Hybrid vs. Separate']['BOS Cost'].append((hopp_outputs['Hybrid']['BOS Cost'][0])
                                                              - (hopp_outputs['Wind']['BOS Cost'][0]
                                                                 + hopp_outputs['Solar']['BOS Cost'][0]))
        hopp_outputs['Hybrid vs. Separate']['BOS Cost percent reduction'].append(100 * ((
                                                                                           hopp_outputs[
                                                                                               'Hybrid vs. Separate'][
                                                                                               'BOS Cost'][0]) /
                                                                                       ((hopp_outputs['Wind']['BOS Cost'][0]
                                                                                         + hopp_outputs['Solar']['BOS Cost'][
                                                                                             0]))))
        hopp_outputs['Hybrid vs. Separate']['Cost / MWh Produced'].append((hopp_outputs['Hybrid']
        ['Cost / MWh Produced'][0])
                                                                         - (hopp_outputs['Wind']['Cost / MWh Produced'][0]
                                                                            + hopp_outputs['Solar']['Cost / MWh Produced'][
                                                                                0]) / 2)
        hopp_outputs['Hybrid vs. Separate']['Cost / MWh Produced percent reduction']. \
            append(100 * ((hopp_outputs['Hybrid vs. Separate']
        ['Cost / MWh Produced'][0])
                          / ((hopp_outputs['Wind']['Cost / MWh Produced'][0]
                              + hopp_outputs['Solar']
                              ['Cost / MWh Produced'][0])/2)))
        cost_per_mw_reduction_hybrid_vs_standalone = ((hopp_outputs['Hybrid']['Cost / MWh Produced'][0])
                                                      - (hopp_outputs['Wind']['Cost / MWh Produced'][0]
                                                         + hopp_outputs['Solar']['Cost / MWh Produced'][0]) / 2)
        cost_per_mw_reduction_hybrid_vs_standalone_percent = (100 * ((hopp_outputs['Hybrid vs. Separate']
        ['Cost / MWh Produced'][0])
                                                                     / ((hopp_outputs['Wind']
                                                                         ['Cost / MWh Produced'][0]
                                                                         + hopp_outputs['Solar']
                                                                         ['Cost / MWh Produced'][0])/2)))
        lcoe_real_reduction_hybrid_percentage_vs_wind = 100 * abs((hopp_outputs['Hybrid']['LCOE - Real'][0]
                                                                   - hopp_outputs['Wind']['LCOE - Real'][0])
                                                                  / hopp_outputs['Wind']['LCOE - Real'][0])
        lcoe_real_reduction_hybrid_percentage = 100 * abs((hopp_outputs['Hybrid']['LCOE - Real'][0]
                                                           - min(hopp_outputs['Wind']['LCOE - Real'][0]
                                                                 , hopp_outputs['Solar']['LCOE - Real'][0]))
                                                          / min(hopp_outputs['Wind']['LCOE - Real'][0]
                                                                , hopp_outputs['Solar']['LCOE - Real'][0]))
        # Determine Percentage Change in COE/reduction in bos for solar to compete with wind
        percent_change_solar_coe_to_compete_with_wind = max(0, 100 * ((hopp_outputs['Solar']['Cost / MWh Produced'][0]
                                                                       - hopp_outputs['Wind']['Cost / MWh Produced'][0])
                                                                      / hopp_outputs['Solar']['Cost / MWh Produced'][0]))
        change_in_npv_hybrid_vs_wind = hopp_outputs['Hybrid']['NPV ($-million)'][0] - \
                                       hopp_outputs['Wind']['NPV ($-million)'][0]

        # Save all relevant outputs for each site
        hopp_outputs_all = establish_save_outputs_resource_loop_dict()
        hopp_outputs_all['Site Lat'].append(Site['Lat'])
        hopp_outputs_all['Site Lon'].append(Site['Lon'])
        hopp_outputs_all['PPA Price'].append(ppa_price)
        hopp_outputs_all['Wind Size(MW)'].append(wind_size)
        hopp_outputs_all['Solar Size(MW)'].append(solar_size)
        hopp_outputs_all['Hybrid Size(MW)'].append(hybrid_size)
        hopp_outputs_all['Wind AEP (GWh)'].append(hopp_outputs['Wind']['AEP (GWh)'][0])
        hopp_outputs_all['Solar AEP (GWh)'].append(hopp_outputs['Solar']['AEP (GWh)'][0])
        hopp_outputs_all['Hybrid AEP (GWh)'].append(hopp_outputs['Hybrid']['AEP (GWh)'][0])
        hopp_outputs_all['Wind NPV ($-million)'].append(hopp_outputs['Wind']['NPV ($-million)'][0])
        hopp_outputs_all['Solar NPV ($-million)'].append(hopp_outputs['Solar']['NPV ($-million)'][0])
        hopp_outputs_all['Hybrid NPV ($-million)'].append(hopp_outputs['Hybrid']['NPV ($-million)'][0])
        hopp_outputs_all['Wind LCOE (real)'].append(hopp_outputs['Wind']['LCOE - Real'][0])
        hopp_outputs_all['Solar LCOE (real)'].append(hopp_outputs['Solar']['LCOE - Real'][0])
        hopp_outputs_all['Hybrid LCOE (real)'].append(hopp_outputs['Hybrid']['LCOE - Real'][0])
        hopp_outputs_all['LCOE(real) reduction (%)'].append(lcoe_real_reduction_hybrid_percentage)
        hopp_outputs_all['LCOE(real) reduction (%) vs wind'].append(lcoe_real_reduction_hybrid_percentage_vs_wind)
        hopp_outputs_all['Wind Cost / MWh Produced'].append(hopp_outputs['Wind']['Cost / MWh Produced'][0])
        hopp_outputs_all['Solar Cost / MWh Produced'].append(hopp_outputs['Solar']['Cost / MWh Produced'][0])
        hopp_outputs_all['Hybrid Cost / MWh Produced'].append(hopp_outputs['Hybrid']['Cost / MWh Produced'][0])
        hopp_outputs_all['Wind BOS Cost'].append(hopp_outputs['Wind']['BOS Cost'][0])
        hopp_outputs_all['Solar BOS Cost'].append(hopp_outputs['Solar']['BOS Cost'][0])
        hopp_outputs_all['Hybrid BOS Cost'].append(hopp_outputs['Hybrid']['BOS Cost'][0])
        hopp_outputs_all['Cost / MWh Produced reduction (%)'].append(
            cost_per_mw_reduction_hybrid_vs_standalone_percent)
        hopp_outputs_all['NPV Benefit ($-million) Hybrid Vs. Wind'].append(change_in_npv_hybrid_vs_wind)
        hopp_outputs_all['Solar Beats Wind AEP'].append(
            1 * (hopp_outputs['Solar']['AEP (GWh)'][0] > hopp_outputs['Wind']['AEP (GWh)'][0]))
        hopp_outputs_all['Solar Beats Wind NPV'].append(
            1 * (hopp_outputs['Solar']['NPV ($-million)'][0] > hopp_outputs['Wind']['NPV ($-million)'][0]))
        hopp_outputs_all['Solar Beats Wind Cost/MWh'].append(1 * (hopp_outputs['Solar']['Cost / MWh Produced'][0]
                                                                            < hopp_outputs['Wind']['Cost / MWh Produced'][
                                                                                0]))
        hopp_outputs_all['Percentage Cost Reduction Solar to Match Wind'].append(percent_change_solar_coe_to_compete_with_wind)
        hopp_outputs_all['Max AEP Index'].append(max_aep_index)
        hopp_outputs_all['Max AEP Value'].append(max_aep_value)
        hopp_outputs_all['Max NPV Index'].append(max_npv_index)
        hopp_outputs_all['Max NPV Value'].append(max_npv_value)
        hopp_outputs_all['Wind Capacity Factor'].append(hopp_outputs['Wind']['Wind Capacity Factor'][0])
        hopp_outputs_all['Solar Capacity Factor'].append(hopp_outputs['Solar']['Solar Capacity Factor'][
                                                                       0])  # save_outputs_resource_loop['NPV @ 100% Solar'].append(npv_at_100_solar)
        hopp_outputs_all['Interconnect Capacity Factor (Wind Case)'].append(
            hopp_outputs['Wind']['Capacity Factor of Interconnect'][0])
        hopp_outputs_all['Interconnect Capacity Factor (Solar Case)'].append(
            hopp_outputs['Solar']['Capacity Factor of Interconnect'][0])
        hopp_outputs_all['Interconnect Capacity Factor (Hybrid Case)'].append(
            hopp_outputs['Hybrid']['Capacity Factor of Interconnect'][0])
        hopp_outputs_all['Percentage Curtailment (Wind)'].append(hopp_outputs['Wind']['Percentage Curtailment'][0])
        hopp_outputs_all['Percentage Curtailment (Solar)'].append(
            hopp_outputs['Solar']['Percentage Curtailment'][0])
        hopp_outputs_all['Percentage Curtailment (Hybrid)'].append(
            hopp_outputs['Hybrid']['Percentage Curtailment'][0])
        hopp_outputs_all['Pearson R Wind V Solar'].append(hopp_outputs['Hybrid']['Pearson R Wind V Solar'][0])
        hopp_outputs_all['Solar File Used'].append(resource_filename_solar)
        hopp_outputs_all['Wind File Used'].append(resource_filename_wind)
        hopp_outputs_all['Time Zone (for solar)'].append(Site['tz'])
        hopp_outputs_all_dataframe = pd.DataFrame(hopp_outputs_all)

        hopp_outputs_all_dataframe.to_csv(os.path.join(results_dir,
                                      each_site_filename))

    return hopp_outputs_all_dataframe


def run_all_hybrid_calcs(site_details, scenario_descriptions, results_dir, load_resource_from_file, wind_size,
                         solar_size, hybrid_size, interconnection_size, bos_details, ppa_price, solar_tracking_mode, hub_height,
                         correct_wind_speed_for_height):
    """
    Performs a multi-threaded run of run_hybrid_calc for the given input parameters.
    Returns a dataframe result for all sites
    :param site_details: DataFrame containing site details for all sites to be analyzed,
    including site_nums, lat, long, wind resource filename and solar resource filename.
    :param scenario_description: Project scenario - e.g. 'Wind Only', 'Solar Only', 'Hybrid - Wind & Solar'.
    :param results_dir: path to results directory
    :param load_resource_from_file: (boolean) flag which determines whether
    :param wind_size: capacity in MW of wind plant.
    :param solar_size: capacity in MW of solar plant.
    :param hybrid_size: capacity in MW of hybrid plant.
    :param bos_details: contains bos details including type of analysis to conduct (cost/mw, json lookup, HybridBOSSE).
    :param ppa_price: ppa price in $(USD)
    :param solar_tracking_mode: solar tracking mode
    :param hub_height: hub height in meters.
    :param correct_wind_speed_for_height: (boolean) flag determining whether wind speed is extrapolated to hub height.
    :return: DataFrame of results for run_hybrid_calc at all sites (save_all_runs)
    """
    # Establish output DataFrame
    save_all_runs = pd.DataFrame()

    # Combine all arguments to pass to run_hybrid_calc
    all_args = zip(site_details['year'], site_details['site_nums'], repeat(scenario_descriptions), repeat(results_dir),
                   repeat(load_resource_from_file),
                   site_details['wind_filenames'], site_details['solar_filenames'],
                   site_details['Lat'], site_details['Lon'],
                   repeat(wind_size), repeat(solar_size), repeat(hybrid_size), repeat(interconnection_size),
                   repeat(bos_details), repeat(ppa_price),
                   repeat(solar_tracking_mode), repeat(hub_height),
                   repeat(correct_wind_speed_for_height))

    # Run a multi-threaded analysis
    with multiprocessing.Pool(16) as p:
        try:
            dataframe_result = p.starmap(run_hybrid_calc, all_args)
            save_all_runs = save_all_runs.append(dataframe_result, sort=False)
        except:
            exception = sys.exc_info()

            def eprint(*args):
                print(*args, file=sys.stderr)

            error = exception[1]
            eprint("Error in run_hopp execution:", error.args[0])
            eprint("Hub Height: ", hub_height)
            raise RuntimeError(error.args[0])

    return save_all_runs


if __name__ == '__main__':
    """
    Full HOPP analysis of the USA using HybridSystem approach
    -Establishes Scenarios to analyze
    -Sets BOS parameters
    -Sets analysis option flags (e.g. load_resource_from_file)
    -Sets range of solar, wind, and hybrid sizes to be analyzed
    -Analyses all sites defined
    -Saves results to file
    """
    # Set paths
    parent_path = os.path.abspath(os.path.dirname(__file__))
    main_path = os.path.abspath(os.path.join(parent_path, 'analysis'))
    print("Parent path: ", parent_path)
    print("Main path", main_path)
    results_dir = os.path.join(parent_path, 'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # Establish Project Scenarios and Parameter Ranges:
    bos_details = dict()
    bos_details['BOSSource'] = 'costpermw'  # costpermw, BOSLookup, HybridBOSSE, HybridBOSSE_manual
    bos_details['BOSFile'] = 'UPDATED_BOS_Summary_Results.json'
    bos_details['BOSScenario'] = 'TBD in analysis'  # Will be set to Wind Only, Solar Only,
    # Variable Ratio Wind and Solar Greenfield, or Solar Addition
    bos_details['BOSScenarioDescription'] = ''  # Blank or 'Overbuild'
    bos_details['Modify Costs'] = True
    bos_details['wind_capex_reduction'] = 0
    bos_details['solar_capex_reduction'] = 0
    bos_details['wind_bos_reduction'] = 0
    bos_details['solar_bos_reduction'] = 0
    bos_details['wind_capex_reduction_hybrid'] = 0
    bos_details['solar_capex_reduction_hybrid'] = 0.1176
    bos_details['wind_bos_reduction_hybrid'] = 0
    bos_details['solar_bos_reduction_hybrid'] = 0
    # global key_counter
    key_counter = 1
    load_resource_from_file = False
    solar_from_file = True
    wind_from_file = True
    on_land_only = False
    in_usa_only = True  # Only use one of (in_usa / on_land) flags
    # Set Analysis Location and Details
    year = 2013
    N_lat = 50 #50  # number of data points
    N_lon = 95 #95
    import numpy as np
    desired_lats = np.linspace(23.833504, 49.3556, N_lat)
    desired_lons = np.linspace(-129.22923, -65.7146, N_lon)

    # Load wind and solar resource files for location nearest desired lats and lons
    # NB this resource information will be overriden by API retrieved data if load_resource_from_file is set to False
    sitelist_name = 'filtered_site_details_{}_lats_{}_lons_{}_year'.format(N_lat, N_lon, year)
    if load_resource_from_file:
        # Loads resource files in 'resource_files', finds nearest files to 'desired_lats' and 'desired_lons'
        site_details = resource_loader_file(resource_dir, desired_lats, desired_lons, year)  # Return contains
        site_details.to_csv(os.path.join(resource_dir, 'site_details.csv'))
        site_details = filter_sites(site_details, location='usa only')
    else:
        # Creates the site_details file containing grid of lats, lons, years, and wind and solar filenames (blank
        # - to force API resource download)
        if os.path.exists(sitelist_name):
            site_details = pd.read_csv(sitelist_name)
        else:
            site_details = site_details_creator.site_details_creator(desired_lats, desired_lons, year)
            # Filter to locations in USA
            site_details = filter_sites(site_details, location='usa only')
            site_details.to_csv(sitelist_name)


    solar_tracking_mode = 'Fixed'  # Currently not making a difference
    ppa_prices = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    solar_bos_reduction_options = [0]  # [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    hub_height_options = [80]  # [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    correct_wind_speed_for_height = True
    interconnection_sizes = [100]
    # wind_sizes = [100, 200]
    # solar_sizes = [100, 200]
    wind_sizes = [100]
    solar_sizes = [100]
    hybrid_sizes = [wind_sizes[i] + solar_sizes[i] for i in range(len(wind_sizes))]
    start_index = 1
    end_index = 2384
    for ppa_price in ppa_prices:
        for solar_bos_reduction in solar_bos_reduction_options:
            for hub_height in hub_height_options:
                for interconnection_size in interconnection_sizes:
                    for i, wind_size in enumerate(wind_sizes):
                        solar_size = solar_sizes[i]
                        hybrid_size = hybrid_sizes[i]
                        if hybrid_size == 400:
                            #TODO: Clean this up after paper analysis
                            interconnection_size = 100
                            bos_details['BOSScenarioDescription'] = "Dont use cost info"
                        else:
                            bos_details['BOSScenarioDescription'] = ""

                        # Establish any additional arguments for analysis
                        bos_details['solar_bos_reduction_hybrid'] = solar_bos_reduction

                        # Run hybrid calculation for all sites
                        save_all_runs = run_all_hybrid_calcs(site_details[start_index:end_index], "greenfield", results_dir,
                                                             load_resource_from_file, wind_size,
                                                             solar_size, hybrid_size, interconnection_size, bos_details,
                                                             ppa_price, solar_tracking_mode, hub_height,
                                                             correct_wind_speed_for_height)

                        # Save master dataframe containing all locations to .csv
                        all_run_filename = 'All_Runs_{}_WindSize_{}_MW_SolarSize_{}_MW_ppa_price_$' \
                                           '{}_solar_bos_reduction_fraction_{}_{}m_hub_height.csv'\
                            .format(bos_details['BOSScenarioDescription'], wind_size, solar_size, ppa_price,
                                    solar_bos_reduction, hub_height)

                        save_all_runs.to_csv(os.path.join(results_dir,
                                             all_run_filename))
                        print(save_all_runs)
