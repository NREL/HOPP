"""
main_dispatch.py

This script calculates the output of a hybrid power plant with wind and solar for a year.
The hybrid plant is 100 MW.

User inputs:
1) allocation of wind and solar (assumed to be 50/50 to start)
2) horizon for dispatch algorithm

TODO:
1) a dispatch algorithm to optimally dispatch storage
2) a battery model
"""
import os

from hybrid.log import *
from defaults.flatirons_site import Site
from hybrid.site_info import SiteInfo
from hybrid.hybrid_system import HybridSystem

import numpy as np
import copy
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    """
    Example script to run the hybrid optimization performance platform (HOPP)
    Custom analysis should be placed a folder within the hybrid_analysis project, which includes HOPP as a submodule
    https://github.com/hopp/hybrid_analysis
    """

    # user inputs:
    dispatch_horizon = 48 # hours
    dispatch_solution = 24 # dispatch solution provided for every 24 hours, simulation advances X hours at a time

    # define hybrid system and site
    solar_mw = 0.5
    wind_mw = 0.5
    interconnect_mw = 100
    # size in mw
    technologies = {'Solar': solar_mw,          # mw system capacity
                    'Wind': wind_mw,            # mw system capacity
                    'Grid': interconnect_mw}    # mw interconnect

    # get resource and create model
    lat = 35.2018863
    lon = -101.945027
    site = SiteInfo(dict({'lat': lat, 'lon': lon}))
    hybrid_plant = HybridSystem(technologies, site, interconnect_kw=interconnect_mw * 1000)

    # prepare results folder
    results_dir = os.path.join('results')

    # size of the hybrid plant
    solar_size_mw = interconnect_mw * solar_mw
    wind_size_mw = interconnect_mw * wind_mw

    hybrid_plant.solar.system_capacity_kw = solar_size_mw * 1000
    hybrid_plant.wind.system_capacity_by_num_turbines(wind_size_mw * 1000)

    actual_solar_pct = hybrid_plant.solar.system_capacity_kw / \
                           (hybrid_plant.solar.system_capacity_kw + hybrid_plant.wind.system_capacity_kw)

    logger.info("Run with solar percent {}".format(actual_solar_pct))

    hybrid_plant.simulate()

    # annual energy production
    annual_energies = hybrid_plant.annual_energies
    hybrid_aep_mw = annual_energies.Hybrid / 1000
    cf_interconnect = 100 * hybrid_aep_mw / (interconnect_mw * 8760)

    # capacity factors
    capacity_factors = hybrid_plant.capacity_factors

    # net present value
    npvs = hybrid_plant.net_present_values # in dollars

    # irradiance
    irrs = hybrid_plant.internal_rate_of_returns

    # time series of wind/solar/total in MW since we are looking at utility-scale
    ts = hybrid_plant.time_series_kW
    ts_wind = ts.Wind/1000
    ts_solar = ts.Solar/1000
    ts_hybrid = ts.Hybrid

    # initialize dispatch variables
    dis_wind = copy.deepcopy(ts_wind)
    dis_solar = copy.deepcopy(ts_solar)
    ti = np.arange(0,8760,dispatch_solution) # in hours
    for i,t in enumerate(ti):

        print('Evaluating day ', i, ' out of ', len(ti))

        forecast_wind = ts_wind[i:i*dispatch_horizon]
        forecast_solar = ts_solar[i:i+dispatch_horizon]

        # TODO: dispatch algorithm to go here
        # example outputs intended to be replaced with actual outputs from dispatch
        curtail_wind = np.random.rand(1) * np.ones(dispatch_solution)
        curtail_solar = np.random.rand(1) * np.ones(dispatch_solution)

        # wind and solar plant outputs
        # print(len(ts_wind[i:i+dispatch_solution]),len(curtail_wind))
        dis_wind[i:i+dispatch_solution] = ts_wind[i:i+dispatch_solution]*curtail_wind
        dis_solar[i:i+dispatch_solution] = ts_solar[i:i+dispatch_solution]*curtail_solar

        # TODO: keep track of battery charge and discharge from the dispatch algorithm

    # plotting
    tt = np.linspace(0, 8760, 8760) # plotting time array
    Np = 4 # number of dispatch time horizone to plot out
    Nf = 10 # fontsize
    plt.figure(figsize=(15,15))
    plt.subplot(3,1,1)
    plt.plot(tt,ts_wind,'k',label='Original')
    plt.plot(tt,dis_wind,'b--',label='Controlled')
    plt.xlim([0,Np*dispatch_horizon])
    plt.grid()
    plt.tick_params(which='both', labelsize=Nf)
    plt.ylabel('Power (MW)', fontsize=Nf)
    plt.title('Wind', fontsize=Nf)
    plt.legend(fontsize=Nf,loc='upper left')

    plt.subplot(3,1,2)
    plt.plot(tt,ts_solar,'k',label='Original')
    plt.plot(tt,dis_solar,'r--',label='Controlled')
    plt.xlim([0,Np*dispatch_horizon])
    plt.grid()
    plt.tick_params(which='both',labelsize=Nf)
    plt.ylabel('Power (MW)', fontsize=Nf)
    plt.title('Solar', fontsize=Nf)
    plt.legend(fontsize=Nf,loc='upper left')

    plt.subplot(3, 1, 3)
    plt.plot(tt, ts_solar + ts_wind, 'k', label='Original')
    plt.plot(tt, dis_solar + dis_wind, 'g--', label='Controlled')
    plt.xlim([0, Np * dispatch_horizon])
    plt.grid()
    plt.tick_params(which='both', labelsize=Nf)
    plt.xlabel('Time (hours)', fontsize=Nf)
    plt.ylabel('Power (MW)', fontsize=Nf)
    plt.title('Total', fontsize=Nf)
    plt.legend(fontsize=Nf,loc='upper left')

    plt.show()






