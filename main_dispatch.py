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
from hybrid_dispatch import *

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

    # O&M costs per technology
    solar_OM = 13 # $13/kW/year -> https://www.nrel.gov/docs/fy17osti/68023.pdf
    wind_OM = 43 # $43/kW/year -> https://www.nrel.gov/docs/fy18osti/72167.pdf

    # define hybrid system and site
    solar_mw = 1.2  # this is precentage of solar and wind
    wind_mw = 1.0
    interconnect_mw = 50

    ## FLAG: There appears to be 
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

    # Creating battery system
    cell = batteryCell()
    battery = Battery(cell, 200000., 500, 50000)
    bsoc0 = 0.5

    # Initializing dispatch
    HP = dispatch_problem(dispatch_horizon, battery, simplebatt = False)

    ## TODO: update operating costs
    CbP = 0.002
    CbN = 0.002
    Clc = 0.06*HP.battery.nomC
    CdeltaW = 0.0
    Cpv = solar_OM/8760.
    Cwf = wind_OM/8760.
    HP.updateCostParams( CbP, CbN, Clc, CdeltaW, Cpv, Cwf)

    # time series of wind/solar/total in kW
    ts = hybrid_plant.time_series_kW
    ts_wind = ts.Wind
    ts_solar = ts.Solar
    ts_hybrid = ts.Hybrid
    ts_wnet = [interconnect_mw*1000]* dispatch_horizon

    # TODO: update pricing
    P = [.5]*4
    P.extend([0.7]*4)
    P.extend([0.25]*8)
    P.extend([0.9]*4)
    P.extend([0.5]*4)
    P.extend(P)
    P = [x/10. for x in P]  # [$/kWh]

    # initialize dispatch variables - for storage
    dis_wind = copy.deepcopy(ts_wind)
    dis_solar = copy.deepcopy(ts_solar)
    dis_bsoc = [0]*8760
    dis_bcharg = [0]*8760
    dis_bdischarg = [0]*8760
    dis_net = [0]*8760
    dis_P = [0]*8760
    dis_apxblc = [0]*365
    dis_calcblc = [0]*365
    dis_PHblc = [0]*365


    ti = np.arange(0,8760,dispatch_solution) # in hours
    for i,t in enumerate(ti):

        print('Evaluating day ', i, ' out of ', len(ti))

        if t == ti[-1]:
            forecast_wind = ts_wind[t:].tolist()
            forecast_solar = ts_solar[t:].tolist()

            forecast_wind.extend(ts_wind[0:dispatch_horizon - dispatch_solution].tolist())
            forecast_solar.extend(ts_solar[0:dispatch_horizon - dispatch_solution].tolist())
        else:
            forecast_wind = ts_wind[t:t+dispatch_horizon].tolist()
            forecast_solar = ts_solar[t:t+dispatch_horizon].tolist()

        # TODO: dispatch algorithm to go here
        HP.updateSolarWindResGrid(P, ts_wnet, forecast_solar, forecast_wind)
        HP.updateInitialConditions(bsoc0)

        HP.hybrid_optimization_call(printlogs=True)

        # battery lifecycle count
        #dis_blc[i] = 
        if HP.simplebatt:
            # power accounting
            dis_apxblc[i] = (HP.OptModel.Delta()/HP.OptModel.CB())*sum((HP.OptModel.gamma()**t)*HP.OptModel.wdotBC[t]() for t in range(dispatch_solution))
        else:
            # current accounting - McCormick envelope
            dis_apxblc[i] = (HP.OptModel.Delta()/HP.OptModel.CB())*sum(0.8*HP.OptModel.iN[t]() - 0.8*HP.OptModel.zN[t]() for t in range(dispatch_solution))
            # Calculate value base on non-linear relationship
            dis_calcblc[i] = (HP.OptModel.Delta()/HP.OptModel.CB())*sum(HP.OptModel.iN[t]()*(0.8 - 0.8*(bsoc0 if t == 0 else HP.OptModel.bsoc[t-1]())) for t in range(dispatch_solution))
            
            #dis_blc[i] = (HP.OptModel.Delta()/HP.OptModel.CB())*sum((HP.OptModel.gamma()**t)*(HP.OptModel.iP[t]()) for t in range(dispatch_solution))

        dis_PHblc[i] = HP.OptModel.blc()

        # store state-of-charge
        bsoc0 = HP.OptModel.bsoc[dispatch_solution]()
        
        ### Outputs
        # wind and solar plant outputs
        dis_wind[t:t+dispatch_solution] = HP.OptModel.wdotWF[:]()[0:dispatch_solution]
        dis_solar[t:t+dispatch_solution] = HP.OptModel.wdotPV[:]()[0:dispatch_solution]
        dis_net[t:t+dispatch_solution] = HP.OptModel.wdotS[:]()[0:dispatch_solution]
        dis_P[t:t+dispatch_solution] = HP.OptModel.P[:]()[0:dispatch_solution]

        # TODO: keep track of battery charge and discharge from the dispatch algorithm
        ## currently these are power into and out of the battery without losses
        dis_bcharg[t:t+dispatch_solution] = HP.OptModel.wdotBC[:]()[0:dispatch_solution]
        dis_bdischarg[t:t+dispatch_solution] = HP.OptModel.wdotBD[:]()[0:dispatch_solution]
        dis_bsoc[t:t+dispatch_solution] = HP.OptModel.bsoc[:]()[0:dispatch_solution]

        #if i == 5:
        #    break



    # tracking battery lifecycles for the year            
    tot_apxblc = sum(dis_apxblc)
    tot_calcblc = sum(dis_calcblc)

    Error_ratio = tot_calcblc/tot_apxblc

    print("McCormick Battery Lifecycles: " + str(tot_apxblc))
    print("Non-linear Calculation Battery Lifecycles: " + str(tot_calcblc))
    print("Error ratio: " + str(Error_ratio))
    
    # plotting
    tt = np.linspace(0, 8760, 8760) # plotting time array
    StartD = 0
    Np = 4 # number of dispatch time horizone to plot out
    Nf = 10 # fontsize
    power_scale = 1/1000.   # kW to MW

    st = StartD*dispatch_solution
    et = st + Np*dispatch_solution

    plt.figure(figsize=(15,15))
    
    # First sub-plot (resources)
    plt.subplot(3,1,1)
    plt.plot(tt[st:et],[x*power_scale for x in dis_wind][st:et],'b',label='Wind Farm Generation')
    plt.plot(tt[st:et],[x*power_scale for x in ts_wind][st:et],'b--',label='Wind Farm Resource')
    plt.plot(tt[st:et],[x*power_scale for x in dis_solar][st:et],'r',label='PV Generation')
    plt.plot(tt[st:et],[x*power_scale for x in ts_solar][st:et],'r--',label='PV Resource')

    plt.xlim([st,et])
    plt.grid()
    plt.tick_params(which='both', labelsize=Nf)
    plt.ylabel('Power (MW)', fontsize=Nf)
    plt.title('Generation Resources', fontsize=Nf)
    plt.legend(fontsize=Nf,loc='upper left')

    # Battery action
    plt.subplot(3,1,2)
    plt.bar(tt[st:et],[x*power_scale for x in dis_bdischarg][st:et], width = 0.9 ,color = 'blue', edgecolor = 'white',label='Battery Discharge')
    plt.bar(tt[st:et],[-x*power_scale for x in dis_bcharg][st:et], width = 0.9 ,color = 'red', edgecolor = 'white',label='Battery Charge')    
    plt.xlim([st,et])
    plt.grid()
    ax1 = plt.gca()
    ax1.legend(fontsize=Nf,loc='upper left')
    ax1.set_ylabel('Power (MW)', fontsize=Nf)

    ax2 = ax1.twinx()
    ax2.plot(tt[st:et], dis_bsoc[st:et], 'k', label='State-of-Charge')
    ax2.set_ylabel('Stat-of-Charge (-)', fontsize=Nf)
    ax2.legend(fontsize=Nf,loc='upper right')

    plt.tick_params(which='both',labelsize=Nf)
    plt.title('Battery Power Flow', fontsize=Nf)
    
    # Net action
    plt.subplot(3, 1, 3)
    plt.plot(tt[st:et], [x*power_scale for x in ts_solar + ts_wind][st:et], 'k--', label='Original Generation')
    plt.plot(tt[st:et], [x*power_scale for x in dis_net][st:et], 'g', label='Optimized Dispatch')
    plt.xlim([st,et])
    plt.grid()
    ax1 = plt.gca()
    ax1.legend(fontsize=Nf,loc='upper left')
    ax1.set_ylabel('Power (MW)', fontsize=Nf)

    ax2 = ax1.twinx()
    ax2.plot(tt[st:et], dis_P[st:et], 'r', label='Price')
    ax2.set_ylabel('Grid Price ($/kWh)', fontsize=Nf)
    ax2.legend(fontsize=Nf,loc='upper right')

    plt.tick_params(which='both', labelsize=Nf)
    plt.xlabel('Time (hours)', fontsize=Nf)
    plt.title('Net Generation', fontsize=Nf)
    

    plt.show()






