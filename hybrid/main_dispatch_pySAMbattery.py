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
from hybrid.log import *
from defaults.flatirons_site import Site
from hybrid.site_info import SiteInfo
from hybrid.hybrid_system import HybridSystem
from hybrid_dispatch import *

import clustering
from clustering import cluster

import numpy as np
import copy, time, csv
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import pandas as pd

import PySAM.StandAloneBattery as battery_model
from PySAM.BatteryTools import *
import PySAM.BatteryStateful as bt

import PySAM.Singleowner

class sim_with_dispatch:
    def __init__(self, horizon, sol_up, solar, wind, price, netlim, istest = False):
        self.dispatch_horizon = horizon
        self.dispatch_solution = sol_up
        self.dis_solar = solar
        self.dis_wind = wind
        self.P = price
        self.ts_wnet = netlim
        if len(self.dis_solar) == len(self.dis_wind) and len(self.dis_wind) == len(self.P) and len(self.P) == len(self.ts_wnet):
            self.Nsteps = len(self.P)
        else:
            print("Solar, wind, and price data must be equal in length")
            return
        self.bsoc0 = 0.0

        # battery dispatch variables dictionary
        self.bat_dispatch = {}
        self.bat_dispatch['SOC'] = [0.]*self.Nsteps 
        self.bat_dispatch['P_charge'] = [0.]*self.Nsteps
        self.bat_dispatch['P_discharge'] = [0.]*self.Nsteps
        self.bat_dispatch['I_charge'] = [0.]*self.Nsteps
        self.bat_dispatch['I_discharge'] = [0.]*self.Nsteps
        self.bat_dispatch['wind'] = [0.]*self.Nsteps
        self.bat_dispatch['solar'] = [0.]*self.Nsteps
        self.bat_dispatch['net'] = [0.]*self.Nsteps
        self.bat_dispatch['Price'] = [0.]*self.Nsteps
        self.bat_dispatch['apxblc'] = [0.]*self.Nsteps
        self.bat_dispatch['clacblc'] = [0.]*self.Nsteps
        self.bat_dispatch['PHblc'] = [0.]*self.Nsteps

        # comparison between dispatch (battery) and hybrid wo battery dictionary
        self.compare = {}
        self.compare['generation_wo_battery'] = [0.]*self.Nsteps
        self.compare['generation_dispatch'] = [0.]*self.Nsteps
        self.compare['OFV_wo_battery'] = [0.]*self.Nsteps
        self.compare['OFV_dispatch'] = [0.]*self.Nsteps
        self.compare['OFV_diff'] = [0.]*self.Nsteps
        self.compare['revenue_wo_battery'] = [0.]*self.Nsteps
        self.compare['revenue_dispatch'] = [0.]*self.Nsteps
        self.compare['curtailment_wo_battery'] = [0.]*self.Nsteps
        self.compare['curtailment_dispatch'] = [0.]*self.Nsteps
        self.compare['capfactor_wo_battery'] = [0.]*self.Nsteps
        self.compare['capfactor_dispatch'] = [0.]*self.Nsteps

        # Stateful battery model variables dictionary
        self.bat_state = {}
        self.bat_state['control'] = [0.]*self.Nsteps
        self.bat_state['response'] = [0.]*self.Nsteps
        self.bat_state['SOC'] = [0.]*self.Nsteps
        self.bat_state['P_charge'] = [0.]*self.Nsteps
        self.bat_state['P_discharge'] = [0.]*self.Nsteps
        self.bat_state['I_charge'] = [0.]*self.Nsteps
        self.bat_state['I_discharge'] = [0.]*self.Nsteps
        self.bat_state['P'] = [0.]*self.Nsteps
        self.bat_state['I'] = [0.]*self.Nsteps
        self.bat_state['max_temp_persolution'] = [0.]*int(self.Nsteps/self.dispatch_solution)
        self.bat_state['temperature'] = [0.]*self.Nsteps

    def setupBatteryandDispatchMods(self, Hybrid_Dispatch, StateFulBatt):
        self.HP = Hybrid_Dispatch
        self.StateBatt = StateFulBatt

    def simulate(self, start_time, Ndays, bsoc0, Nprev_sim=0,printlogs=True):
        '''
        INPUTS:
            start_time
            Ndays
            bsoc0

        OBJECTS:
            HP
            StateBatt

        Updates:
            bat_state
            compare
            bat_dispatch
        '''
        if not hasattr(self, 'HP'):
            print("Please setup battery and hybrid dispatch before running simulation.")
            return
        else:        
            HP = self.HP
            StateBatt = self.StateBatt
        
        ti = np.arange(start_time, start_time + Ndays*self.dispatch_solution, self.dispatch_solution) # in hours
        for i,t in enumerate(ti):        
            #### Update Solar, Wind, and price forecasts
            # Handling end of year analysis window - Assumes the same 
            if self.Nsteps - t < self.dispatch_horizon:
                forecast_wind = self.dis_wind[t:].tolist()
                forecast_solar = self.dis_solar[t:].tolist()
                forecast_prices = self.P[t:]
                forecast_wnet = self.ts_wnet[t:]

                # Extends the last day of the year problem with data from the first day
                forecast_wind.extend(self.dis_wind[0:self.dispatch_horizon - len(forecast_wind)].tolist())
                forecast_solar.extend(self.dis_solar[0:self.dispatch_horizon - len(forecast_solar)].tolist())
                forecast_prices.extend(self.P[0:self.dispatch_horizon - len(forecast_prices)])
                forecast_wnet.extend(self.ts_wnet[0:self.dispatch_horizon - len(forecast_wnet)])
            else: 
                forecast_wind = self.dis_wind[t:t+self.dispatch_horizon].tolist()
                forecast_solar = self.dis_solar[t:t+self.dispatch_horizon].tolist()
                forecast_prices = self.P[t:t+self.dispatch_horizon]
                forecast_wnet = self.ts_wnet[t:t+self.dispatch_horizon]

            HP.updateSolarWindResGrid(forecast_prices, forecast_wnet, forecast_solar, forecast_wind)
            HP.updateInitialConditions(bsoc0)

            ## TODO: Make call more robust (no solution)
            HP.hybrid_optimization_call(printlogs=printlogs)
            ## Simple battery model scales well - could automatically toggle simple battery on if detail battery reaches solve limits

            # Running stateful battery model to step through the solution
            batt_max_temp = 0.0
            for x in range(self.dispatch_solution):
                if HP.simplebatt:
                    if HP.OptModel.wdotBC[x]() > HP.OptModel.wdotBD[x]():     # Charging
                        control_value =  - HP.OptModel.wdotBC[x]()
                    else:   # Discharging
                        control_value =  HP.OptModel.wdotBD[x]()
                else:
                    if HP.OptModel.iP[x]() > HP.OptModel.iN[x]():         # Charging
                        control_value = - HP.OptModel.iP[x]()*1000. # [kA] -> [A]
                    else:   # Discharging
                        control_value = HP.OptModel.iN[x]()*1000.   # [kA] -> [A]
                
                StateBatt.value(control_var, control_value)
                StateBatt.execute()

                # Only store infromation if passe the previous day simulation (used in clustering)
                if i >= Nprev_sim:
                    # Storing State battery information
                    self.bat_state['control'][t+x] = control_value
                    if HP.simplebatt:
                        self.bat_state['response'][t+x] = StateBatt.StatePack.P
                    else:
                        self.bat_state['response'][t+x] = StateBatt.StatePack.I
                    self.bat_state['SOC'][t+x] = StateBatt.StatePack.SOC/100.
                    self.bat_state['I'][t+x] = StateBatt.StatePack.I/1000.
                    self.bat_state['P'][t+x] = StateBatt.StatePack.P/1000.
                    if StateBatt.StatePack.P > 0.0:
                        self.bat_state['P_discharge'][t+x] = StateBatt.StatePack.P/1000.
                        self.bat_state['I_discharge'][t+x] = StateBatt.StatePack.I/1000.
                        self.bat_state['P_charge'][t+x] = 0.0
                        self.bat_state['I_charge'][t+x] = 0.0
                    else:
                        self.bat_state['P_discharge'][t+x] = 0.0
                        self.bat_state['I_discharge'][t+x] = 0.0
                        self.bat_state['P_charge'][t+x] = - StateBatt.StatePack.P/1000.
                        self.bat_state['I_charge'][t+x] = - StateBatt.StatePack.I/1000.
                    
                    self.bat_state['temperature'][t+x] = StateBatt.StatePack.T_batt
                    batt_max_temp = max(batt_max_temp, StateBatt.StatePack.T_batt)

            # Only store infromation if passe the previous day simulation (used in clustering)
            if i >= Nprev_sim:  
                print("Max Battery Temperature for the Day: {0:5.2f} C".format(batt_max_temp))
                sol = int(start_time/self.dispatch_solution)
                self.bat_state['max_temp_persolution'][sol] = batt_max_temp
            
                #===============================================
                # ========== Post-Day Calculations =============
                #===============================================

                # ========== Generation, Revenue, Curtailment =======================
                for x in range(self.dispatch_solution):
                    self.compare['generation_wo_battery'][t+x] = HP.OptModel.Wpv[x]() + HP.OptModel.Wwf[x]() if HP.OptModel.Wpv[x]() + HP.OptModel.Wwf[x]() < HP.OptModel.Wnet[x]() else HP.OptModel.Wnet[x]()
                    self.compare['generation_dispatch'][t+x] = HP.OptModel.wdotWF[x]() + HP.OptModel.wdotPV[x]() + (self.bat_state['P_discharge'][t+x] - self.bat_state['P_charge'][t+x])*1000. 
                    self.compare['revenue_wo_battery'][t+x] = HP.OptModel.Delta()*HP.OptModel.P[x]()*self.compare['generation_wo_battery'][t+x]
                    self.compare['revenue_dispatch'][t+x] = HP.OptModel.Delta()*HP.OptModel.P[x]()*self.compare['generation_dispatch'][t+x]
                    self.compare['curtailment_wo_battery'][t+x] = HP.OptModel.Wpv[x]() + HP.OptModel.Wwf[x]() - HP.OptModel.Wnet[x]() if HP.OptModel.Wpv[x]() + HP.OptModel.Wwf[x]() > HP.OptModel.Wnet[x]() else 0.0
                    self.compare['curtailment_dispatch'][t+x] = HP.OptModel.Wpv[x]() + HP.OptModel.Wwf[x]() - HP.OptModel.wdotWF[x]() - HP.OptModel.wdotPV[x]()
                    self.compare['capfactor_wo_battery'][t+x] = self.compare['generation_wo_battery'][t+x]/HP.OptModel.Wnet[x]()
                    self.compare['capfactor_dispatch'][t+x] = self.compare['generation_dispatch'][t+x]/HP.OptModel.Wnet[x]()

                # ====== battery lifecycle count ==========
                if HP.simplebatt:
                    # power accounting
                    self.bat_dispatch['apxblc'][sol] = (HP.OptModel.Delta()/HP.OptModel.CB())*sum((HP.OptModel.gamma()**x)*HP.OptModel.wdotBC[x]() for x in range(self.dispatch_solution))
                else:
                    # current accounting - McCormick envelope
                    self.bat_dispatch['apxblc'][sol] = (HP.OptModel.Delta()/HP.OptModel.CB())*sum(0.8*HP.OptModel.iN[x]() - 0.8*HP.OptModel.zN[x]() for x in range(self.dispatch_solution))
                    # Calculate value base on non-linear relationship
                    self.bat_dispatch['clacblc'][sol] = (HP.OptModel.Delta()/HP.OptModel.CB())*sum(HP.OptModel.iN[x]()*(0.8 - 0.8*(self.bsoc0 if x == 0 else HP.OptModel.bsoc[x-1]())) for x in range(self.dispatch_solution))
                    #dis_blc[i] = (HP.OptModel.Delta()/HP.OptModel.CB())*sum((HP.OptModel.gamma()**t)*(HP.OptModel.iP[t]()) for t in range(dispatch_solution))

                self.bat_dispatch['PHblc'][sol] = HP.OptModel.blc()

                # ========== Objective Function Comparsion ============
                self.compare['OFV_dispatch'][sol] = sum((HP.OptModel.gamma()**t)*HP.OptModel.Delta()*HP.OptModel.P[t]()*(HP.OptModel.wdotS[t]() - HP.OptModel.wdotP[t]()) 
                                        - ((1/HP.OptModel.gamma())**t)*HP.OptModel.Delta()*(HP.OptModel.Cpv()*HP.OptModel.wdotPV[t]() 
                                                                            + HP.OptModel.Cwf()*HP.OptModel.wdotWF[t]() 
                                                                            + HP.OptModel.CbP()*HP.OptModel.wdotBC[t]() 
                                                                            + HP.OptModel.CbN()*HP.OptModel.wdotBD[t]()) 
                                        for t in range(self.dispatch_solution)) - HP.OptModel.Clc()*HP.OptModel.blc()

                self.compare['OFV_wo_battery'][sol] = sum((HP.OptModel.gamma()**t)*HP.OptModel.Delta()*HP.OptModel.P[t]()*(HP.OptModel.Wpv[t]() + HP.OptModel.Wwf[t]() if HP.OptModel.Wpv[t]() + HP.OptModel.Wwf[t]() < HP.OptModel.Wnet[t]() else HP.OptModel.Wnet[t]()) 
                                        - ((1/HP.OptModel.gamma())**t)*HP.OptModel.Delta()*(HP.OptModel.Cpv()*HP.OptModel.Wpv[t]()
                                                                            + HP.OptModel.Cwf()*HP.OptModel.Wwf[t]() ) for t in range(self.dispatch_solution))

                self.compare['OFV_diff'][sol] = self.compare['OFV_dispatch'][sol] - self.compare['OFV_wo_battery'][sol]

                ### ============ Outputs ===============
                # wind and solar plant outputs
                # Dealing with the end of analysis period
                if self.Nsteps - t < self.dispatch_solution:
                    sol_len = self.Nsteps - t
                else:
                    sol_len = self.dispatch_solution
                
                self.bat_dispatch['wind'][t:t+sol_len] = HP.OptModel.wdotWF[:]()[0:sol_len]
                self.bat_dispatch['solar'][t:t+sol_len] = HP.OptModel.wdotPV[:]()[0:sol_len]
                self.bat_dispatch['net'][t:t+sol_len] = HP.OptModel.wdotS[:]()[0:sol_len]
                self.bat_dispatch['Price'][t:t+sol_len] = HP.OptModel.P[:]()[0:sol_len]

                # TODO: keep track of battery charge and discharge from the dispatch algorithm
                ## currently these are power into and out of the battery without losses
                self.bat_dispatch['P_charge'][t:t+sol_len] = HP.OptModel.wdotBC[:]()[0:sol_len]
                self.bat_dispatch['P_discharge'][t:t+sol_len] = HP.OptModel.wdotBD[:]()[0:sol_len]
                self.bat_dispatch['SOC'][t:t+sol_len] = HP.OptModel.bsoc[:]()[0:sol_len]
                if not HP.simplebatt:
                    self.bat_dispatch['I_charge'][t:t+sol_len] = HP.OptModel.iP[:]()[0:sol_len]
                    self.bat_dispatch['I_discharge'][t:t+sol_len] = HP.OptModel.iN[:]()[0:sol_len]

            print(HP.OptModel.bsocm[:]())

            # store state-of-charge
            bsoc0 = StateBatt.StatePack.SOC/100.
            self.bsoc0 = bsoc0

            if istest:
                if i == 5:
                    break

if __name__ == '__main__':
    """
    Example script to run the hybrid optimization performance platform (HOPP)
    Custom analysis should be placed a folder within the hybrid_analysis project, which includes HOPP as a submodule
    https://github.com/hopp/hybrid_analysis
    """
    istest = False
    isclustering = False  ## TODO: Clustering requires weighting factors 
    # user inputs:
    dispatch_horizon = 48 #168 #48 # hours
    dispatch_solution = 24 #24 # dispatch solution provided for every 24 hours, simulation advances X hours at a time

    # O&M costs per technology
    solar_OM = 13 # $13/kW/year -> https://www.nrel.gov/docs/fy17osti/68023.pdf
    wind_OM = 43 # $43/kW/year -> https://www.nrel.gov/docs/fy18osti/72167.pdf
    batt_OM = 3 # $3/kW/year

    # define hybrid system and site
    solar_mw = 70
    wind_mw = 50
    interconnect_mw = 50

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
    hybrid_plant.solar.system_capacity_kw = solar_mw * 1000
    hybrid_plant.wind.system_capacity_by_num_turbines(wind_mw * 1000)

    actual_solar_pct = hybrid_plant.solar.system_capacity_kw / \
                           (hybrid_plant.solar.system_capacity_kw + hybrid_plant.wind.system_capacity_kw)

    logger.info("Run with solar percent {}".format(actual_solar_pct))

    # Simulate hybrid system (without storage)
    hybrid_plant.simulate()

    # annual energy production
    annual_energies = hybrid_plant.annual_energies
    hybrid_aep_mw = annual_energies.Hybrid / 1000
    cf_interconnect = 100 * hybrid_aep_mw / (interconnect_mw * 8760)

    # capacity factors
    capacity_factors = hybrid_plant.capacity_factors

    # net present value
    npvs = hybrid_plant.net_present_values # in dollars

    # IRRs: seems to be not working
    irrs = hybrid_plant.internal_rate_of_returns

    ############################### BATTERY STARTS HERE ########################
    analysis_period = 1 # years
    steps_in_year = 8760 # currently hours in year, multiply this for subhourly tests (example * 12 for 5 minute tests)
    days_in_year = 365

    # Battery Specifications
    desired_power = 50000           # [kW] 
    desired_capacity = 200000.      # [kWh]
    desired_voltage = 500.          # [Volts]
    isDisBatSimple = False          # True for simple dispatch battery, False for detailed dispatch battery model

    # # Create the model using PySAM's defaults
    battery = battery_model.default("GenericBatterySingleOwner") # this models has to run a full year
    
    battery_size_specs = battery_model_sizing(battery, desired_power, desired_capacity, desired_voltage)
    calcMassSurfArea(battery)
    # Set up inputs needed by the model.
    battery.BatteryCell.batt_room_temperature_celsius = [25] * (steps_in_year * analysis_period) # degrees C, room temperature. Would normally come from weather file
    #battery.BatteryCell.batt_h_to_ambient = 5000.0 # Water-Cooled?
    #battery.BatteryCell.batt_maximum_SOC = 85.0
    #battery.BatteryCell.batt_minimum_SOC = 30.0

    # If linear constraint is made from exp to nom point, Dispatch problem gets alot harder??? - A, B voltage calculation needs work!
    #battery.BatteryCell.batt_Vexp = 3.9
    #battery.BatteryCell.batt_Qexp = 0.85

    ## Creating Stateful battery object
    StateBatt = bt.new()
    setStatefulUsingStandAlone(StateBatt, battery)
    if isDisBatSimple:
        StateBatt.value("control_mode", 1.0)    # Power control
        control_var = "input_power"
    else:
        StateBatt.value("control_mode", 0.0)    # Current control
        control_var = "input_current"

    ############################### Dispatch model Set-up ########################
    
    if False: # using simple battery calculation model in hybrid_dispatch.py
        # Creating battery system - In Hybrid dispatch - outdated
        cell = batteryCell()
        battery = SimpleBattery(cell, desired_capacity, desired_voltage, desired_power)
        bsoc0 = 0.5
    else:
        bsoc0 = battery.BatteryCell.batt_initial_SOC/100.

    # Initializing dispatch
    HP = dispatch_problem(dispatch_horizon, battery, simplebatt = isDisBatSimple)

    ## TODO: update operating costs
    CbP = 0.0       #0.002
    CbN = batt_OM/8760.   #0.002
    if HP.battery.__class__.__name__ == 'StandAloneBattery':
        Clc = 0.06*HP.battery.BatterySystem.batt_computed_bank_capacity
    elif HP.battery.__class__.__name__ == 'SimpleBattery':
        Clc = 0.06*HP.battery.nomC
    Clc /= 100.
    print("Battery Life Cycle Cost: $"+ str(Clc))
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

    # Creating price data for dispatch optimization
    pricedata = True
    P = []
    if pricedata:
        price_data = "pricing-data-2015-IronMtn-002_factors.csv"
        with open(price_data) as csv_file:
            csv_reader = csv.reader(csv_file)
            count = 0
            for row in csv_reader:
                if float(row[0]) <= 0:   # Removing negative prices causing infeasability in dispatch problem (need to address)
                    print("Price is negative at timestep {:d}, Price of {:5.2f} replace with approximately zero".format(count, float(row[0])))
                    P.append(float(0.0001))
                else:
                    P.append(float(row[0]))
                count += 1
    else:
        np.random.seed(0)
        for i in range(int(8)):
            P.extend([np.random.rand()]*3)
        P_day = copy.deepcopy(P)
        for i in range(364):
            P.extend(P_day)
    P = [x/10. for x in P]  # [$/kWh]

    ############################### Set-up Data Clustering ########################
    if isclustering:
        # Clustering parameters
        Ndays = int(dispatch_horizon/24)        # assumes hourly time steps
        Nprev = int(dispatch_solution/24)
        Nnext = int((dispatch_horizon - dispatch_solution)/24)           # Use 1 subsequent day if dispatch optimization is enabled
        Nclusters = 20
        run_continuous = False
        initial_state = bsoc0      # None (uses default), numerical value, or 'heuristic'.  Note 'heuristic' can only be used if simulation is not run continuously
        cluster_inputs = None
        weatherfile = site.solar_resource.filename

        inputs = {}
        inputs['run_continuous_with_skipped_days'] = run_continuous 
        inputs['nprev'] = Nprev
        inputs['initial_charge'] = initial_state    #state-of-charge (TES)
        for key in ['nday', 'day_start', 'group_weight', 'avg_ppamult', 'avg_sfadjust']:
            inputs[key] = None

        if cluster_inputs is None:  # Re-calculate cluster inputs
            cluster_inputs = clustering.setup_clusters(weatherfile, P, Nclusters, Ndays, Nprev, Nnext)
        for key in cluster_inputs.keys():
            inputs[key] = cluster_inputs[key]   #This was used if cluster_inputs were already known
        
        # Combine consecutive exemplars into a single simulation
        days = inputs['day_start']
        base_weight = inputs['weights']
        ## TODO: remove sf_adjust_tot
        sf_adjust_tot = [1.]*len(P) #dummy data
        avg_sfadjust = clustering.compute_cluster_avg_from_timeseries(sf_adjust_tot, inputs['partition_matrix'], Ndays = Ndays, Nprev = Nprev, Nnext = Nnext,adjust_wt = True, k1 = inputs['first_pt_cluster'], k2 = inputs['last_pt_cluster'])
        avg_ppamult = inputs['avg_ppamult']
        combined = clustering.combine_consecutive_exemplars(days, base_weight, avg_ppamult, avg_sfadjust, Ndays, Nprev = Nprev, Nnext = Nnext)  # Combine simulations of consecutive exemplars
        inputs['day_start'] = combined['start_days']
        inputs['nday'] = combined['Nsim_days'] 
        inputs['avg_ppamult'] = combined['avg_ppa']
        inputs['group_weight'] = combined['weights']
        inputs['avg_sfadjust'] = combined['avg_sfadj']

    ############################### Dispatch Optimization Simulation with Rolling Horizon ########################
    ## Annual Simulation
    dis_wind = copy.deepcopy(ts_wind)
    dis_solar = copy.deepcopy(ts_solar)

    ts_wnet = [interconnect_mw*1000]* steps_in_year

    S = sim_with_dispatch(dispatch_horizon, dispatch_solution, dis_solar, dis_wind, P, ts_wnet, istest=True)
    S.setupBatteryandDispatchMods(HP, StateBatt)

    start_time = time.time()
    if isclustering:
        Ng = len(inputs['day_start'])       # Number of simulations 
        for g in range(Ng):
            d1 = inputs['day_start'][g] - Nprev               # First day to be included in simulation group g
            Nprev_sim = inputs['day_start'][g] - max(0,d1)    # Number of previous days actually allowed  in the simulation
            Ndays_tot = inputs['nday'][g] + Nprev #+ Nnext     # Number of days to be simulated
            tstart = int(24.*(inputs['day_start'][g] - Nprev_sim)) # TODO: fix the 24
            S.simulate(tstart, Ndays_tot, bsoc0, Nprev_sim = Nprev_sim)

        clusters = {'exemplars':inputs['exemplars'], 'partition_matrix':inputs['partition_matrix']}
        res_dicts = ['bat_dispatch', 'compare', 'bat_state']      # Should I calculate 'compare' by it self?
        for r in res_dicts:
            res_dict = getattr(S,r)
            for key in res_dict.keys():
                if len(res_dict[key]) > 365:
                    res_dict[key]  = clustering.compute_annual_array_from_clusters(res_dict[key], clusters, Ndays, adjust_wt = True, k1 = inputs['first_pt_cluster'], k2 = inputs['last_pt_cluster'])
                else:
                    print('Partial data contained in ' + r + '['+ key + '].  ' + 'Current clustering methods can not fill array with representive data.')
    else:
        ti = np.arange(0, steps_in_year, dispatch_solution) # in hours
        ### Create a simulate_with_dispatch function
        for i,t in enumerate(ti):
            print('Evaluating day ', i, ' out of ', len(ti))
            S.simulate(t, 1, bsoc0)
            bsoc0 = S.bsoc0

    #===================== End of Annual Simulation =======================

    elapsed_time = time.time() - start_time
    print("Elapsed time: {0:5.2f} Minutes".format(elapsed_time/60.0))

    bat_dispatch = S.bat_dispatch
    compare = S.compare
    bat_state = S.bat_state

    bat_dispatch['I'] = [DC - C for (DC, C) in zip(bat_dispatch['I_discharge'], bat_dispatch['I_charge'])]
    bat_dispatch['P'] = [DC - C for (DC, C) in zip(bat_dispatch['P_discharge'], bat_dispatch['P_charge'])]

    # =================== Annual Metrics Comparison =======================
    metrics = ['generation', 'revenue', 'curtailment','capfactor', 'OFV']
    units = ['[GWhe]','[$M]','[GWhe]','[%]','[$M]']
    for met in metrics:
        compare['ann_'+met+'_wo_battery'] = sum(compare[met+'_wo_battery'])
        compare['ann_'+met+'_dispatch'] = sum(compare[met+'_dispatch'])
        if met == 'capfactor':
            compare['ann_'+met+'_wo_battery'] *= 100./len(compare[met+'_wo_battery'])
            compare['ann_'+met+'_dispatch'] *= 100./len(compare[met+'_dispatch'])
        compare['ann_'+met+'_diff'] = compare['ann_'+met+'_dispatch'] - compare['ann_'+met+'_wo_battery']
        compare['ann_'+met+'_rel_diff'] = (compare['ann_'+met+'_diff']/compare['ann_'+met+'_wo_battery']) * 100.

    print('#'*10 + ' Comparison Table '+ '#'*10)
    table_header = 'Metric' + '\t\t |\t' + 'Units' + '\t | \t' + 'W/O Battery' + '\t | \t' + 'W/ Dispatch' +  '\t | \t' + 'Difference' + '\t | \t' + 'Rel Diff'
    print(table_header)
    for met,unit in zip(metrics,units):
        if met == 'capfactor':
            convert = 1.
        else:
            convert = 1.e6
        
        if len(met)< 9:
            tmet = met + '\t'
        else:
            tmet = met
        cols = ['_wo_battery','_dispatch','_diff','_rel_diff']
        table_line = tmet + '\t | \t' + unit + '\t | \t'
        print(tmet + '\t | \t' + unit + '\t | \t {0:^11.2f} \t | \t {1:^11.2f} \t | \t {2:^10.2f} \t | \t {3:^8.2f}'.format(compare['ann_'+met+'_wo_battery']/convert, compare['ann_'+met+'_dispatch']/convert, compare['ann_'+met+'_diff']/convert, compare['ann_'+met+'_rel_diff']))

    #====================== Capacity Payments ================================
    cap_pay = {}
    cap_pay['price'] = P[0:len(compare['generation_wo_battery'])]
    cap_pay['gen_wo_bat'] = compare['generation_wo_battery']
    cap_pay['gen_disp'] = [x if x <interconnect_mw*1e3 else interconnect_mw*1e3 for x in compare['generation_dispatch']]  # TODO: should fix this
    cap_pay['net_lim'] = [interconnect_mw*1000]*len(cap_pay['gen_wo_bat'])

    CPdf = pd.DataFrame(cap_pay)
    CPdf = CPdf.sort_values(by=['price'], ascending=False)
    CPdf = CPdf.reset_index()
    CPdf['cumCF_wo_bat'] = CPdf.gen_wo_bat.cumsum()/CPdf.net_lim.cumsum()
    CPdf['cumCF_disp'] = CPdf.gen_disp.cumsum()/CPdf.net_lim.cumsum()

    Nf = 14
    plt.figure()
    plt.plot([x for x in range(len(CPdf['cumCF_wo_bat']))], CPdf['cumCF_wo_bat'], linewidth = 2.5, label = 'Hybrid without battery')
    plt.plot([x for x in range(len(CPdf['cumCF_wo_bat']))], CPdf['cumCF_disp'], linewidth = 2.5, label = 'Hybrid with battery dispatch')
    plt.xlim([0,100])
    plt.tick_params(which='both', labelsize=Nf)
    plt.ylabel('Capacity Credit Fraction [-]', fontsize=Nf)
    plt.xlabel('Number of Highest-priced Hours', fontsize=Nf)
    plt.legend(fontsize=Nf)
    plt.tight_layout()
    plt.show()

    #=======================================================================
    #=======================================================================
    ### TODO: NEED to update and solve StandAloneBattery Module
    if False: # testing
        # 24 hours of data to duplicate for the test. Would need to add data here for subhourly
        lifetime_generation = []
        lifetime_dispatch = []
        daily_generation = [0]*24
        #daily_generation = [0, 0, 0, 0, 0, 0, 0, 200, 400, 600, 800, 1000, 1000, 1000, 1000, 800, 600, 400, 200, 0, 0, 0, 0, 0] # kW
        daily_dispatch = [0, 0, 0, 0, 0, 0, 0, -200, -400, -600, -800, -1000, -1000, 0, 0, 200, 400, 600, 800, 1000, 1000, 0, 0, 0] #kW, negative is charging

        # Extend daily lists for entire analysis period
        for i in range(0, days_in_year * analysis_period):
            lifetime_generation.extend(daily_generation)
            lifetime_dispatch.extend(daily_dispatch)

        # Normally output from pvsamv1, need to set up custom system generation here
        battery.SystemOutput.gen = lifetime_generation # converts list to tuple

        # set the lifetime analysis period to 1
        battery.Lifetime.system_use_lifetime_output = 1
        battery.Lifetime.analysis_period = 1

        # Change from default dispatch to custom dispatch
        battery.BatteryDispatch.batt_dispatch_auto_can_gridcharge = 1.0 # True, allows generation = 0
        battery.BatteryDispatch.batt_dispatch_choice = 3 # custom dispatch
        battery.BatteryDispatch.batt_custom_dispatch = lifetime_dispatch

        # Run the model. Change argument to 1 for verbose
        battery.execute(1)

        # Export outputs to a dictionary. All outputs on readthedocs page are exported
        output = battery.export()
        print("Roundtrip efficiency: " + str(output["Outputs"]["average_battery_roundtrip_efficiency"]))
        print("Battery cycles over lifetime: " + str(max(output["Outputs"]["batt_cycles"])))
    
    # tracking battery lifecycles for the year            
    tot_apxblc = sum(bat_dispatch['apxblc'])
    tot_calcblc = sum(bat_dispatch['clacblc'])

    if tot_apxblc == 0.0:
        Error_ratio = None
    else:
        Error_ratio = tot_calcblc/tot_apxblc

    print("McCormick Battery Lifecycles: {0:5.2f}".format(tot_apxblc))
    print("Non-linear Calculation Battery Lifecycles: {0:5.2f}".format(tot_calcblc))
    print("Error ratio: {0:5.2f}".format(Error_ratio))
    print("StateBattery number of cycles: {0:5.2f}".format(StateBatt.StateCell.n_cycles))

    ############# Setting up finanicial models #############
    #wind_fin = PySAM.Singleowner.new()
    if False:
        plt.figure()
        plt.hist(bat_state['SOC'][::dispatch_solution])
        plt.xlabel('Begining of day SOC [-]')
        plt.ylabel('Number of Occurences')
        plt.show()







    ############## Plotting error between dispatch and state battery model
    Nf = 14 # fontsize
    plt.figure(figsize=(15,15))
    saveplots = False
    
    # First sub-plot SOC
    if HP.simplebatt:
        Nsubplt = 2
    else:
        Nsubplt = 3

    subplt = 1
    plt.subplot(2,Nsubplt,subplt)
    plt.plot([0,1.0],[0,1.0], 'r--')
    plt.scatter(bat_dispatch['SOC'], bat_state['SOC'])
    plt.tick_params(which='both', labelsize=Nf)
    plt.ylabel('SOC (state model) [-]', fontsize=Nf)
    plt.xlabel('SOC (dispatch model) [-]', fontsize=Nf)
    subplt+=1

    bat_dispatch['P'] = [x/1000. for x in bat_dispatch['P']]    
    plt.subplot(2,Nsubplt,subplt)
    maxpoint = max(max(bat_dispatch['P']), max(bat_state['P']))
    minpoint = min(min(bat_dispatch['P']), min(bat_state['P']))
    maxpoint*= 1.025
    minpoint*= 1.025
    plt.plot([minpoint,maxpoint],[0,0], 'k--')
    plt.plot([0,0],[minpoint,maxpoint], 'k--')
    plt.text(minpoint*0.85, minpoint, "Charging", fontsize=Nf)
    plt.text(maxpoint*0.01, maxpoint, "Discharging", fontsize=Nf)

    plt.plot([minpoint,maxpoint],[minpoint,maxpoint], 'r--')
    plt.scatter(bat_dispatch['P'], bat_state['P'])
    plt.tick_params(which='both', labelsize=Nf)
    plt.ylabel('Power (state model) [MW]', fontsize=Nf)
    plt.xlabel('Power (dispatch model) [MW]', fontsize=Nf)
    subplt+=1

    if not HP.simplebatt:
        plt.subplot(2,Nsubplt,subplt)
        maxpoint = max(max(bat_dispatch['I']), max(bat_state['I']))
        minpoint = min(min(bat_dispatch['I']), min(bat_state['I']))
        maxpoint*= 1.025
        minpoint*= 1.025
        plt.plot([minpoint,maxpoint],[0,0], 'k--')
        plt.plot([0,0],[minpoint,maxpoint], 'k--')
        plt.text(minpoint*0.85, minpoint, "Charging", fontsize=Nf)
        plt.text(maxpoint*0.01, maxpoint, "Discharging", fontsize=Nf)

        plt.plot([minpoint,maxpoint],[minpoint,maxpoint], 'r--')
        plt.scatter(bat_dispatch['I'], bat_state['I'])
        plt.tick_params(which='both', labelsize=Nf)
        plt.ylabel('Current (state model) [kA]', fontsize=Nf)
        plt.xlabel('Current (dispatch model) [kA]', fontsize=Nf)
        subplt+=1

    plt.subplot(2,Nsubplt,subplt)
    plt.hist([state - dispatch for (state, dispatch) in zip(bat_state['SOC'], bat_dispatch['SOC'])], alpha = 0.5)
    plt.tick_params(which='both', labelsize=Nf)
    plt.ylabel('Number of Occurrences', fontsize=Nf)
    plt.xlabel('SOC Error (state) - (dispatch) [-]', fontsize=Nf)
    subplt+=1

    plt.subplot(2,Nsubplt,subplt)
    bat_dispatch['P_charge'] = [x/1000. for x in bat_dispatch['P_charge']]  
    bat_dispatch['P_discharge'] = [x/1000. for x in bat_dispatch['P_discharge']]  
    cP_err = [state - dispatch for (state, dispatch) in zip(bat_state['P_charge'], bat_dispatch['P_charge']) if abs(state - dispatch) > 1e-10]
    dcP_err = [state - dispatch for (state, dispatch) in zip(bat_state['P_discharge'], bat_dispatch['P_discharge']) if abs(state - dispatch) > 1e-10]
    min_err = min(cP_err + dcP_err)
    max_err = max(cP_err + dcP_err)
    bins = [x for x in range(int(min_err-1),int(max_err+1))]
    
    plt.hist(cP_err, bins, alpha = 0.5, label = 'Charging')
    plt.hist(dcP_err, bins, alpha = 0.5, label = 'Discharging')

    plt.tick_params(which='both', labelsize=Nf)
    plt.ylabel('Number of Occurrences', fontsize=Nf)
    plt.xlabel('Power Error (state) - (dispatch) [-]', fontsize=Nf)
    plt.legend(fontsize=Nf)
    subplt+=1

    if not HP.simplebatt:
        plt.subplot(2,Nsubplt,subplt)
        cI_err = [state - dispatch for (state, dispatch) in zip(bat_state['I_charge'], bat_dispatch['I_charge']) ]
        dcI_err = [state - dispatch for (state, dispatch) in zip(bat_state['I_discharge'], bat_dispatch['I_discharge']) ]
        min_err = min(cI_err + dcI_err)
        max_err = max(cI_err + dcI_err)
        bins = [x for x in range(int(min_err-1),int(max_err+1))]

        plt.hist(cI_err, bins, alpha = 0.5, label = 'Charging')
        plt.hist(dcI_err, bins, alpha = 0.5, label = 'Discharging')

        plt.tick_params(which='both', labelsize=Nf)
        plt.ylabel('Number of Occurrences', fontsize=Nf)
        plt.xlabel('Current Error (state) - (dispatch) [-]', fontsize=Nf)
        plt.legend(fontsize=Nf)
        subplt+=1

    if saveplots:
        plt.savefig("state_v_dispatch.png")
    else:
        plt.show()

    if True:
        ############## Plotting an example set of the solution
        tt = np.linspace(0, steps_in_year, steps_in_year) # plotting time array
        StartD = 10#150  #350 #65
        Np = 5 # number of dispatch time horizon to plot out
        Nf = 14 # fontsize
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
        ax = plt.gca()
        ax.xaxis.set_ticks(np.arange(st, et, 24))
        plt.grid()
        plt.tick_params(which='both', labelsize=Nf)
        plt.ylabel('Power (MW)', fontsize=Nf)
        plt.title('Generation Resources', fontsize=Nf)
        plt.legend(fontsize=Nf,loc='upper left')

        # Battery action
        plt.subplot(3,1,2)
        plt.bar(tt[st:et],[x*power_scale for x in bat_dispatch['P_discharge']][st:et], width = 0.9 ,color = 'blue', edgecolor = 'white',label='Battery Discharge')
        plt.bar(tt[st:et],[-x*power_scale for x in bat_dispatch['P_charge']][st:et], width = 0.9 ,color = 'red', edgecolor = 'white',label='Battery Charge')    
        plt.xlim([st,et])
        ax = plt.gca()
        ax.xaxis.set_ticks(np.arange(st, et, 24))
        plt.grid()
        ax1 = plt.gca()
        ax1.legend(fontsize=Nf,loc='upper left')
        ax1.set_ylabel('Power (MW)', fontsize=Nf)

        ax2 = ax1.twinx()
        ax2.plot(tt[st:et], bat_dispatch['SOC'][st:et], 'k', label='State-of-Charge')
        ax2.plot(tt[st:et], bat_state['SOC'][st:et], '.', label='StateFul Battery')
        ax2.set_ylabel('Stat-of-Charge (-)', fontsize=Nf)
        ax2.legend(fontsize=Nf,loc='upper right')

        plt.tick_params(which='both',labelsize=Nf)
        plt.title('Battery Power Flow', fontsize=Nf)
        
        # Net action
        plt.subplot(3, 1, 3)
        plt.plot(tt[st:et], [x*power_scale for x in ts_solar + ts_wind][st:et], 'k--', label='Original Generation')
        plt.plot(tt[st:et], [x*power_scale for x in bat_dispatch['net']][st:et], 'g', label='Optimized Dispatch')
        plt.xlim([st,et])
        ax = plt.gca()
        ax.xaxis.set_ticks(np.arange(st, et, 24))
        plt.grid()
        ax1 = plt.gca()
        ax1.legend(fontsize=Nf,loc='upper left')
        ax1.set_ylabel('Power (MW)', fontsize=Nf)

        ax2 = ax1.twinx()
        ax2.plot(tt[st:et], bat_dispatch['Price'][st:et], 'r', label='Price')
        ax2.set_ylabel('Grid Price ($/kWh)', fontsize=Nf)
        ax2.legend(fontsize=Nf,loc='upper right')

        plt.tick_params(which='both', labelsize=Nf)
        plt.xlabel('Time (hours)', fontsize=Nf)
        plt.title('Net Generation', fontsize=Nf)
        
        if saveplots:
            plt.savefig("system_gen_vTime.png")
        else:
            plt.show()






