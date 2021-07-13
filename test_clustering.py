import clustering
from clustering import cluster
import numpy as np
import copy
import csv

if __name__ == "__main__":
    
    weatherfile = "resource_files/solar/40.0167_-105.25_tm2_60.csv"
    # TODO: update pricing
    pricedata = True
    P = []
    if pricedata:
        price_data = "pricing-data-2015-IronMtn-002_factors.csv"
        with open(price_data) as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                P.append(float(row[0]))
    else:
        np.random.seed(0)
        for i in range(int(8)):
            P.extend([np.random.rand()]*3)
        P_day = copy.deepcopy(P)
        for i in range(364):
            P.extend(P_day)
        P = [x/10. for x in P]  # [$/kWh]

    ### This was within objective_function.py simulation functions for handle clustering approach
    Ndays = 2
    Nprev = 1
    Nnext = 1           # Use 1 subsequent day if dispatch optimization is enabled
    Nclusters = 20
    run_continuous = False
    initial_state = None       # None (uses default), numerical value, or 'heuristic'.  Note 'heuristic' can only be used if simulation is not run continuously
    cluster_inputs = None

    inputs = {}
    inputs['run_continuous_with_skipped_days'] = run_continuous 
    inputs['nprev'] = Nprev
    inputs['initial_charge'] = initial_state    #state-of-charge (TES)
    for key in ['nday', 'day_start', 'group_weight', 'avg_ppamult', 'avg_sfadjust']:
        inputs[key] = None

    if cluster_inputs is None:  # Re-calculate Cluster inputs
        cluster_inputs = clustering.setup_clusters(weatherfile, P, Nclusters, Ndays, Nprev, Nnext)
    for key in cluster_inputs.keys():
        inputs[key] = cluster_inputs[key]   #This was used if cluster_inputs were already known
    
    # Set initial TES state-of-charge (TODO: update to battery SOC)   
    if type(inputs['initial_charge']) is float or type(inputs['initial_charge']) is int:
        vd['csp.pt.tes.init_hot_htf_percent'] = inputs['initial_charge']

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
    Ng = len(inputs['day_start'])       # Number of simulations 

    ### TODO: set-up simulation and create annual arrays (create_annual_array_with_cluster_average_values) or (compute_annual_array_from_clusters)

    """

    ### This is from sdk out-of-date
    # Run simulations
    steps_per_day = 24
    s = {}
    hourly_results_keys = ['generation', 'pricing_mult', 'sf_thermal_gen', 'disp_qsfprod_expected', 'disp_wpb_expected']
    for key in hourly_results_keys:
        s[key] = [0.0 for i in range(nrec)]     # Array of yearly generation

    #### I don't need to use this part (ssc specific)
        # Simulate continuously, but only allow full calculations on selected days
    if inputs['run_continuous_with_skipped_days']:  

        select_days = [0 for v in range(365)]
        for g in range(Ng):    # Number of simulation groupings  
            d1 = inputs['day_start'][g] - Nprev     # First day to be included in simulation group g 
            for j in range(inputs['nday'][g] + Nprev ):
                if (d1+j)>=0 and (d1+j)<365:
                    select_days[d1 + j] = 1           
        vd['select_simulation_days'] = select_days

        # Set up array of Cluster-average solar field availability
        sf_adjust_hourly = list(sf_adjust_tot)    # Hourly adjustment factors from previous calculation  
        if inputs['avg_sfadjust'] is not None:
            sf_adjust_hourly = dni_clustering.create_annual_array_with_cluster_average_values(sf_adjust_hourly, inputs['avg_sfadjust'], inputs['day_start'], inputs['nday'], Nprev = inputs['nprev'], Nnext = Nnext, overwrite_surrounding_days = True)
            vd['sf_adjust:hourly'] = [100.*(1.-v) for v in sf_adjust_hourly]
            
        sgroup = pysdk.run_simulation(vd, design)      # Run continuous simulation

    ### This runs simulations for each group
    for g in range(Ng):     

        # Run simulation only for group g
        if not inputs['run_continuous_with_skipped_days']:    # Run simulation for group g
            d1 = inputs['day_start'][g] - Nprev               # First day to be included in simulation group g 
            Nprev_sim = inputs['day_start'][g] - max(0,d1)    # Number of previous days actually allowed  in the simulation
            Ndays_tot = inputs['nday'][g] + Nprev + Nnext     # Number of days to be simulated
            tstart = int(max(0, d1*24.*3600.))   
            tend = int(min(8760.*3600., (d1+Ndays_tot)*24.*3600.)) 
            vd['time_start'] = tstart
            vd['time_stop'] = tend
            vd['vacuum_arrays'] = True 
            vd['time_steps_per_hour'] = wf_steps_per_hour

            # Update solar field hourly adjustment factors to reflect Cluster-average values
            if inputs['avg_sfadjust'] is not None:
                sf_adjust_hourly_new = list(sf_adjust_tot)
                Nsim = inputs['nday'][g] + Nprev + Nnext    
                for h in range(Nsim*steps_per_day):
                    time_pt = (inputs['day_start'][g] - Nprev)*steps_per_day+ h 
                    if h>=0 and h<8760*wf_steps_per_hour:
                        sf_adjust_hourly_new[time_pt] = inputs['avg_sfadjust'][g][h]
                vd['sf_adjust:hourly'] = [100.*(1.-v) for v in sf_adjust_hourly_new]
            else:
                vd['sf_adjust:hourly'] = [100.*(1.-v) for v in sf_adjust_tot]


            # Update initial charge state
            if type(inputs['initial_charge']) is float or type(inputs['initial_charge']) is int:
                vd['csp.pt.tes.init_hot_htf_percent'] = inputs['initial_charge']
            elif inputs['initial_charge'] is 'heuristic':
                sm = vd['solarm']
                h = max(0,(d1-1)*24)   # First hour in day prior to first simulated day 
                avgdni = sum(dni[h+6:h+18]) / 12. 
                if avgdni<=500:
                    vd['csp.pt.tes.init_hot_htf_percent'] = 5.
                else:
                    if sm<1.5:
                        vd['csp.pt.tes.init_hot_htf_percent'] = 5.
                    elif sm <2:
                        vd['csp.pt.tes.init_hot_htf_percent'] = 10.
                    else:
                        vd['csp.pt.tes.init_hot_htf_percent'] = 20.

            sgroup = pysdk.run_simulation(vd, design) # Run simulation for selected set of days

        # Collect simulation results in full annual array 
        for d in range(inputs['nday'][g]):                  # Days counted in simulation grouping
            day_of_year = inputs['day_start'][g] + d          
            if inputs['run_continuous_with_skipped_days']:    # Simulation output contains full annual array
                i = day_of_year * steps_per_day     
            else:                                              # Simulation output contains only simulated subset of days
                i = (Nprev_sim+d)*steps_per_day    

            for key in hourly_results_keys:
                for h in range(steps_per_day):    # Hours in day d
                    s[key][day_of_year*steps_per_day + h] = sgroup[key][i+h]


    clusters = {'exemplars':inputs['exemplars'], 'partition_matrix':inputs['partition_matrix']}
    
    for key in hourly_results_keys:
        s[key]  = dni_clustering.compute_annual_array_from_clusters(s[key], clusters, Ndays, adjust_wt = True, k1 = inputs['first_pt_cluster'], k2 = inputs['last_pt_cluster'])
        
    s['pricing_mult'] = self.params.dispatch_factors_ts
    s['total_installed_cost'] = sgroup['total_installed_cost']
    s['system_capacity'] = sgroup['system_capacity']
    
    s['annual_generation'] = 0.0
    s['revenue_units'] = 0.0
    for h in range(len(s['generation'])):
        s['annual_generation'] += s['generation'][h] / float(wf_steps_per_hour)
        s['revenue_units'] += s['generation'][h] * s['pricing_mult'][h] / float(wf_steps_per_hour)
    """
    pass