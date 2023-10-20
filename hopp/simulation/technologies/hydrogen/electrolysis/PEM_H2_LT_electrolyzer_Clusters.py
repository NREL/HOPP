## Low-Temperature PEM Electrolyzer Model
"""
Python model of H2 PEM low-temp electrolyzer.

Quick Hydrogen Physics:

1 kg H2 <-> 11.1 N-m3 <-> 33.3 kWh (LHV) <-> 39.4 kWh (HHV)

High mass energy density (1 kg H2= 3,77 l gasoline)
Low volumetric density (1 Nm³ H2= 0,34 l gasoline

Hydrogen production from water electrolysis (~5 kWh/Nm³ H2)

Power:1 MW electrolyser <-> 200 Nm³/h  H2 <-> ±18 kg/h H2
Energy:+/-55 kWh of electricity --> 1 kg H2 <-> 11.1 Nm³ <-> ±10 liters
demineralized water

Power production from a hydrogen PEM fuel cell from hydrogen (+/-50%
efficiency):
Energy: 1 kg H2 --> 16 kWh
"""
import math
import numpy as np
import sys
import pandas as pd
from matplotlib import pyplot as plt
import scipy
import rainflow
from scipy import interpolate

np.set_printoptions(threshold=sys.maxsize)

# def calc_current(P_T,p1,p2,p3,p4,p5,p6): #calculates i-v curve coefficients given the stack power and stack temp
#     pwr,tempc=P_T
#     i_stack=p1*(pwr**2) + p2*(tempc**2)+ (p3*pwr*tempc) +  (p4*pwr) + (p5*tempc) + (p6)
#     return i_stack 
def calc_current(P_T,p1,p2,p3,p4,p5,p6): #calculates i-v curve coefficients given the stack power and stack temp
    pwr,tempc=P_T
    # i_stack=p1*(pwr**2) + p2*(tempc**2)+ (p3*pwr*tempc) +  (p4*pwr) + (p5*tempc) + (p6)
    i_stack=p1*(pwr**3) + p2*(pwr**2) +  (p3*pwr) + (p4*pwr**(1/2)) + p5
    return i_stack 

class PEM_H2_Clusters:
    """
    Create an instance of a low-temperature PEM Electrolyzer System. Each
    stack in the electrolyzer system in this model is rated at 1 MW_DC.

    Parameters
    _____________
    np_array P_input_external_kW
        1-D array of time-series external power supply

    string voltage_type
        Nature of voltage supplied to electrolyzer from the external power
        supply ['variable' or 'constant]

    float power_supply_rating_MW
        Rated power of external power supply

    Returns
    _____________

    """
    #Remove: estimate_lifetime_capacity_factor
    #Remove: make_lifetime_performance_df_all_opt [x]
    def __init__(self, cluster_size_mw, plant_life, user_defined_EOL_percent_eff_loss, eol_eff_percent_loss=[],user_defined_eff = False,rated_eff_kWh_pr_kg=[],include_degradation_penalty=True,turndown_ratio=0.1,dt=3600):
        #self.input_dict = input_dict
        # print('RUNNING CLUSTERS PEM')
        self.set_max_h2_limit=False # TODO: add as input
        self.plant_life_years = plant_life
        if user_defined_eff:
            self.create_system_for_target_eff(rated_eff_kWh_pr_kg)
        
        self.include_deg_penalty = include_degradation_penalty
        self.use_onoff_deg = True
        self.use_uptime_deg= True
        self.use_fatigue_deg = True
        reset_uptime_deg_to_target = True #this re-calculates the uptime
        #degradation rate to align with 80,000 operational hrs at 0.97 CF 
        #of no fatigue and no on/off cycles to be end-of-life

        self.output_dict = {}
        self.dt=dt
        self.max_stacks = cluster_size_mw
        self.stack_input_voltage_DC = 250 #unused

        # Assumptions:
        self.min_V_cell = 1.62  # Only used in variable voltage scenario
        self.p_s_h2_bar = 31  # H2 outlet pressure

        # self.stack_input_current_lower_bound = 400 #[A] any current below this amount (10% rated) will saturate the H2 production to zero, used to be 500 (12.5% of rated)
        self.stack_rating_kW = 1000  # 1 MW
        self.cell_active_area = 1920#1250 #[cm^2]
        self.N_cells = 130
        self.membrane_thickness=0.018 #cm
        self.cell_max_current_density = 2 #[A/cm^2]
        self.max_cell_current=self.cell_max_current_density*self.cell_active_area #PEM electrolyzers have a max current density of approx 2 A/cm^2 so max current is 2*cell_area
        # self.stack_input_current_lower_bound = 0.1*self.max_cell_current
        self.stack_input_current_lower_bound = turndown_ratio*self.max_cell_current
        
        self.turndown_ratio = turndown_ratio

        # Constants:
        self.moles_per_g_h2 = 0.49606 #[1/weight_h2]
        self.V_TN = 1.48  # Thermo-neutral Voltage (Volts) in standard conditions
        self.F = 96485.34  # Faraday's Constant (C/mol) or [As/mol]
        self.R = 8.314  # Ideal Gas Constant (J/mol/K)
        self.eta_h2_hhv=39.41

        #Additional Constants
        self.T_C = 80 #stack temperature in [C]
        self.mmHg_2_Pa = 133.322 #convert between mmHg to Pa
        self.patmo = 101325 #atmospheric pressure [Pa]
        self.mmHg_2_atm = self.mmHg_2_Pa/self.patmo #convert from mmHg to atm

        #Default degradation values
        
        self.onoff_deg_rate=1.47821515e-04 #[V/off-cycle]
        self.rate_fatigue = 3.33330244e-07 #multiply by rf_track
        
        self.curve_coeff=self.iv_curve() #this initializes the I-V curve to calculate current
        

        self.make_BOL_efficiency_curve()
        if user_defined_EOL_percent_eff_loss:
            self.eol_eff_drop = eol_eff_percent_loss/100
            self.d_eol=self.find_eol_voltage_val(eol_eff_percent_loss)
            self.find_eol_voltage_curve(eol_eff_percent_loss)
        else:
            self.eol_eff_drop = 0.1
            eol_eff_percent_loss = 10
            self.d_eol=self.find_eol_voltage_val(eol_eff_percent_loss)
            self.find_eol_voltage_curve(eol_eff_percent_loss)
            # self.d_eol = 0.7212

        if reset_uptime_deg_to_target:
            self.steady_deg_rate=self.reset_uptime_degradation_rate()
        else:
            self.steady_deg_rate=1.41737929e-10 #[V/s] 

        
    def run(self,input_external_power_kw):
        startup_time=600 #[sec]
        startup_ratio = 1-(startup_time/self.dt)
        #saturate input power at rated power
        input_power_kw = self.external_power_supply(input_external_power_kw)
        #determine whether stack is on or off
        #0: off 1: on
        self.cluster_status = self.system_design(input_power_kw,self.max_stacks)
        #calculate number of on/off cycles
        #no delay at beginning of sim
        cluster_cycling = [0] + list(np.diff(self.cluster_status)) 
        cluster_cycling = np.array(cluster_cycling)
        
        #how much to reduce h2 by based on cycling status
        h2_multiplier = np.where(cluster_cycling > 0, startup_ratio, 1)
        #number of "stacks" on at a single time
        self.n_stacks_op = self.max_stacks*self.cluster_status
        #n_stacks_op is now either number of pem per cluster or 0 if cluster is off!

        #split power evenly amongst all stacks
        power_per_stack = np.where(self.n_stacks_op>0,input_power_kw/self.n_stacks_op,0)
        #calculate current from power
        stack_current =calc_current((power_per_stack,self.T_C), *self.curve_coeff)

        if self.include_deg_penalty:
            V_init=self.cell_design(self.T_C,stack_current)
            _,deg_signal=self.full_degradation(V_init)
            
            stack_current=self.find_equivalent_input_power_4_deg(power_per_stack,V_init,deg_signal) #fixed
            V_cell_equiv = self.cell_design(self.T_C,stack_current) #mabye this isn't necessary
            V_cell = V_cell_equiv + deg_signal
        else:
            V_init=self.cell_design(self.T_C,stack_current)
            _,deg_signal=self.full_degradation(V_init)
            # lifetime_performance_df =self.make_lifetime_performance_df_all_opt(deg_signal,V_init,power_per_stack)
            V_cell=self.cell_design(self.T_C,stack_current) #+self.total_Vdeg_per_hr_sys
            
        
        time_until_replacement,stack_life = self.calc_stack_replacement_info(deg_signal)
        annual_performance = self.make_yearly_performance_dict(power_per_stack,deg_signal,V_init,I_op=[],grid_connected=False)
        # self.make_yearly_performance_dict(power_per_stack,lifetime_performance_df['Time until replacement [hours]'],deg_signal,V_init) #TESTING
        stack_power_consumed = (stack_current * V_cell * self.N_cells)/1000
        system_power_consumed = self.n_stacks_op*stack_power_consumed
        
        h2_kg_hr_system_init = self.h2_production_rate(stack_current,self.n_stacks_op)
        # h20_gal_used_system=self.water_supply(h2_kg_hr_system_init)
        p_consumed_max,rated_h2_hr = self.rated_h2_prod()
        #scales h2 production to account for start-up time if going from off->on
        h2_kg_hr_system = h2_kg_hr_system_init * h2_multiplier 

        h20_gal_used_system=self.water_supply(h2_kg_hr_system)

        pem_cf = np.sum(h2_kg_hr_system)/(rated_h2_hr*len(input_power_kw)*self.max_stacks)
        efficiency = self.system_efficiency(input_power_kw,stack_current) #Efficiency as %-HHV
        
        h2_results={}
        h2_results_aggregates={}
        h2_results['Input Power [kWh]'] = input_external_power_kw
        h2_results['hydrogen production no start-up time']=h2_kg_hr_system_init
        h2_results['hydrogen_hourly_production']=h2_kg_hr_system
        h2_results['water_hourly_usage_kg'] =h20_gal_used_system*3.79
        h2_results['electrolyzer_total_efficiency_perc'] = efficiency
        h2_results['kwh_per_kgH2'] = input_power_kw / h2_kg_hr_system
        h2_results['Power Consumed [kWh]'] = system_power_consumed
        h2_results_aggregates['Warm-Up Losses on H2 Production'] = np.sum(h2_kg_hr_system_init) - np.sum(h2_kg_hr_system)
        
        
        h2_results_aggregates['Stack Life [hours]'] = stack_life
        h2_results_aggregates['Time until replacement [hours]'] = time_until_replacement
        h2_results_aggregates['Stack Rated Power Consumed [kWh]'] = p_consumed_max
        h2_results_aggregates['Stack Rated H2 Production [kg/hr]'] = rated_h2_hr
        h2_results_aggregates['Cluster Rated Power Consumed [kWh]'] = p_consumed_max*self.max_stacks
        h2_results_aggregates['Cluster Rated H2 Production [kg/hr]'] = rated_h2_hr*self.max_stacks
        h2_results_aggregates['gal H20 per kg H2'] = np.sum(h20_gal_used_system)/np.sum(h2_kg_hr_system)
        h2_results_aggregates['Stack Rated Efficiency [kWh/kg]'] = p_consumed_max/rated_h2_hr
        h2_results_aggregates['Cluster Rated H2 Production [kg/yr]'] = rated_h2_hr*len(input_power_kw)*self.max_stacks
        h2_results_aggregates['Operational Time / Simulation Time (ratio)'] = self.percent_of_sim_operating #added
        h2_results_aggregates['Fraction of Life used during sim'] = self.frac_of_life_used #added
        #TODO: add results for stack replacement stuff based on RATED voltage, not distribution

        # h2_results_aggregates['Number of Lifetime Cluster Replacements'] = nsr_life
        h2_results_aggregates['PEM Capacity Factor (simulation)'] = pem_cf
        
        h2_results_aggregates['Total H2 Production [kg]'] =np.sum(h2_kg_hr_system)
        h2_results_aggregates['Total Input Power [kWh]'] =np.sum(input_external_power_kw)
        h2_results_aggregates['Total kWh/kg'] =np.sum(input_external_power_kw)/np.sum(h2_kg_hr_system)
        h2_results_aggregates['Total Uptime [sec]'] = np.sum(self.cluster_status * self.dt)
        h2_results_aggregates['Total Off-Cycles'] = np.sum(self.off_cycle_cnt)
        h2_results_aggregates['Final Degradation [V]'] =self.cumulative_Vdeg_per_hr_sys[-1]
        # h2_results_aggregates['IV curve coeff'] = self.curve_coeff
        # h2_results_aggregates.update(lifetime_performance_df.to_dict()) 
        h2_results_aggregates['Performance By Year'] = annual_performance #double check if errors
        
        []
        return h2_results, h2_results_aggregates
        # return h2_results_aggregates
   

    def find_eol_voltage_curve(self,eol_eff_percent_loss):
        eol_eff_mult = (100+eol_eff_percent_loss)/100
        i_bol = self.output_dict['BOL Efficiency Curve Info']['Current'].values
        V_bol = self.output_dict['BOL Efficiency Curve Info']['Cell Voltage'].values
        h2_bol = self.output_dict['BOL Efficiency Curve Info']['H2 Produced'].values
        h2_eol = h2_bol/eol_eff_mult

        i_eol_no_faradaic_loss=(h2_eol*1000*2*self.F*self.moles_per_g_h2)/(1*self.N_cells*self.dt)
        n_f=self.faradaic_efficiency(i_eol_no_faradaic_loss)
        i_eol =  (h2_eol*1000*2*self.F*self.moles_per_g_h2)/(n_f*self.N_cells*self.dt)
        self.d_eol_curve=((i_bol*V_bol)/i_eol) - V_bol #simple method
        
        

    def find_equivalent_input_power_4_deg(self,power_in_kW,V_init,V_deg):
        #TODO: rename function since it outputs current
        I_in = calc_current((power_in_kW,self.T_C), *self.curve_coeff)
        eff_mult = (V_init + V_deg)/V_init #(1 + eff drop)
        I_deg = I_in/eff_mult

        return I_deg



    def full_degradation(self,voltage_signal):
        #TODO: add reset if hits end of life degradation limit!
        voltage_signal = voltage_signal*self.cluster_status
        if self.use_uptime_deg:
            V_deg_uptime = self.calc_uptime_degradation(voltage_signal)
        else:
            V_deg_uptime=np.zeros(len(voltage_signal))
        if self.use_onoff_deg:
            V_deg_onoff = self.calc_onoff_degradation()
        else:
            V_deg_onoff = np.zeros(len(voltage_signal))
        
        V_signal = voltage_signal + np.cumsum(V_deg_uptime) + np.cumsum(V_deg_onoff)
        if self.use_fatigue_deg:
            V_fatigue=self.approx_fatigue_degradation(V_signal)
        else:
            V_fatigue=np.zeros(len(voltage_signal))
        deg_signal = np.cumsum(V_deg_uptime) + np.cumsum(V_deg_onoff) + V_fatigue

        
        self.cumulative_Vdeg_per_hr_sys=deg_signal
        voltage_final=voltage_signal + deg_signal
        
        self.output_dict['Cumulative Degradation Breakdown']=pd.DataFrame({'Uptime':np.cumsum(V_deg_uptime),'On/off':np.cumsum(V_deg_onoff),'Fatigue':V_fatigue})
        return voltage_final, deg_signal
   
    
    def calc_stack_replacement_info(self,deg_signal):
        """Stack life optimistic estimate based on rated efficiency"""
        #[V] degradation at end of simulation
        d_sim = deg_signal[-1] 
        
        #fraction of life that has been "spent" during simulation
        frac_of_life_used = d_sim/self.d_eol_curve[-1]
        #number of hours simulated
        sim_time_dt = len(deg_signal) 
        #number of hours operating
        operational_time_dt=np.sum(self.cluster_status) 
        
        #time between replacement [hrs] based on number of hours operating
        t_eod_operation_based = (1/frac_of_life_used)*operational_time_dt #[hrs]
        #time between replacement [hrs] based on simulation length
        t_eod_existance_based = (1/frac_of_life_used)*sim_time_dt
        #CF shouldn't be different
        #just report out one capacity factor
        self.frac_of_life_used = frac_of_life_used
        self.percent_of_sim_operating = operational_time_dt/sim_time_dt

        return t_eod_existance_based,t_eod_operation_based
    
    def make_yearly_performance_dict(self,power_in_kW,V_deg,V_cell,I_op,grid_connected):
        #NOTE: this is not the most accurate for cases where simulation length is not close to 8760
        #I_op only needed if grid connected, should be singular value
        refturb_schedule = np.zeros(self.plant_life_years)
        # refturb_period=int(np.floor(time_between_replacements/8760))
        # refturb_schedule[refturb_period:int(self.plant_life_years):refturb_period]=1
        
        sim_length = len(V_cell)
        
        death_threshold = self.d_eol_curve[-1]
        
        cluster_cycling = [0] + list(np.diff(self.cluster_status)) #no delay at beginning of sim
        cluster_cycling = np.array(cluster_cycling)
        startup_ratio = 1-(600/3600)#TODO: don't have this hard-coded
        h2_multiplier = np.where(cluster_cycling > 0, startup_ratio, 1)
        
        _,rated_h2_pr_stack_BOL=self.rated_h2_prod()
        rated_h2_pr_sim = rated_h2_pr_stack_BOL*self.max_stacks*sim_length

        kg_h2_pr_sim = np.zeros(int(self.plant_life_years))
        capfac_per_sim = np.zeros(int(self.plant_life_years))
        d_sim = np.zeros(int(self.plant_life_years))
        power_pr_yr_kWh = np.zeros(int(self.plant_life_years))
        Vdeg0 = 0
        
        for i in range(int(self.plant_life_years)): #assuming sim is close to a year
            V_deg_pr_sim = Vdeg0 + V_deg
            
            # it_died = any(V_deg_pr_sim>death_threshold)
            if np.max(V_deg_pr_sim)>death_threshold:
                #it died
                idx_dead = np.argwhere(V_deg_pr_sim>death_threshold)[0][0]
                V_deg_pr_sim = np.concatenate([V_deg_pr_sim[0:idx_dead],V_deg[idx_dead:sim_length]])
                # i_sim_dead = i
                refturb_schedule[i]=self.max_stacks
                
            if not grid_connected:
                stack_current = self.find_equivalent_input_power_4_deg(power_in_kW,V_cell,V_deg_pr_sim)
                h2_kg_hr_system_init = self.h2_production_rate(stack_current,self.n_stacks_op)
                # total_sim_input_power = self.max_stacks*np.sum(power_in_kW)
                power_pr_yr_kWh[i] = self.max_stacks*np.sum(power_in_kW)
            else:
                h2_kg_hr_system_init = self.h2_production_rate(I_op,self.n_stacks_op)
                h2_kg_hr_system_init = h2_kg_hr_system_init*np.ones(len(power_in_kW))
                annual_power_consumed_kWh = self.max_stacks*I_op*(V_cell + V_deg_pr_sim)*self.N_cells/1000
                # total_sim_input_power = np.sum(annual_power_consumed_kWh)
                power_pr_yr_kWh[i] = np.sum(annual_power_consumed_kWh)

            h2_kg_hr_system = h2_kg_hr_system_init*h2_multiplier
            kg_h2_pr_sim[i] = np.sum(h2_kg_hr_system)
            capfac_per_sim[i] = np.sum(h2_kg_hr_system)/rated_h2_pr_sim
            d_sim[i] = V_deg_pr_sim[sim_length-1]
            Vdeg0 = V_deg_pr_sim[sim_length-1]
        performance_by_year = {}
        year = np.arange(0,int(self.plant_life_years),1)
        
        performance_by_year['Capacity Factor [-]'] = dict(zip(year,capfac_per_sim))
        performance_by_year['Refurbishment Schedule [MW replaced/year]'] = dict(zip(year,refturb_schedule))
        performance_by_year['Annual H2 Production [kg/year]'] = dict(zip(year,kg_h2_pr_sim))
        performance_by_year['Annual Average Efficiency [kWh/kg]'] = dict(zip(year,power_pr_yr_kWh/kg_h2_pr_sim))
        performance_by_year['Annual Average Efficiency [%-HHV]'] = dict(zip(year,self.eta_h2_hhv/(power_pr_yr_kWh/kg_h2_pr_sim)))
        performance_by_year['Annual Energy Used [kWh/year]'] = dict(zip(year,power_pr_yr_kWh))

        return performance_by_year

        
    def reset_uptime_degradation_rate(self):
        #TODO: make ref_operational_hours an input
        ref_operational_hours_life = 80000 #50-60k
        #make the ref_operational_hours_life an input
        ref_cf=0.97
        ref_operational_hours = ref_operational_hours_life*ref_cf
        I_max = calc_current((self.stack_rating_kW,self.T_C),*self.curve_coeff)
        V_rated = self.cell_design(self.T_C,I_max)
        new_deg_rate=self.d_eol/(V_rated*ref_operational_hours*3600)
        return new_deg_rate

    def calc_uptime_degradation(self,voltage_signal):
        #steady_deg_rate = 1.12775521e-09
        steady_deg_per_hr=self.dt*self.steady_deg_rate*voltage_signal*self.cluster_status
        cumulative_Vdeg=np.cumsum(steady_deg_per_hr)
        self.output_dict['Total Uptime [sec]'] = np.sum(self.cluster_status * self.dt)
        self.output_dict['Total Uptime Degradation [V]'] = cumulative_Vdeg[-1]

        return steady_deg_per_hr
        
    def calc_onoff_degradation(self):
        
        
        change_stack=np.diff(self.cluster_status)
        cycle_cnt = np.where(change_stack < 0, -1*change_stack, 0)
        cycle_cnt = np.array([0] + list(cycle_cnt))
        self.off_cycle_cnt = cycle_cnt
        stack_off_deg_per_hr= self.onoff_deg_rate*cycle_cnt
        self.output_dict['System Cycle Degradation [V]'] = np.cumsum(stack_off_deg_per_hr)[-1]
        self.output_dict['Off-Cycles'] = cycle_cnt
        return stack_off_deg_per_hr

    def approx_fatigue_degradation(self,voltage_signal,dt_fatigue_calc_hrs=168):
        #should not use voltage values when voltage_signal = 0
        #aka - should only be counted when electrolyzer is on
        
        t_calc=np.arange(0,len(voltage_signal)+dt_fatigue_calc_hrs ,dt_fatigue_calc_hrs ) 
        v_max=np.max(voltage_signal)
        v_min=np.min(voltage_signal)
        if v_max==v_min:
            rf_sum=0
            lifetime_fatigue_deg=0
            V_fatigue_ts=np.zeros(len(voltage_signal))

        else:

            rf_cycles = rainflow.count_cycles(voltage_signal, nbins=10)
            rf_sum = np.sum([pair[0] * pair[1] for pair in rf_cycles])
            lifetime_fatigue_deg=rf_sum*self.rate_fatigue
            self.output_dict['Approx Total Fatigue Degradation [V]'] = lifetime_fatigue_deg
            rf_track=0
            V_fatigue_ts=np.zeros(len(voltage_signal))
            for i in range(len(t_calc)-1):
                voltage_signal_temp = voltage_signal[np.nonzero(voltage_signal[t_calc[i]:t_calc[i+1]])]
                if np.size(voltage_signal_temp)==0:
                    rf_sum = 0
                else:
                    v_max=np.max(voltage_signal_temp)
                    v_min=np.min(voltage_signal_temp)

                    if v_max == v_min:
                        rf_sum=0
                    else:
                        
                        rf_cycles=rainflow.count_cycles(voltage_signal_temp, nbins=10)
                    # rf_cycles=rainflow.count_cycles(voltage_signal[t_calc[i]:t_calc[i+1]], nbins=10)
                        rf_sum=np.sum([pair[0] * pair[1] for pair in rf_cycles])
                rf_track+=rf_sum
                V_fatigue_ts[t_calc[i]:t_calc[i+1]]=rf_track*self.rate_fatigue
                #already cumulative!
            self.output_dict['Sim End RF Track'] = rf_track #TODO: remove
            self.output_dict['Total Actual Fatigue Degradation [V]'] = V_fatigue_ts[-1] #TODO: remove

        return V_fatigue_ts #already cumulative!

    def grid_connected_func(self,h2_kg_hr_system_required):
        """
        Calculate power and current required to meet a constant hydrogen demand
        """
        # df=self.output_dict['BOL Efficiency Curve Info'][['H2 Produced','Current','Power Sent [kWh]','Power Consumed [kWh]']]

        max_h2kg_single_stack=self.h2_production_rate(self.max_cell_current,1)
        # EOL_max_h2_stack=self.h2_production_rate(self.max_cell_current,1)
        min_n_stacks=np.ceil(h2_kg_hr_system_required/max_h2kg_single_stack)
        if min_n_stacks>self.max_stacks:
            print("ISSUE")
        h2_per_stack_min=h2_kg_hr_system_required/self.max_stacks #change var name
        
        I_reqd_BOL_noFaradaicLoss=(h2_per_stack_min*1000*2*self.F*self.moles_per_g_h2)/(1*self.N_cells*self.dt)
        n_f=self.faradaic_efficiency(I_reqd_BOL_noFaradaicLoss)
        I_reqd=(h2_per_stack_min*1000*2*self.F*self.moles_per_g_h2)/(n_f*self.N_cells*self.dt)
        V_reqd = self.cell_design(self.T_C,I_reqd)
        #TODO: only include deg if user-requested
        V_deg_per_hr=self.steady_deg_rate*V_reqd*self.dt
        V_steady_deg=np.arange(0,self.d_eol+V_deg_per_hr,V_deg_per_hr)
        P_reqd_per_hr_stack=I_reqd*(V_reqd + V_steady_deg)*self.N_cells/1000 #kW
        P_required_per_hr_system=self.max_stacks*P_reqd_per_hr_stack #kW

        output_system_power = P_required_per_hr_system[0:8760]
        stack_current_signal = I_reqd*np.ones(len(output_system_power))
        return output_system_power, stack_current_signal

        


    def create_system_for_target_eff(self,user_def_eff_perc):
        print("User defined efficiency capability not yet added in electrolyzer model, using default")
        pass

    def find_eol_voltage_val(self,eol_rated_eff_drop_percent):
        #TODO: change this 
        rated_power_idx=self.output_dict['BOL Efficiency Curve Info'].index[self.output_dict['BOL Efficiency Curve Info']['Power Sent [kWh]']==self.stack_rating_kW].to_list()[0]
        rated_eff_df=self.output_dict['BOL Efficiency Curve Info'].iloc[rated_power_idx]
        i_rated=rated_eff_df['Current']
        h2_rated_kg=rated_eff_df['H2 Produced']
        vcell_rated=rated_eff_df['Cell Voltage']
        bol_eff_kWh_per_kg=rated_eff_df['Efficiency [kWh/kg]']
        eol_eff_kWh_per_kg=bol_eff_kWh_per_kg*(1+eol_rated_eff_drop_percent/100)
        eol_power_consumed_kWh = eol_eff_kWh_per_kg*h2_rated_kg
        v_tot_eol=eol_power_consumed_kWh*1000/(self.N_cells*i_rated)
        d_eol = v_tot_eol - vcell_rated
        return d_eol


    def system_efficiency(self,P_sys,I):
        #
        system_h2_prod_rate=self.h2_production_rate(I,self.n_stacks_op)
        eff_kWh_pr_kg = P_sys/system_h2_prod_rate #kWh/kg
        system_eff= self.eta_h2_hhv/eff_kWh_pr_kg #[%-HHV]
        
        return system_eff #[%]

    def make_BOL_efficiency_curve(self):
        #TODO: remove all other function calls to self.output_dict['BOL Efficiency Curve Info']
        #this should be done differntly
        # power_in_signal=np.arange(0.1,1.1,0.1)*self.stack_rating_kW
        power_in_signal=np.arange(self.turndown_ratio,1.1,0.1)*self.stack_rating_kW
        stack_I = calc_current((power_in_signal,self.T_C),*self.curve_coeff)
        stack_V = self.cell_design(self.T_C,stack_I)
        power_used_signal = (stack_I*stack_V*self.N_cells)/1000
        h2_stack_kg= self.h2_production_rate(stack_I ,1)
        kWh_per_kg=power_in_signal/h2_stack_kg
        power_error_BOL = power_in_signal-power_used_signal
        self.BOL_powerIn2error = interpolate.interp1d(power_in_signal,power_error_BOL)
        data=pd.DataFrame({'Power Sent [kWh]':power_in_signal,'Current':stack_I,'Cell Voltage':stack_V,
        'H2 Produced':h2_stack_kg,'Efficiency [kWh/kg]':kWh_per_kg,
        'Power Consumed [kWh]':power_used_signal,'IV Curve BOL Error [kWh]':power_error_BOL})
        self.output_dict['BOL Efficiency Curve Info']=data
        
        #return kWh_per_kg, power_in_signal



    def rated_h2_prod(self):
        i=self.output_dict['BOL Efficiency Curve Info'].index[self.output_dict['BOL Efficiency Curve Info']['Power Sent [kWh]']==self.stack_rating_kW]
        I_max=self.output_dict['BOL Efficiency Curve Info']['Current'].iloc[i].values[0]
        #I_max = calc_current((self.stack_rating_kW,self.T_C),*self.curve_coeff)
        V_max = self.cell_design(self.T_C,I_max)
        P_consumed_stack_kw = I_max*V_max*self.N_cells/1000
        max_h2_stack_kg= self.h2_production_rate(I_max,1)
        return P_consumed_stack_kw,max_h2_stack_kg

        
    def external_power_supply(self,input_external_power_kw):
        """
        External power source (grid or REG) which will need to be stepped
        down and converted to DC power for the electrolyzer.

        Please note, for a wind farm as the electrolyzer's power source,
        the model assumes variable power supplied to the stack at fixed
        voltage (fixed voltage, variable power and current)

        TODO: extend model to accept variable voltage, current, and power
        This will replicate direct DC-coupled PV system operating at MPP
        """
        power_converter_efficiency = 1.0 #this used to be 0.95 but feel free to change as you'd like
        # if self.input_dict['voltage_type'] == 'constant':
        power_curtailed_kw=np.where(input_external_power_kw > self.max_stacks * self.stack_rating_kW,\
        input_external_power_kw - self.max_stacks * self.stack_rating_kW,0)

        input_power_kw = \
            np.where(input_external_power_kw >
                        (self.max_stacks * self.stack_rating_kW),
                        (self.max_stacks * self.stack_rating_kW),
                        input_external_power_kw)

        self.output_dict['Curtailed Power [kWh]'] = power_curtailed_kw
        
        return input_power_kw
        # else:
        #     pass  # TODO: extend model to variable voltage and current source
    def iv_curve(self):
        """
        This is a new function that creates the I-V curve to calculate current based
        on input power and electrolyzer temperature

        current range is 0: max_cell_current+10 -> PEM have current density approx = 2 A/cm^2

        temperature range is 40 degC : rated_temp+5 -> temperatures for PEM are usually within 60-80degC

        calls cell_design() which calculates the cell voltage
        """
        # current_range = np.arange(0,self.max_cell_current+10,10) 
        current_range = np.arange(self.stack_input_current_lower_bound,self.max_cell_current+10,10) 
        temp_range = np.arange(40,self.T_C+5,5)
        idx = 0
        powers = np.zeros(len(current_range)*len(temp_range))
        currents = np.zeros(len(current_range)*len(temp_range))
        temps_C = np.zeros(len(current_range)*len(temp_range))
        for i in range(len(current_range)):
            
            for t in range(len(temp_range)):
                powers[idx] = current_range[i]*self.cell_design(temp_range[t],current_range[i])*self.N_cells*(1e-3) #stack power
                currents[idx] = current_range[i]
                temps_C[idx] = temp_range[t]
                idx = idx+1
        df=pd.DataFrame({'Power':powers,'Current':currents,'Temp':temps_C}) #added
        temp_oi_idx = df.index[df['Temp']==self.T_C]      #added  
        # curve_coeff, curve_cov = scipy.optimize.curve_fit(calc_current, (powers,temps_C), currents, p0=(1.0,1.0,1.0,1.0,1.0,1.0)) #updates IV curve coeff
        curve_coeff, curve_cov = scipy.optimize.curve_fit(calc_current, (df['Power'][temp_oi_idx].values,df['Temp'][temp_oi_idx].values), df['Current'][temp_oi_idx].values, p0=(1.0,1.0,1.0,1.0,1.0,1.0))
        return curve_coeff

    def system_design(self,input_power_kw,cluster_size_mw):
        """
        Calculate whether the cluster is on or off based on input power

        TODO: add 0.1 (default turndown ratio) as input
        """
        # cluster_min_power = 0.1*self.max_stacks
        # cluster_min_power = 0.1*cluster_size_mw
        cluster_min_power = self.turndown_ratio*cluster_size_mw
        cluster_status=np.where(input_power_kw<cluster_min_power,0,1)
       
        return cluster_status

    def cell_design(self, Stack_T, Stack_Current):
        """
        Basically: calc_cell_voltage
        """
        E_cell = self.calc_reversible_cell_voltage(Stack_T)
        V_act = self.calc_V_act(Stack_T,Stack_Current,self.cell_active_area)
        V_ohmic=self.calc_V_ohmic(Stack_T,Stack_Current,self.cell_active_area,self.membrane_thickness)
        V_cell = E_cell + V_act + V_ohmic
       
        return V_cell
    def calc_reversible_cell_voltage(self,Stack_T):
        """
        inputs::
            Stack_T [C]: operating temperature
        
        returns:: 
            E_cell [V/cell]: reversible overpotential

        Reference:
        
        """
        T_K=Stack_T+ 273.15  # convert Celsius to Kelvin
        #Reversible potential at 25degC - Nerst Equation
        E_rev0 = 1.229  #[V] 
        panode_atm=1 #[atm] total pressure at the anode
        pcathode_atm=1 #[atm] total pressure at the cathode
        patmo_atm=1 #atmospheric prestture
    
        #coefficient for Antoine formulas
        A = 8.07131
        B = 1730.63
        C = 233.426

        #vapor pressure of water in [mmHg] using Antoine formula
        #valid for T<283 Kelvin
        p_h2o_sat_mmHg = 10 ** (A - (B / (C + Stack_T)))  
        #convert mmHg to atm
        p_h20_sat_atm=p_h2o_sat_mmHg*self.mmHg_2_atm  
        #p_H2 = p_cat - p_h2O
        #p_O2 = p_an - p_h2O
        # p_h2O_sat_Pa = (0.61121* np.exp((18.678 - (Stack_T / 234.5)) * (Stack_T / (257.14 + Stack_T)))) * 1e3  # (Pa) #ARDEN-BUCK
        # p_h20_sat_atm=p_h2O_sat_Pa/self.patmo
                # Cell reversible voltage kind of explain in Equations (12)-(15) of below source
        # https://www.sciencedirect.com/science/article/pii/S0360319906000693
        # OR see equation (8) in the source below
        # https://www.sciencedirect.com/science/article/pii/S0360319917309278?via%3Dihub
        #h2 outlet pressure would be 5000 kPa and O2 outlet pressure could be 200 kPa
        #Nerst Equation
        E_cell=E_rev0 + ((self.R*T_K)/(2*self.F))*(np.log(((panode_atm-p_h20_sat_atm)/patmo_atm)*np.sqrt((pcathode_atm-p_h20_sat_atm)/patmo_atm))) 
        return E_cell

    def calc_V_act(self,Stack_T,I_stack,cell_active_area):
        """
        inputs::
            stack_T [C]: operating temperature
            I_stack [A]: stack current
            cell_active_area [cm^2]: electrode area
        
        returns:: 
            V_act [V/cell]: anode and cathode activation overpotential

        Reference:
        
        """
        T_K=Stack_T+ 273.15 
        #current density [A/cm^2]
        i = I_stack/cell_active_area
        # Anode charge transfer coefficient
        a_a = 2  
        # Cathode charge transfer coefficient
        a_c = 0.5  
        #anode exchange current density
        i_o_a = 2 * (10 ** (-7)) 
        #cathode exchange current density
        i_o_c = 2 * (10 ** (-3)) 
        V_anode = (((self.R * T_K) / (a_a * self.F)) * np.arcsinh(i / (2 * i_o_a)))
        V_cathode= (((self.R * T_K) / (a_c * self.F)) * np.arcsinh(i / (2 * i_o_c)))
        V_act = V_anode + V_cathode
        return V_act

    def calc_V_ohmic(self,Stack_T,I_stack,cell_active_area,delta_cm):
        T_K=Stack_T+ 273.15 
        #current density [A/cm^2]
        i = I_stack/cell_active_area
        lambda_water_content = ((-2.89556 + (0.016 * T_K)) + 1.625) / 0.1875
        # membrane proton conductivity [S/cm]
        sigma = ((0.005139 * lambda_water_content) - 0.00326) * np.exp(
            1268 * ((1 / 303) - (1 / T_K)))   
        #ionic resistance [ohms*cm^2]
        R_cell = (delta_cm / sigma) 
        # [ohms*cm^2] from Table 1 in  https://journals.utm.my/jurnalteknologi/article/view/5213/3557
        R_elec=3.5*(10 ** (-5))
        V_ohmic=(i *( R_cell + R_elec)) 
        return V_ohmic
    def dynamic_operation(self): #UNUSED Here but capability is included
        """
        Model the electrolyzer's realistic response/operation under variable RE

        TODO: add this capability to the model
        """
        # When electrolyzer is already at or near its optimal operation
        # temperature (~80degC)
        
        warm_startup_time_secs = 30
        cold_startup_time_secs = 5 * 60  # 5 minutes


    def faradaic_efficiency(self,stack_current): #ONLY EFFICIENCY CONSIDERED RIGHT NOW
        """`
        Text background from:
        [https://www.researchgate.net/publication/344260178_Faraday%27s_
        Efficiency_Modeling_of_a_Proton_Exchange_Membrane_Electrolyzer_
        Based_on_Experimental_Data]

        In electrolyzers, Faraday’s efficiency is a relevant parameter to
        assess the amount of hydrogen generated according to the input
        energy and energy efficiency. Faraday’s efficiency expresses the
        faradaic losses due to the gas crossover current. The thickness
        of the membrane and operating conditions (i.e., temperature, gas
        pressure) may affect the Faraday’s efficiency.

        Equation for n_F obtained from:
        https://www.sciencedirect.com/science/article/pii/S0360319917347237#bib27

        Parameters
        ______________
        float f_1
            Coefficient - value at operating temperature of 80degC (mA2/cm4)

        float f_2
            Coefficient - value at operating temp of 80 degC (unitless)

        np_array current_input_external_Amps
            1-D array of current supplied to electrolyzer stack from external
            power source


        Returns
        ______________

        float n_F
            Faradaic efficiency (unitless)

        """
        f_1 = 250  # Coefficient (mA2/cm4)
        f_2 = 0.996  # Coefficient (unitless)
        I_cell = stack_current * 1000

        # Faraday efficiency
        n_F = (((I_cell / self.cell_active_area) ** 2) /
               (f_1 + ((I_cell / self.cell_active_area) ** 2))) * f_2

        return n_F

    


    def h2_production_rate(self,stack_current,n_stacks_op):
        """
        H2 production rate calculated using Faraday's Law of Electrolysis
        (https://www.sciencedirect.com/science/article/pii/S0360319917347237#bib27)

        Parameters
        _____________

        np_array
            1-D array of current supplied to electrolyzer stack from external
            power source


        Returns
        _____________

        """
        # Single stack calculations:
        n_Tot = self.faradaic_efficiency(stack_current)
        h2_production_rate = n_Tot * ((self.N_cells *
                                       stack_current) /
                                      (2 * self.F))  # mol/s
        h2_production_rate_g_s = h2_production_rate / self.moles_per_g_h2
        h2_produced_kg_hr = h2_production_rate_g_s * (self.dt/1000 ) #3.6 #Fixed: no more manual scaling
        self.output_dict['stack_h2_produced_g_s']= h2_production_rate_g_s
        self.output_dict['stack_h2_produced_kg_hr'] = h2_produced_kg_hr

        # Total electrolyzer system calculations:
        h2_produced_kg_hr_system = n_stacks_op  * h2_produced_kg_hr
        # h2_produced_kg_hr_system = h2_produced_kg_hr
        self.output_dict['h2_produced_kg_hr_system'] = h2_produced_kg_hr_system

        return h2_produced_kg_hr_system


    def water_supply(self,h2_kg_hr):
        """
        Calculate water supply rate based system efficiency and H2 production
        rate
        TODO: Add water-to-hydrogen ratio as input, currently hard-coded to 10
        """
        # ratio of water_used:h2_kg_produced depends on power source
        # h20_kg:h2_kg with PV 22-126:1 or 18-25:1 without PV but considering water deminersalisation
        # stoichometrically its just 9:1 but ... theres inefficiencies in the water purification process
        
        water_used_kg_hr_system = h2_kg_hr * 10
        self.output_dict['water_used_kg_hr'] = water_used_kg_hr_system
        self.output_dict['water_used_kg_annual'] = np.sum(water_used_kg_hr_system)
        water_used_gal_hr_system = water_used_kg_hr_system/3.79
        return water_used_gal_hr_system 

    
    def run_grid_connected_workaround(self,power_input_signal,current_signal):
        #power input signal is total system input power
        #current signal is current per stack
        startup_time=600 #[sec]
        startup_ratio = 1-(startup_time/self.dt)
        self.cluster_status = self.system_design(power_input_signal,self.max_stacks)
        self.n_stacks_op = self.max_stacks*self.cluster_status
        cluster_cycling = [0] + list(np.diff(self.cluster_status)) #no delay at beginning of sim
        cluster_cycling = np.array(cluster_cycling)
        power_per_stack = np.where(self.n_stacks_op>0,power_input_signal/self.n_stacks_op,0)
        
        h2_multiplier = np.where(cluster_cycling > 0, startup_ratio, 1)
        self.n_stacks_op = self.max_stacks*self.cluster_status

        V_init=self.cell_design(self.T_C,current_signal)
        V_cell_deg,deg_signal=self.full_degradation(V_init)
        
        time_until_replacement,stack_life = self.calc_stack_replacement_info(deg_signal)
        annual_performance= self.make_yearly_performance_dict(power_per_stack,deg_signal,V_init,current_signal[0],grid_connected=True) #TESTING
        
        stack_power_consumed = (current_signal * V_cell_deg * self.N_cells)/1000
        system_power_consumed = self.n_stacks_op*stack_power_consumed
        
        h2_kg_hr_system_init = self.h2_production_rate(current_signal,self.n_stacks_op)
        p_consumed_max,rated_h2_hr = self.rated_h2_prod()
        h2_kg_hr_system = h2_kg_hr_system_init * h2_multiplier #scales h2 production to account
        #for start-up time if going from off->on
        h20_gal_used_system=self.water_supply(h2_kg_hr_system)

        pem_cf = np.sum(h2_kg_hr_system)/(rated_h2_hr*len(power_input_signal)*self.max_stacks)
        efficiency = self.system_efficiency(power_input_signal,current_signal)

        h2_results={}
        h2_results_aggregates={}
        h2_results['Input Power [kWh]'] = power_input_signal
        h2_results['hydrogen production no start-up time']=h2_kg_hr_system_init
        h2_results['hydrogen_hourly_production']=h2_kg_hr_system
        # h2_results['water_hourly_usage_gal'] =h20_gal_used_system
        h2_results['water_hourly_usage_kg'] =h20_gal_used_system*3.79
        h2_results['electrolyzer_total_efficiency_perc'] = efficiency
        h2_results['kwh_per_kgH2'] = power_input_signal / h2_kg_hr_system
        h2_results['Power Consumed [kWh]'] = system_power_consumed
        h2_results_aggregates['Warm-Up Losses on H2 Production'] = np.sum(h2_kg_hr_system_init) - np.sum(h2_kg_hr_system)

        h2_results_aggregates['Stack Rated Power Consumed [kWh]'] = p_consumed_max
        h2_results_aggregates['Stack Rated H2 Production [kg/hr]'] = rated_h2_hr
        h2_results_aggregates['Cluster Rated Power Consumed [kWh]'] = p_consumed_max*self.max_stacks
        h2_results_aggregates['Cluster Rated H2 Production [kg/hr]'] = rated_h2_hr*self.max_stacks
        h2_results_aggregates['Stack Rated Efficiency [kWh/kg]'] = p_consumed_max/rated_h2_hr
        h2_results_aggregates['Cluster Rated H2 Production [kg/yr]'] = rated_h2_hr*len(power_input_signal)*self.max_stacks
        h2_results_aggregates['gal H20 per kg H2'] = np.sum(h20_gal_used_system)/np.sum(h2_kg_hr_system)
        h2_results_aggregates['PEM Capacity Factor (simulation)'] = pem_cf
        
        h2_results_aggregates['Operational Time / Simulation Time (ratio)'] = self.percent_of_sim_operating #added
        h2_results_aggregates['Fraction of Life used during sim'] = self.frac_of_life_used #added

        h2_results_aggregates['Total H2 Production [kg]'] =np.sum(h2_kg_hr_system)
        h2_results_aggregates['Total Input Power [kWh]'] =np.sum(power_input_signal)
        h2_results_aggregates['Total kWh/kg'] =np.sum(power_input_signal)/np.sum(h2_kg_hr_system)
        h2_results_aggregates['Total Uptime [sec]'] = np.sum(self.cluster_status * self.dt)
        h2_results_aggregates['Total Off-Cycles'] = np.sum(self.off_cycle_cnt)
        h2_results_aggregates['Final Degradation [V]'] =self.cumulative_Vdeg_per_hr_sys[-1]
        
        h2_results_aggregates['Performance By Year'] = annual_performance #double check if errors
        # h2_results_aggregates['Stack Life Summary'] = self.stack_life_opt

        h2_results_aggregates['Stack Life [hours]'] = stack_life
        h2_results_aggregates['Time until replacement [hours]'] = time_until_replacement

      
        []
        return h2_results, h2_results_aggregates