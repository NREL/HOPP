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
# Updated as of 10/31/2022
import math
import numpy as np
import sys
import pandas as pd
from matplotlib import pyplot as plt
import scipy
from scipy.optimize import fsolve
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

    def __init__(self, cluster_size_mw, plant_life, user_defined_EOL_percent_eff_loss, eol_eff_percent_loss=[],user_defined_eff = False,rated_eff_kWh_pr_kg=[],include_degradation_penalty=True,dt=3600):
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
        self.stack_input_current_lower_bound = 0.1*self.max_cell_current
        

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
        input_power_kw = self.external_power_supply(input_external_power_kw)
        self.cluster_status = self.system_design(input_power_kw,self.max_stacks)
        # cluster_cycling = [self.cluster_status[0]] + list(np.diff(self.cluster_status))
        cluster_cycling = [0] + list(np.diff(self.cluster_status)) #no delay at beginning of sim
        cluster_cycling = np.array(cluster_cycling)
        
        h2_multiplier = np.where(cluster_cycling > 0, startup_ratio, 1)
        self.n_stacks_op = self.max_stacks*self.cluster_status
        #n_stacks_op is now either number of pem per cluster or 0 if cluster is off!

        #self.external_power_supply(electrical_generation_ts,n_stacks_op) 
        power_per_stack = np.where(self.n_stacks_op>0,input_power_kw/self.n_stacks_op,0)
        stack_current =calc_current((power_per_stack,self.T_C), *self.curve_coeff)
        # stack_current =np.where(stack_current >self.stack_input_current_lower_bound,stack_current,0)

        if self.include_deg_penalty:
            V_init=self.cell_design(self.T_C,stack_current)
            V_cell_deg,deg_signal=self.full_degradation(V_init)
            
            lifetime_performance_df =self.make_lifetime_performance_df_all_opt(deg_signal,V_init,power_per_stack)
            # lifetime_performance_df = self.estimate_lifetime_capacity_factor(power_per_stack,V_init,deg_signal) #new
            #below is to find equivalent current (NEW)
            stack_current=self.find_equivalent_input_power_4_deg(power_per_stack,V_init,deg_signal) #fixed
            V_cell_equiv = self.cell_design(self.T_C,stack_current) #mabye this isn't necessary
            V_cell = V_cell_equiv + deg_signal
        else:
            V_init=self.cell_design(self.T_C,stack_current)
            V_ignore,deg_signal=self.full_degradation(V_init)
            lifetime_performance_df =self.make_lifetime_performance_df_all_opt(deg_signal,V_init,power_per_stack)
            V_cell=self.cell_design(self.T_C,stack_current) #+self.total_Vdeg_per_hr_sys
            
        #TODO: Add stack current saturation limit here!
        #if self.set_max_h2_limit:
        #set_max_current_limit(h2_kg_max_cluster,stack_current_unlim,Vdeg,input_power_kW)
        stack_power_consumed = (stack_current * V_cell * self.N_cells)/1000
        system_power_consumed = self.n_stacks_op*stack_power_consumed
        
        h2_kg_hr_system_init = self.h2_production_rate(stack_current,self.n_stacks_op)
        # h20_gal_used_system=self.water_supply(h2_kg_hr_system_init)
        p_consumed_max,rated_h2_hr = self.rated_h2_prod()
        h2_kg_hr_system = h2_kg_hr_system_init * h2_multiplier #scales h2 production to account
        #for start-up time if going from off->on
        h20_gal_used_system=self.water_supply(h2_kg_hr_system)

        pem_cf = np.sum(h2_kg_hr_system)/(rated_h2_hr*len(input_power_kw)*self.max_stacks)
        efficiency = self.system_efficiency(input_power_kw,stack_current)
        # avg_hrs_til_replace=self.simple_degradation()
        # maximum_eff_perc,max_eff_kWhperkg = self.max_eff()
        # total_eff = 39.41*np.sum(h2_kg_hr_system)/np.sum(input_external_power_kw)
        h2_results={}
        h2_results_aggregates={}
        h2_results['Input Power [kWh]'] = input_external_power_kw
        h2_results['hydrogen production no start-up time']=h2_kg_hr_system_init
        h2_results['hydrogen_hourly_production']=h2_kg_hr_system
        h2_results['water_hourly_usage_gal'] =h20_gal_used_system
        h2_results['water_hourly_usage_kg'] =h20_gal_used_system*3.79
        h2_results['electrolyzer_total_efficiency_perc'] = efficiency
        h2_results['kwh_per_kgH2'] = input_power_kw / h2_kg_hr_system
        h2_results['Power Consumed [kWh]'] = system_power_consumed
        h2_results_aggregates['Warm-Up Losses on H2 Production'] = np.sum(h2_kg_hr_system_init) - np.sum(h2_kg_hr_system)
        
        h2_results_aggregates['Stack Rated Power Consumed [kWh]'] = p_consumed_max
        h2_results_aggregates['Stack Rated H2 Production [kg/hr]'] = rated_h2_hr
        h2_results_aggregates['Cluster Rated Power Consumed [kWh]'] = p_consumed_max*self.max_stacks
        h2_results_aggregates['Cluster Rated H2 Production [kg/hr]'] = rated_h2_hr*self.max_stacks

        h2_results_aggregates['Stack Rated Efficiency [kWh/kg]'] = p_consumed_max/rated_h2_hr
        h2_results_aggregates['Cluster Rated H2 Production [kg/yr]'] = rated_h2_hr*len(input_power_kw)*self.max_stacks
        # h2_results_aggregates['Avg [hrs] until Replacement Per Stack'] = self.time_between_replacements #removed
        #below outputs are set in new_calc_stack_replacement_info (I think)
        # h2_results_aggregates['Time between stack replacement [hrs]'] = self.time_between_replacements #added
        # h2_results_aggregates['Operational time between stack replacement [hrs]'] = self.operational_time_between_replacements #added
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
        h2_results_aggregates.update(lifetime_performance_df.to_dict()) 
        # h2_results_aggregates['Stack Life Summary'] = self.stack_life_opt

        h2_results['Stacks on'] = self.n_stacks_op
        h2_results['Power Per Stack [kW]'] = power_per_stack
        h2_results['Stack Current [A]'] = stack_current
        h2_results['V_cell No Deg'] = V_init
        h2_results['V_cell With Deg'] = V_cell
        h2_results['System Degradation [V]']=self.cumulative_Vdeg_per_hr_sys
        
      
        []
        return h2_results, h2_results_aggregates
        # return h2_results_aggregates
    def make_lifetime_performance_df_all_opt(self,deg_signal,V_init,power_per_stack):

        t_eod_existance_based_rated,t_eod_operation_based_rated = self.calc_stack_replacement_info(deg_signal)
        
        # cf = self.estimate_life(power_per_stack,V_init,deg_signal,t_eod_operation_based_rated,t_eod_existance_based_rated)
        old_life_est = self.estimate_lifetime_capacity_factor(power_per_stack,V_init,deg_signal,t_eod_existance_based_rated) #new
        t_eod_opt = [t_eod_operation_based_rated,t_eod_existance_based_rated]
        
        t_eod_desc = ['Stack Life [hours]','Time until replacement [hours]']
        
        if self.include_deg_penalty:
            desc = 'full losses'
        else:
            desc = 'warm-up losses'
        
        df = pd.concat([old_life_est.loc[desc],pd.Series(dict(zip(t_eod_desc,t_eod_opt)))])
        return df
        # return pd.Series(life_data_df)
    def estimate_life(self,power_per_stack,V_cell,V_deg,stack_life,time_until_replacement):
        #this is buggy
        cause_desc = ['Steady','On/Off','Fatigue']
        end_sim_deg_per_cause = np.array([self.output_dict['Total Uptime Degradation [V]'],self.output_dict['System Cycle Degradation [V]'],self.output_dict['Total Actual Fatigue Degradation [V]']])
        percent_of_deg = end_sim_deg_per_cause/V_deg[-1]
        eol_deg_per_cause = self.d_eol_curve[-1]*percent_of_deg
        eol_Vsteady_deg = eol_deg_per_cause[0]
        eol_Vonoff_deg = eol_deg_per_cause[1]
        eol_Vfatigue_deg = eol_deg_per_cause[2]

        frac_of_life_operating = stack_life/time_until_replacement #hours on / sim time
        
        cluster_cycling = [0] + list(np.diff(self.cluster_status)) #no delay at beginning of sim
        cluster_cycling = np.array(cluster_cycling)
        t_sim = len(power_per_stack)
        frac_of_time_on = np.sum(self.cluster_status)/t_sim

        startup_time=600 #[sec]
        startup_ratio = 1-(startup_time/self.dt)
        #number of off-cycles
        #1 if turning off
        offcycle_cnt = np.where(cluster_cycling < 0, -1*cluster_cycling, 0)
        offcycle_cnt = np.array([0] + list(offcycle_cnt))
        ncycles_pr_dt = np.sum(offcycle_cnt)/t_sim
        ncycles_until_replacement = ncycles_pr_dt*time_until_replacement

        h2_multiplier = np.where(cluster_cycling > 0, startup_ratio, 1)
        turned_on_status = np.where(cluster_cycling > 0, 1, 0)
        warmup_mult = np.where(cluster_cycling > 0, 5/6, 1)
        
        power_binwidth_kW = 10
        power_bin_edges = np.arange(0.1*self.stack_rating_kW,self.stack_rating_kW+power_binwidth_kW,power_binwidth_kW)
        power_kW_bins = power_bin_edges[:-1] + (power_binwidth_kW/2)
        # power_kW_bins = np.linspace(0.1,1,50)*self.stack_rating_kW #center point
        # bin_offset_power = (power_kW_bins[1]-power_kW_bins[0])/2
        # power_bin_edges = power_kW_bins - bin_offset_power
        # power_bin_edges = np.insert(power_bin_edges,len(power_bin_edges),power_kW_bins[-1])
        power_cnt,bins = np.histogram(power_per_stack,bins=power_bin_edges)
        below_min_power_cnt = len(power_per_stack) - np.sum(power_cnt)
        # power_pdf = power_cnt/np.sum(power_cnt)
        power_pdf = power_cnt/len(power_per_stack) #probability of operating at a load range relative to time until replacement

        #sum(power_pdf) = frac_of_time_on (sanity check)
        turned_on_power_cnt,bins = np.histogram(turned_on_status*power_per_stack,bins=power_bin_edges)
        #power after being turned on

        I_per_bin = calc_current((power_kW_bins,self.T_C),*self.curve_coeff)
        # W_faradic_loss = self.faradaic_efficiency(I_per_bin)
        h2_nom_per_bin = self.max_stacks*self.h2_production_rate(I_per_bin,n_stacks_op=1)
        V_cell_per_bin = self.cell_design(self.T_C,I_per_bin)
        W_steady_deg = self.dt*self.steady_deg_rate*V_cell_per_bin

        rf_cycles = rainflow.count_cycles(V_cell, nbins=10)
        rf_sum = np.sum([pair[0] * pair[1] for pair in rf_cycles])
        #((eol_Vfatigue_deg/self.rate_fatigue)/rf_sum)*frac_of_time_on approx = stack_life
        #
        #
        V_fatigue_sim=rf_sum*self.rate_fatigue
        V_fatigue_pr_dt = V_fatigue_sim/t_sim #avg
        V_fatigue_until_replacement = V_fatigue_pr_dt*time_until_replacement

        V_deg_onoff_fatigue_until_replacement = V_fatigue_until_replacement + (ncycles_until_replacement*self.onoff_deg_rate)
        V_steady_deg_until_deol = self.d_eol_curve[-1] - V_deg_onoff_fatigue_until_replacement
        life_Vcell_est = V_steady_deg_until_deol/(self.dt*self.steady_deg_rate)
        
        avg_on_hourly_V_deg_fatigue_onoff = (V_fatigue_sim/(t_sim*frac_of_time_on)) + (np.sum(offcycle_cnt)*self.onoff_deg_rate/(t_sim*frac_of_time_on))
        #NEWWW-----
        #self.output_dict['Sim End RF Track']
        #sanity check: np.sum(power_pdf*time_until_replacement) == stack_life
        op_hrs_pr_life = power_pdf*time_until_replacement
        V_steady_deg_life_Wpdf = op_hrs_pr_life*self.dt*self.steady_deg_rate*V_cell_per_bin
        #sanity check: sum(V_steady_deg_life_Wpdf) == eol_Vsteady_deg
        n_life_offcycles = eol_Vonoff_deg/self.onoff_deg_rate
        
        n_life_offhours = (1-frac_of_time_on)*time_until_replacement
        avg_offtime_duration = n_life_offhours/n_life_offcycles
        avg_ontime_between_off = (time_until_replacement-n_life_offhours)/n_life_offcycles
        avg_full_cycle_duration = avg_offtime_duration + avg_ontime_between_off
        V_steady_deg_per_ontime_cycle = (power_pdf*avg_full_cycle_duration)*self.dt*self.steady_deg_rate*V_cell_per_bin
        V_fatigue_deg_pr_ontime_cycle = avg_ontime_between_off*(eol_Vfatigue_deg/stack_life)
        V_cell_pr_ontime_cycle= (power_pdf*avg_full_cycle_duration)*V_cell_per_bin
        H2_nom_per_ontime_cycle = (power_pdf*avg_full_cycle_duration)*h2_nom_per_bin
        I_nom_per_ontime_cycle = (power_pdf*avg_full_cycle_duration)*I_per_bin
        #double check below
        # relative_eff_change_per_ontime_cycle = ((V_cell_pr_ontime_cycle + (V_steady_deg_per_ontime_cycle + (V_fatigue_deg_pr_ontime_cycle/len(power_pdf))))/V_cell_pr_ontime_cycle)-1
        turn_on_power_pdf = turned_on_power_cnt/np.sum(turned_on_power_cnt) 
        #above is only 1 cycle
        n_cycles = np.arange(1,np.ceil(n_life_offcycles)+2,1)
        life_h2_est = 0
        V_deg_track = 0
        warm_up_loss = 0
        deg_loss = 0
        no_loss=0
        steady_deg = np.zeros(len(n_cycles))
        fatigue_deg = np.zeros(len(n_cycles))
        cycle_deg = np.zeros(len(n_cycles))
        tot_deg = np.zeros(len(n_cycles))
        for i,n in enumerate(n_cycles):
            steady_deg[i] = np.sum(n*(V_steady_deg_per_ontime_cycle))
            fatigue_deg[i] = n*(V_fatigue_deg_pr_ontime_cycle)
            cycle_deg[i] = (n-1)*(self.onoff_deg_rate)
            # v_deg_pr_bin = n*V_steady_deg_per_ontime_cycle + (n-1)*self.onoff_deg_rate
            # v_deg_pr_bin = n*(V_steady_deg_per_ontime_cycle + (V_fatigue_deg_pr_ontime_cycle/len(power_pdf))) + ((n-1)*(self.onoff_deg_rate))#((n-1)*(self.onoff_deg_rate/len(power_pdf)))
            v_deg_pr_bin = n*(V_steady_deg_per_ontime_cycle)  + ((n-1)*(self.onoff_deg_rate + V_fatigue_deg_pr_ontime_cycle))#((n-1)*(self.onoff_deg_rate/len(power_pdf)))
            tot_deg[i] = np.sum(v_deg_pr_bin)
            eff_multiplier = (V_cell_per_bin + v_deg_pr_bin)/V_cell_per_bin
            I_deg_per_bin = I_per_bin/eff_multiplier
            h2_actual_pr_bin_kg = self.max_stacks*self.h2_production_rate(I_deg_per_bin,n_stacks_op=1)
            # eff_multiplier = (V_cell_pr_ontime_cycle + v_deg_pr_bin)/V_cell_pr_ontime_cycle
            # h2_actual_pr_bin = H2_nom_per_ontime_cycle/eff_multiplier
            h2_actual_pr_bin = (power_pdf*avg_full_cycle_duration)*h2_actual_pr_bin_kg
            warmup_losses = (startup_time/self.dt)*turn_on_power_pdf*h2_actual_pr_bin
            warm_up_loss +=np.sum(warmup_losses)
            life_h2_est += np.sum(h2_actual_pr_bin)
            deg_loss += (np.sum(H2_nom_per_ontime_cycle)-np.sum(h2_actual_pr_bin))
            no_loss += np.sum(H2_nom_per_ontime_cycle)
            # relative_eff_change_per_ontime_cycle = ((V_cell_pr_ontime_cycle + (V_steady_deg_per_ontime_cycle + (V_fatigue_deg_pr_ontime_cycle/len(power_pdf))))/V_cell_pr_ontime_cycle)-1
            # n_life_offcycles*(np.sum(V_steady_deg_per_ontime_cycle + (V_fatigue_deg_pr_ontime_cycle/len(power_pdf)) + (self.onoff_deg_rate/len(power_pdf))))
        #
        V_deg_eol = steady_deg[-1]+cycle_deg[-1]+fatigue_deg[-1]
        []
        # h2_losses_from_turnon_life = ncycles_until_replacement*(startup_time/self.dt)*h2_nom_per_bin*(turned_on_power_cnt/len(power_per_stack)) #h2(power)*prob_of_turn_on(power)
        p_consumed_max,rated_h2_hr = self.rated_h2_prod()
        life_cf = (life_h2_est-warm_up_loss) / (avg_full_cycle_duration*n*rated_h2_hr)

        # life_cf = (life_h2_est-np.sum(h2_losses_from_turnon_life)) / (avg_full_cycle_duration*n*rated_h2_hr)
        return life_cf
        
        

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
        
        I_in = calc_current((power_in_kW,self.T_C), *self.curve_coeff)
        eff_mult = (V_init + V_deg)/V_init #(1 + eff drop)
        I_deg = I_in/eff_mult

        return I_deg


    def set_max_current_limit(self,h2_kg_max_cluster,stack_current_unlim,Vdeg,input_power_kW):
        #self.stack_input_current_lower_bound
        I_min_for_operation=calc_current((0.1*self.stack_rating_kW,self.T_C),*self.curve_coeff)
        # I_max_for_operation=calc_current((self.stack_rating_kW,self.T_C),*self.curve_coeff)
        # max_cluster_h2=self.h2_production_rate(I_max_for_operation,self.max_stacks)
        min_cluster_h2=self.h2_production_rate(I_min_for_operation,self.max_stacks)
        max_cluster_h2=self.h2_production_rate(self.max_cell_current,self.max_stacks)
        #min_cluster_h2=self.h2_production_rate(self.stack_input_current_lower_bound,self.max_stacks)
        df=self.output_dict['BOL Efficiency Curve Info'][['H2 Produced','Current','Power Sent [kWh]','Power Consumed [kWh]']]
        kg_h2_per_stack=h2_kg_max_cluster/self.max_stacks
        f_i_of_h2=interpolate.interp1d(df['H2 Produced'].values,df['Current'].values)
        I_max=f_i_of_h2(kg_h2_per_stack)
        
        if I_max < I_min_for_operation:
            I_sat=stack_current_unlim
            power_curtailed_kW=np.zeros(len(I_sat))
            change_req_h2=np.abs(h2_kg_max_cluster-min_cluster_h2)
            print("Requested H2 production results in non-operational stack current")
            print("H2 production saturation limit cannot be below {}kg for a {} MW rated stack".format(round(min_cluster_h2,3),self.max_stacks))
            print("Please increase H2 production saturation limit by at least {}kg PER ELECTROLYZER CLUSTER".format(round(change_req_h2,3)))
            print("Running electrolyzer simulation without H2 saturation")
        #elif I_max>I_max_for_operation:
        elif I_max>self.max_cell_current:
            change_req_h2=np.abs(max_cluster_h2-h2_kg_max_cluster)
            I_sat=stack_current_unlim
            power_curtailed_kW=np.zeros(len(I_sat))
            print("Requested H2 production capacity is too high!")
            print("H2 production saturation limit cannot exceed {}kg for a {} MW rated stack".format(round(max_cluster_h2,3),self.max_stacks))
            print("Please reduce H2 production saturation limit by at least {}kg PER ELECTROLYZER CLUSTER".format(round(change_req_h2,3)))
            print("Running electrolyzer simulation without H2 saturation")
        else:
            I_sat=np.where(stack_current_unlim>I_max,I_max,stack_current_unlim)
            V_sat=self.cell_design(self.T_C,I_sat)
            V_tot_lim=V_sat + Vdeg
            stack_power_consumed_kW_sat=I_sat*V_tot_lim*self.N_cells/1000
            system_power_consumed_kW_sat=self.n_stacks_op*stack_power_consumed_kW_sat
            power_curtailed_kW=system_power_consumed_kW_sat-input_power_kW
        return I_sat,power_curtailed_kW
            


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
    def call_degradation_calculations(self,cell_voltage_signal):
        #NOTE: unused as of right now
        deg_df=pd.DataFrame()
        min_possible_life_hrs=self.d_eol/self.onoff_deg_rate
        max_possible_stackrep_during_sim=np.ceil(len(cell_voltage_signal)/min_possible_life_hrs)
        n_stackrep_per_sim=0
        loop_counter=0
        stack_lived_hrs=[]
        #init_voltage_df=self.output_dict['Cumulative Degradation Breakdown'].copy(deep=True)
        degraded_voltage_signal,Vdeg_signal=self.full_degradation(cell_voltage_signal)
        deg_df=pd.concat([deg_df,self.output_dict['Cumulative Degradation Breakdown']])
        stack_died,next_stack_will_die,hour_of_death,V_tot,Vdeg=self.check_aliveness(Vdeg_signal,cell_voltage_signal)
        if stack_died:
            # n_stackrep_per_sim +=1
            # stack_lived_hrs.append(hour_of_death)
            if next_stack_will_die:

                while next_stack_will_die:
                    stack_died,next_stack_will_die,hour_of_death,V_tot,Vdeg=self.check_aliveness(Vdeg,cell_voltage_signal)
                    deg_df=pd.concat([deg_df,self.output_dict['Cumulative Degradation Breakdown']])
                    stack_lived_hrs.append(hour_of_death)
                    n_stackrep_per_sim +=1
                    loop_counter+=1
                    if loop_counter > max_possible_stackrep_during_sim:
                        print("something is afoot at the call_degradation_calculations function")
                        break
            else:
                deg_df=pd.concat([deg_df,self.output_dict['Cumulative Degradation Breakdown']])
                n_stackrep_per_sim +=1
                stack_lived_hrs.append(hour_of_death)

            V_tot_final = V_tot
            Vdeg_final = Vdeg

        else:
            Vdeg_final=Vdeg_signal
            V_tot_final=degraded_voltage_signal
        
        if n_stackrep_per_sim>0:
            stack_replacement_schedule_yrs=self.make_stack_replacement_schedule(stack_lived_hrs,Vdeg_final)
        else:
            stack_replacement_schedule_yrs=np.zeros(self.plant_life_years)
            refturb_period=np.floor(hour_of_death/8760)
            stack_replacement_schedule_yrs[refturb_period:self.plant_life_years:refturb_period]=1
            
            
        self.cumulative_Vdeg_per_hr_sys=Vdeg_final
        self.output_dict['Degradation Breakdown - NEW']=deg_df
        self.stack_repair_schedule=stack_replacement_schedule_yrs
        return V_tot_final,Vdeg_final

    def make_stack_replacement_schedule(self,stack_lived_hrs):
        #NOTE: unused as of now
        #NOTE: this has not been checked for correctness
        #This is probably overcomplicated also
        
        plant_life_hrs=self.plant_life_years*8760
        sim_length_hrs=len(self.cluster_status)
        num_sims_4_life=np.ceil(plant_life_hrs/sim_length_hrs)
        replacement_schedule_yrs_temp=np.zeros(len(self.plant_life_years))
        life_start=0
        life_start_operation=0
        operational_hours=[]
        
        stack_life_length=[stack_lived_hrs[0]] + list(np.diff(stack_lived_hrs))
        for life in stack_life_length:
            life_start_operation+=np.sum(self.cluster_status[life_start:life])
            operational_hours.append(life_start_operation)
            # life_existing_hrs.append(len(self.cluster_status[life_start:life]))
            # life_operating_hrs.append(np.sum(self.cluster_status[life_start:life]))
            life_start+=life
                
        #years_of_operation_to_replace=np.floor(operational_hours/8760)
        for life_duration_hrs in operational_hours:
            replacement_year=np.floor(life_duration_hrs/8760)
            replacement_schedule_yrs_temp[replacement_year]+=1
        
        replace_sched=list(replacement_schedule_yrs_temp)*num_sims_4_life
        replacement_schedule_yrs=np.array(replace_sched[0:self.plant_life_years])
        return replacement_schedule_yrs
        
    def check_aliveness(self,deg_signal_init,voltage_signal_noDeg):
        #NOTE: unused as of now!
        if deg_signal_init[-1]>self.d_eol:
            idx_dead=np.argwhere(deg_signal_init>self.d_eol)[0][0]
            deg_signal_this_life=deg_signal_init[0:idx_dead] #V_deg
            voltage_signal_this_life=voltage_signal_noDeg[0:idx_dead] #V_cell
            v_tot_this_life=deg_signal_this_life + voltage_signal_this_life #V_cell + V_deg
            voltage_signal_next_life=voltage_signal_noDeg[idx_dead:] #no deg
            v_tot_next_life,deg_next_life=self.full_degradation(voltage_signal_next_life)
            voltage_no_deg=np.concatenate((voltage_signal_this_life,voltage_signal_next_life))
            voltage_plus_deg_signal=np.concatenate((v_tot_this_life,v_tot_next_life))
            Vdeg=np.concatenate((deg_signal_this_life,deg_next_life))
            stack_died=True
            hour_died=np.copy(idx_dead)
            if deg_next_life[-1]>self.d_eol:
                stack_will_die=True
            else:
                stack_will_die=False
            if len(Vdeg) != len(deg_signal_init):
                print("ISSUES ARE HAPPENING IN check_aliveness of PEM_H2_LT_electrolyzer_Clusters")
        else:
            voltage_plus_deg_signal=voltage_signal_noDeg + deg_signal_init
            voltage_no_deg=voltage_signal_noDeg #really for debug purposes
            Vdeg=deg_signal_init
            stack_died=False
            stack_will_die=False
            self.calc_stack_replacement_info(deg_signal_init) #number life stack rep
            hour_died=self.time_between_replacements
        return stack_died,stack_will_die,hour_died,voltage_plus_deg_signal,Vdeg
        




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
    def new_calc_stack_replacement_info(self,deg_signal,V_cell):
        
        d_sim = deg_signal[-1] #[V] dgradation at end of simulation
        #t_eod=(self.d_eol/d_sim)*(t_sim_sec/3600) #time between replacement [hrs]
        # stack_operational_time_sec=np.sum(self.cluster_status * self.dt)
        stack_operational_hrs = np.sum(self.cluster_status*self.dt)/3600
        sim_length_hrs = len(V_cell)*self.dt/3600

        power_in_signal=np.linspace(0.1,1,50)*self.stack_rating_kW
        stack_I = calc_current((power_in_signal,self.T_C),*self.curve_coeff)
        stack_V = self.cell_design(self.T_C,stack_I)
        bin_offset = (stack_V[1]-stack_V[0])/2
        V_bins = stack_V - bin_offset
        V_bins = np.insert(V_bins,len(V_bins),stack_V[-1])
        cnt,bins = np.histogram(V_cell,bins=V_bins)
        
        
        eff_drop_per_bin = ((stack_V + d_sim)/stack_V) - 1
        #pdf weighted by eff loss from degradation at end of sim
        weighted_pdf = eff_drop_per_bin*(cnt/np.sum(cnt)) 
        avg_sim_eff_drop = np.sum(weighted_pdf)
        self.frac_of_life_used = avg_sim_eff_drop / self.eol_eff_drop
        
        #number of "awake" hours until death
        t_eod_operation_based = (self.eol_eff_drop/avg_sim_eff_drop)*(stack_operational_hrs) 
        #number of total (awake + asleep) hours until death
        t_eod_existance_based = (self.eol_eff_drop/avg_sim_eff_drop)*(sim_length_hrs)
        #time of death 
        #stack life either 1) stack is on 2) years in between replacements
        #HFTO says stack life based on hours of operation
        #capacity factor is the same

        self.percent_of_sim_operating = stack_operational_hrs/sim_length_hrs

        self.time_between_replacements=t_eod_existance_based #Time between replacement
        self.operational_time_between_replacements=t_eod_operation_based #Stack life based on hours of operation

        return t_eod_existance_based,t_eod_operation_based
    
    def estimate_lifetime_capacity_factor(self,power_in_kW,V_cell,deg_signal,time_between_replacements):
        # self.new_calc_stack_replacement_info(deg_signal,V_cell)
        stack_operational_time_sec=np.sum(self.cluster_status * self.dt)
        num_sim_until_dead = time_between_replacements/(stack_operational_time_sec/3600)
        full_sims_until_dead = int(np.floor(num_sim_until_dead))
        # partial_sim_until_dead = num_sim_until_dead-full_sims_until_dead  
        #Alternative approach:
        cluster_cycling = [0] + list(np.diff(self.cluster_status)) #no delay at beginning of sim
        cluster_cycling = np.array(cluster_cycling)
        startup_ratio = 1-(600/3600)#TODO: don't have this hard-coded
        h2_multiplier = np.where(cluster_cycling > 0, startup_ratio, 1)
        
        # sim_length = len(power_in_kW)
        #TODO: change it from operational hours life to sim-based life: DONE
        lifetime_power_kW = np.tile(power_in_kW,int(full_sims_until_dead+1))#[0:int(np.ceil(time_between_replacements))]
        I_lifetime_noDeg = calc_current((lifetime_power_kW,self.T_C), *self.curve_coeff)
        V_cell_lifetime = self.cell_design(self.T_C,I_lifetime_noDeg)
        n_stacks_on_life = np.tile(self.n_stacks_op,int(full_sims_until_dead+1))#[0:int(np.ceil(time_between_replacements))]
        h2_lifetime_noDeg_noWarmup = self.h2_production_rate(I_lifetime_noDeg,n_stacks_on_life) #if no start-up
        h2_warmup_multiplier_lifetime = np.tile(h2_multiplier,int(full_sims_until_dead+1))#[0:int(np.ceil(time_between_replacements))]
        
        #steady deg
        lifetime_cluster_status = self.system_design(lifetime_power_kW,self.max_stacks)
        steady_deg_per_hr_lifetime=self.dt*self.steady_deg_rate*V_cell_lifetime*lifetime_cluster_status

        #on-off deg
        change_stack=np.diff(lifetime_cluster_status)
        cycle_cnt = np.where(change_stack < 0, -1*change_stack, 0)
        cycle_cnt = np.array([0] + list(cycle_cnt))
        stack_off_deg_per_hr_lifetime= self.onoff_deg_rate*cycle_cnt

        #fatigue 
        V_fatigue_lifetime = self.approx_fatigue_degradation(V_cell_lifetime)

        Vdeg_lifetime = np.cumsum(steady_deg_per_hr_lifetime) + np.cumsum(stack_off_deg_per_hr_lifetime) + V_fatigue_lifetime
        eff_mult_lifetime = (V_cell_lifetime + Vdeg_lifetime)/V_cell_lifetime #(1 + eff drop)
        
        V_cell_rated = self.output_dict['BOL Efficiency Curve Info']['Cell Voltage'].values[-1]
        rated_eff_mult_lifetime = (V_cell_rated + Vdeg_lifetime)/V_cell_rated
        idx_dead = np.argwhere(rated_eff_mult_lifetime>(1+self.eol_eff_drop))[0][0]
        # idx_dead = np.argwhere(eff_mult_lifetime>(1+self.eol_eff_drop))[0][0]
        I_deg_lifetime = I_lifetime_noDeg/eff_mult_lifetime
        h2_prod_lifetime_deg_noWarmup = self.h2_production_rate(I_deg_lifetime,n_stacks_on_life) 
        lifetime_h2_deg_warmup = h2_warmup_multiplier_lifetime*h2_prod_lifetime_deg_noWarmup

        _,rated_h2_pr_stack_BOL=self.rated_h2_prod()
        # lifetime_rated_h2_nodeg = rated_h2_pr_stack_BOL*len(lifetime_power_kW)*self.max_stacks
        lifetime_rated_h2_nodeg = rated_h2_pr_stack_BOL*idx_dead*self.max_stacks
        # capfac_noDeg_noWarmup = np.sum(h2_lifetime_noDeg_noWarmup)/lifetime_rated_h2_nodeg
        # capfac_noDeg_withWarmup = np.sum(h2_lifetime_noDeg_noWarmup*h2_warmup_multiplier_lifetime)/lifetime_rated_h2_nodeg
        # capfac_deg_noWarmup = np.sum(h2_prod_lifetime_deg_noWarmup)/lifetime_rated_h2_nodeg
        # capfac_deg_withWarmup = np.sum(lifetime_h2_deg_warmup)/lifetime_rated_h2_nodeg
        
        h2_lifetime_noDeg_withWarmup=h2_lifetime_noDeg_noWarmup*h2_warmup_multiplier_lifetime
        losses_desc = ['no losses','warm-up losses','degradation losses','full losses']
        # case_desc = ['Simulation Based','Lifetime Estimate']
        params = ['Lifetime Capacity Factor [-]','Lifetime Hydrogen Produced [kg]','Lifetime Average Annual Hydrogen Produced [kg]','Average Efficiency [kWh/kg]']
        # case_desc = ['Simulation','Lifetime (no losses)','Lifetime (warmup losses)','Lifetime (deg losses)','Lifetime (full losses)']
        
        lifetime_est_vals = [h2_lifetime_noDeg_noWarmup[:idx_dead],h2_lifetime_noDeg_withWarmup[:idx_dead],h2_prod_lifetime_deg_noWarmup[:idx_dead],lifetime_h2_deg_warmup[:idx_dead]]

        cf = lambda lifetime_h2,life_h2_capacity : np.sum(lifetime_h2)/life_h2_capacity
        total_h2 = lambda lifetime_h2: np.sum(lifetime_h2)
        avg_annual_h2 = lambda lifetime_h2: 8760*np.sum(lifetime_h2)/len(lifetime_h2)
        # avg_eff_kWh_pr_kg = lambda lifetime_h2,lifetime_power_kW: np.sum(lifetime_power_kW)/np.sum(lifetime_h2)
        avg_eff_kWh_pr_kg = lambda lifetime_h2,lifetime_power_kW,n_stacks_on_life: np.sum(lifetime_power_kW*n_stacks_on_life)/np.sum(lifetime_h2)
        
        cf_vals = [cf(lh2,lifetime_rated_h2_nodeg) for lh2 in lifetime_est_vals]
        lifetime_h2_vals = [total_h2(lh2) for lh2 in lifetime_est_vals]
        avg_yearly_h2_vals = [avg_annual_h2(lh2) for lh2 in lifetime_est_vals]
        # avg_eff_vals =[avg_eff_kWh_pr_kg(lh2,lifetime_power_kW) for lh2 in lifetime_est_vals]
        avg_eff_vals =[avg_eff_kWh_pr_kg(lh2,lifetime_power_kW[:idx_dead],n_stacks_on_life[:idx_dead]) for lh2 in lifetime_est_vals]

        lifetime_performance_df=pd.DataFrame(dict(zip(params,[cf_vals,lifetime_h2_vals,avg_yearly_h2_vals,avg_eff_vals])),index = losses_desc)
        
        #see when eff_mult from sim gives the change required to go from end of last full life year to 
        # end
        # eff_drop_sim = 1-eff_mult[-1]
        # end_of_sims_eff_drop = np.cumsum(eff_drop_sim*np.ones(full_sims_until_dead))
        # num_sim_until_dead = self.eol_eff_drop/eff_drop_sim
        return lifetime_performance_df

        
    def reset_uptime_degradation_rate(self):
        
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
        # import rainflow
        
        
        # dt_fatigue_calc_hrs = 24*7#calculate per week
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
            self.output_dict['Sim End RF Track'] = rf_track
            self.output_dict['Total Actual Fatigue Degradation [V]'] = V_fatigue_ts[-1]

        return V_fatigue_ts #already cumulative!

    def grid_connected_func(self,h2_kg_hr_system_required):
        df=self.output_dict['BOL Efficiency Curve Info'][['H2 Produced','Current','Power Sent [kWh]','Power Consumed [kWh]']]

        max_h2kg_single_stack=self.h2_production_rate(self.max_cell_current,1)
        # EOL_max_h2_stack=self.h2_production_rate(self.max_cell_current,1)
        min_n_stacks=np.ceil(h2_kg_hr_system_required/max_h2kg_single_stack)
        if min_n_stacks>self.max_stacks:
            print("ISSUE")
        h2_per_stack_min=h2_kg_hr_system_required/self.max_stacks #change var name
        
        
        #f_i_of_h2=interpolate.interp1d(df['H2 Produced'].values,df['Current'].values)
        #I_reqd_BOL=f_i_of_h2(h2_per_stack_min)
        #n_f=self.faradaic_efficiency(I_reqd_BOL)
        
        I_reqd_BOL_noFaradaicLoss=(h2_per_stack_min*1000*2*self.F*self.moles_per_g_h2)/(1*self.N_cells*self.dt)
        n_f=self.faradaic_efficiency(I_reqd_BOL_noFaradaicLoss)
        I_reqd=(h2_per_stack_min*1000*2*self.F*self.moles_per_g_h2)/(n_f*self.N_cells*self.dt)
        V_reqd = self.cell_design(self.T_C,I_reqd)

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
        # e_h2=39.41 #kWh/kg - HHV
        # system_power_in_kw=P_sys #self.input_dict['P_input_external_kW'] #all stack input power
        system_h2_prod_rate=self.h2_production_rate(I,self.n_stacks_op)
        eff_kWh_pr_kg = P_sys/system_h2_prod_rate
        system_eff= self.eta_h2_hhv/eff_kWh_pr_kg
        # system_eff=(e_h2 * system_h2_prod_rate)/system_power_in_kw
        return system_eff #[%]

    def make_BOL_efficiency_curve(self):
        power_in_signal=np.arange(0.1,1.1,0.1)*self.stack_rating_kW
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
        For now, system design is solely a function of max. external power
        supply; i.e., a rated power supply of 50 MW means that the electrolyzer
        system developed by this model is also rated at 50 MW

        TODO: Extend model to include this capability.
        Assume that a PEM electrolyzer behaves as a purely resistive load
        in a circuit, and design the configuration of the entire electrolyzer
        system - which may consist of multiple stacks connected together in
        series, parallel, or a combination of both.
        """
        # cluster_min_power = 0.1*self.max_stacks
        cluster_min_power = 0.1*cluster_size_mw
        cluster_status=np.where(input_power_kw<cluster_min_power,0,1)
        # stack_min_power_kw=0.1*self.stack_rating_kW 
        # num_stacks_on_per_hr=np.floor(input_power_kw/cluster_min_power)
        # num_stacks_on=[1 if st>cluster_min_power else st for st in num_stacks_on_per_hr]
        #self.n_stacks_op=num_stacks_on
        # n_stacks = (self.electrolyzer_system_size_MW * 1000) / \
        #                            self.stack_rating_kW
        # self.output_dict['electrolyzer_system_size_MW'] = self.electrolyzer_system_size_MW
        return cluster_status#np.array(cluster_stat)

    def cell_design(self, Stack_T, Stack_Current):
        # self.cell_active_area
        E_cell = self.calc_reversible_cell_voltage(Stack_T)
        V_act = self.calc_V_act(Stack_T,Stack_Current,self.cell_active_area)
        V_ohmic=self.calc_V_ohmic(Stack_T,Stack_Current,self.cell_active_area,self.membrane_thickness)
        # self.output_dict['Voltage Breakdown']=
        V_cell = E_cell + V_act + V_ohmic
       
        return V_cell
    def calc_reversible_cell_voltage(self,Stack_T):
        T_K=Stack_T+ 273.15  # in Kelvins
        E_rev0 = 1.229  # (in Volts) Reversible potential at 25degC - Nerst Equation (see Note below)
        # NOTE: E_rev is unused right now, E_rev0 is the general nerst equation for operating at 25 deg C at atmospheric pressure
        # (whereas we will be operating at higher temps). From the literature above, it appears that E_rev0 is more correct
        # https://www.sciencedirect.com/science/article/pii/S0360319911021380 
        panode_atm=1
        pcathode_atm=1
        patmo_atm=1
        E_rev = 1.5184 - (1.5421 * (10 ** (-3)) * T_K) + \
                 (9.523 * (10 ** (-5)) * T_K * np.log(T_K)) + \
                 (9.84 * (10 ** (-8)) * (T_K ** 2))
        
        A = 8.07131
        B = 1730.63
        C = 233.426

        p_h2o_sat_mmHg = 10 ** (A - (B / (C + Stack_T)))  #vapor pressure of water in [mmHg] using Antoine formula
        p_h20_sat_atm=p_h2o_sat_mmHg*self.mmHg_2_atm #convert mmHg to atm

        # p_h2O_sat_Pa = (0.61121* np.exp((18.678 - (Stack_T / 234.5)) * (Stack_T / (257.14 + Stack_T)))) * 1e3  # (Pa) #ARDEN-BUCK
        # p_h20_sat_atm=p_h2O_sat_Pa/self.patmo
                # Cell reversible voltage kind of explain in Equations (12)-(15) of below source
        # https://www.sciencedirect.com/science/article/pii/S0360319906000693
        # OR see equation (8) in the source below
        # https://www.sciencedirect.com/science/article/pii/S0360319917309278?via%3Dihub
        E_cell=E_rev0 + ((self.R*T_K)/(2*self.F))*(np.log(((panode_atm-p_h20_sat_atm)/patmo_atm)*np.sqrt((pcathode_atm-p_h20_sat_atm)/patmo_atm))) 
        return E_cell

    def calc_V_act(self,Stack_T,I_stack,cell_active_area):
        T_K=Stack_T+ 273.15 
        i = I_stack/cell_active_area
        a_a = 2  # Anode charge transfer coefficient
        a_c = 0.5  # Cathode charge transfer coefficient
        i_o_a = 2 * (10 ** (-7)) #anode exchange current density
        i_o_c = 2 * (10 ** (-3)) #cathode exchange current density
        V_anode = (((self.R * T_K) / (a_a * self.F)) * np.arcsinh(i / (2 * i_o_a)))
        V_cathode= (((self.R * T_K) / (a_c * self.F)) * np.arcsinh(i / (2 * i_o_c)))
        V_act = V_anode + V_cathode
        return V_act

    def calc_V_ohmic(self,Stack_T,I_stack,cell_active_area,delta_cm):
        T_K=Stack_T+ 273.15 
        i = I_stack/cell_active_area
        lambda_water_content = ((-2.89556 + (0.016 * T_K)) + 1.625) / 0.1875
        sigma = ((0.005139 * lambda_water_content) - 0.00326) * np.exp(
            1268 * ((1 / 303) - (1 / T_K)))   # membrane proton conductivity [S/cm]
        R_cell = (delta_cm / sigma) #ionic resistance [ohms*cm^2]
        R_elec=3.5*(10 ** (-5)) # [ohms*cm^2] from Table 1 in  https://journals.utm.my/jurnalteknologi/article/view/5213/3557
        V_ohmic=(i *( R_cell + R_elec)) 
        return V_ohmic
    def dynamic_operation(self): #UNUSED
        """
        Model the electrolyzer's realistic response/operation under variable RE

        TODO: add this capability to the model
        """
        # When electrolyzer is already at or near its optimal operation
        # temperature (~80degC)
        
        warm_startup_time_secs = 30
        cold_startup_time_secs = 5 * 60  # 5 minutes

    def water_electrolysis_efficiency(self): #UNUSED
        """
        https://www.sciencedirect.com/science/article/pii/S2589299119300035#b0500

        According to the first law of thermodynamics energy is conserved.
        Thus, the conversion efficiency calculated from the yields of
        converted electrical energy into chemical energy. Typically,
        water electrolysis efficiency is calculated by the higher heating
        value (HHV) of hydrogen. Since the electrolysis process water is
        supplied to the cell in liquid phase efficiency can be calculated by:

        n_T = V_TN / V_cell

        where, V_TN is the thermo-neutral voltage (min. required V to
        electrolyze water)

        Parameters
        ______________

        Returns
        ______________

        """
        # From the source listed in this function ...
        # n_T=V_TN/V_cell NOT what's below which is input voltage -> this should call cell_design()
        n_T = self.V_TN / (self.stack_input_voltage_DC / self.N_cells)
        return n_T

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

    def compression_efficiency(self): #UNUSED AND MAY HAVE ISSUES
        # Should this only be used if we plan on storing H2?
        """
        In industrial contexts, the remaining hydrogen should be stored at
        certain storage pressures that vary depending on the intended
        application. In the case of subsequent compression, pressure-volume
        work, Wc, must be performed. The additional pressure-volume work can
        be related to the heating value of storable hydrogen. Then, the total
        efficiency reduces by the following factor:
        https://www.mdpi.com/1996-1073/13/3/612/htm

        Due to reasons of material properties and operating costs, large
        amounts of gaseous hydrogen are usually not stored at pressures
        exceeding 100 bar in aboveground vessels and 200 bar in underground
        storages
        https://www.sciencedirect.com/science/article/pii/S0360319919310195

        Partial pressure of H2(g) calculated using:
        The hydrogen partial pressure is calculated as a difference between
        the  cathode  pressure, 101,325 Pa, and the water saturation
        pressure
        [Source: Energies2018,11,3273; doi:10.3390/en11123273]

        """
        n_limC = 0.825  # Limited efficiency of gas compressors (unitless)
        H_LHV = 241  # Lower heating value of H2 (kJ/mol)
        K = 1.4  # Average heat capacity ratio (unitless)
        C_c = 2.75  # Compression factor (ratio of pressure after and before compression)
        n_F = self.faradaic_efficiency()
        j = self.current/self.cell_active_area#self.output_dict['stack_current_density_A_cm2']
        n_x = ((1 - n_F) * j) * self.cell_active_area
        n_h2 = j * self.cell_active_area
        Z = 1  # [Assumption] Average compressibility factor (unitless)
        T_in = 273.15 + self.T_C  # (Kelvins) Assuming electrolyzer operates at 80degC
        W_1_C = (K / (K - 1)) * ((n_h2 - n_x) / self.F) * self.R * T_in * Z * \
                ((C_c ** ((K - 1) / K)) - 1)  # Single stage compression

        # Calculate partial pressure of H2 at the cathode: This is the Antoine formula (see link below)
        #https://www.omnicalculator.com/chemistry/vapour-pressure-of-water#antoine-equation
        A = 8.07131
        B = 1730.63
        C = 233.426
        p_h2o_sat = 10 ** (A - (B / (C + self.T_C)))  # [mmHg]
        p_cat = 101325  # Cathode pressure (Pa)
        # Fixed unit bug between mmHg and Pa
        
        p_h2_cat = p_cat - (p_h2o_sat*self.mmHg_2_Pa) #convert mmHg to Pa
        p_s_h2_Pa = self.p_s_h2_bar * 1e5

        s_C = math.log((p_s_h2_Pa / p_h2_cat), 10) / math.log(C_c, 10)
        W_C = round(s_C) * W_1_C  # Pressure-Volume work - energy reqd. for compression
        net_energy_carrier = n_h2 - n_x  # C/s
        net_energy_carrier = np.where((n_h2 - n_x) == 0, 1, net_energy_carrier)
        n_C = 1 - ((W_C / (((net_energy_carrier) / self.F) * H_LHV * 1000)) * (1 / n_limC))
        n_C = np.where((n_h2 - n_x) == 0, 0, n_C)
        return n_C

    def total_efficiency(self,stack_current):
        """
        Aside from efficiencies accounted for in this model
        (water_electrolysis_efficiency, faradaic_efficiency, and
        compression_efficiency) all process steps such as gas drying above
        2 bar or water pumping can be assumed as negligible. Ultimately, the
        total efficiency or system efficiency of a PEM electrolysis system is:

        n_T = n_p_h2 * n_F_h2 * n_c_h2
        https://www.mdpi.com/1996-1073/13/3/612/htm
        """
        #n_p_h2 = self.water_electrolysis_efficiency() #no longer considered
        n_F_h2 = self.faradaic_efficiency(stack_current)
        #n_c_h2 = self.compression_efficiency() #no longer considered

        #n_T = n_p_h2 * n_F_h2 * n_c_h2 #No longer considers these other efficiencies
        n_T=n_F_h2
        self.output_dict['total_efficiency'] = n_T
        return n_T

    def h2_production_rate(self,stack_current,n_stacks_op):
        """
        H2 production rate calculated using Faraday's Law of Electrolysis
        (https://www.sciencedirect.com/science/article/pii/S0360319917347237#bib27)

        Parameters
        _____________

        float f_1
            Coefficient - value at operating temperature of 80degC (mA2/cm4)

        float f_2
            Coefficient - value at operating temp of 80 degC (unitless)

        np_array
            1-D array of current supplied to electrolyzer stack from external
            power source


        Returns
        _____________

        """
        # Single stack calculations:
        n_Tot = self.total_efficiency(stack_current)
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

    
    def degradation(self):
        """
        TODO
        Add a time component to the model - for degradation ->
        https://www.hydrogen.energy.gov/pdfs/progress17/ii_b_1_peters_2017.pdf
        """
        pass

    def water_supply(self,h2_kg_hr):
        """
        Calculate water supply rate based system efficiency and H2 production
        rate
        TODO: Add this capability to the model
        """
        # ratio of water_used:h2_kg_produced depends on power source
        # h20_kg:h2_kg with PV 22-126:1 or 18-25:1 without PV but considering water deminersalisation
        # stoichometrically its just 9:1 but ... theres inefficiencies in the water purification process
        max_water_feed_mass_flow_rate_kg_hr = 411  # kg per hour
        water_used_kg_hr_system = h2_kg_hr * 10
        self.output_dict['water_used_kg_hr'] = water_used_kg_hr_system
        self.output_dict['water_used_kg_annual'] = np.sum(water_used_kg_hr_system)
        water_used_gal_hr_system = water_used_kg_hr_system/3.79
        return water_used_gal_hr_system 

    def h2_storage(self):
        """
        Model to estimate Ideal Isorthermal H2 compression at 70degC
        https://www.sciencedirect.com/science/article/pii/S036031991733954X

        The amount of hydrogen gas stored under pressure can be estimated
        using the van der Waals equation

        p = [(nRT)/(V-nb)] - [a * ((n^2) / (V^2))]

        where p is pressure of the hydrogen gas (Pa), n the amount of
        substance (mol), T the temperature (K), and V the volume of storage
        (m3). The constants a and b are called the van der Waals coefficients,
        which for hydrogen are 2.45 × 10−2 Pa m6mol−2 and 26.61 × 10−6 ,
        respectively.
        """

        pass
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
        # nsr_life=self.calc_stack_replacement_info(deg_signal)
        lifetime_performance_df =self.make_lifetime_performance_df_all_opt(deg_signal,V_init,power_per_stack)

        # nsr_life=self.new_calc_stack_replacement_info(deg_signal,V_init) #new
        # lifetime_performance_df = self.estimate_lifetime_capacity_factor(power_per_stack,V_init,deg_signal) #new

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
        h2_results['water_hourly_usage_gal'] =h20_gal_used_system
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
        # h2_results_aggregates['Avg [hrs] until Replacement Per Stack'] = self.time_between_replacements
        # h2_results_aggregates['Number of Lifetime Cluster Replacements'] = nsr_life
        # h2_results_aggregates['PEM Capacity Factor'] = pem_cf
        h2_results_aggregates['PEM Capacity Factor (simulation)'] = pem_cf
        
        h2_results_aggregates['Operational Time / Simulation Time (ratio)'] = self.percent_of_sim_operating #added
        h2_results_aggregates['Fraction of Life used during sim'] = self.frac_of_life_used #added

        h2_results_aggregates['Total H2 Production [kg]'] =np.sum(h2_kg_hr_system)
        h2_results_aggregates['Total Input Power [kWh]'] =np.sum(power_input_signal)
        h2_results_aggregates['Total kWh/kg'] =np.sum(power_input_signal)/np.sum(h2_kg_hr_system)
        h2_results_aggregates['Total Uptime [sec]'] = np.sum(self.cluster_status * self.dt)
        h2_results_aggregates['Total Off-Cycles'] = np.sum(self.off_cycle_cnt)
        h2_results_aggregates['Final Degradation [V]'] =self.cumulative_Vdeg_per_hr_sys[-1]
        h2_results_aggregates['IV curve coeff'] = self.curve_coeff
        # h2_results_aggregates['Life'] = lifetime_performance_df
        h2_results_aggregates.update(lifetime_performance_df.to_dict()) 
        # h2_results_aggregates['Stack Life Summary'] = self.stack_life_opt

        h2_results['Stacks on'] = self.n_stacks_op
        h2_results['Power Per Stack [kW]'] = power_per_stack
        h2_results['Stack Current [A]'] = current_signal
        h2_results['V_cell No Deg'] = V_init
        h2_results['V_cell With Deg'] = V_cell_deg
        h2_results['System Degradation [V]']=self.cumulative_Vdeg_per_hr_sys
        
      
        []
        return h2_results, h2_results_aggregates




if __name__=="__main__":
    # Example on how to use this model:
    in_dict = dict()
    in_dict['electrolyzer_system_size_MW'] = 15
    out_dict = dict()

    electricity_profile = pd.read_csv('sample_wind_electricity_profile.csv')
    in_dict['P_input_external_kW'] = electricity_profile.iloc[:, 1].to_numpy()

    el = PEM_electrolyzer_LT(in_dict, out_dict)
    el.h2_production_rate()
    print("Hourly H2 production by stack (kg/hr): ", out_dict['stack_h2_produced_kg_hr'][0:50])
    print("Hourly H2 production by system (kg/hr): ", out_dict['h2_produced_kg_hr_system'][0:50])
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('PEM H2 Electrolysis Results for ' +
                str(out_dict['electrolyzer_system_size_MW']) + ' MW System')

    axs[0, 0].plot(out_dict['stack_h2_produced_kg_hr'])
    axs[0, 0].set_title('Hourly H2 production by stack')
    axs[0, 0].set_ylabel('kg_h2 / hr')
    axs[0, 0].set_xlabel('Hour')

    axs[0, 1].plot(out_dict['h2_produced_kg_hr_system'])
    axs[0, 1].set_title('Hourly H2 production by system')
    axs[0, 1].set_ylabel('kg_h2 / hr')
    axs[0, 1].set_xlabel('Hour')

    axs[1, 0].plot(in_dict['P_input_external_kW'])
    axs[1, 0].set_title('Hourly Energy Supplied by Wind Farm (kWh)')
    axs[1, 0].set_ylabel('kWh')
    axs[1, 0].set_xlabel('Hour')

    total_efficiency = out_dict['total_efficiency']
    system_h2_eff = (1 / total_efficiency) * 33.3
    system_h2_eff = np.where(total_efficiency == 0, 0, system_h2_eff)

    axs[1, 1].plot(system_h2_eff)
    axs[1, 1].set_title('Total Stack Energy Usage per mass net H2')
    axs[1, 1].set_ylabel('kWh_e/kg_h2')
    axs[1, 1].set_xlabel('Hour')

    plt.show()
    print("Annual H2 production (kg): ", np.sum(out_dict['h2_produced_kg_hr_system']))
    print("Annual energy production (kWh): ", np.sum(in_dict['P_input_external_kW']))
    print("H2 generated (kg) per kWH of energy generated by wind farm: ",
          np.sum(out_dict['h2_produced_kg_hr_system']) / np.sum(in_dict['P_input_external_kW']))
