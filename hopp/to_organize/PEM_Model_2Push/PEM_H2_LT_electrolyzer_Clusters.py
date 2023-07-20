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
            self.d_eol=self.find_eol_voltage_val(eol_eff_percent_loss)
        else:
            self.d_eol = 0.7212

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
            nsr_life=self.calc_stack_replacement_info(deg_signal)
            #below is to find equivalent current (NEW)
            stack_current=self.find_equivalent_input_power_4_deg(power_per_stack,V_init,deg_signal)
            V_cell_equiv = self.cell_design(self.T_C,stack_current)
            V_cell = V_cell_equiv + deg_signal
        else:
            V_init=self.cell_design(self.T_C,stack_current)
            V_ignore,deg_signal=self.full_degradation(V_init)
            V_cell=self.cell_design(self.T_C,stack_current) #+self.total_Vdeg_per_hr_sys
            nsr_life=self.calc_stack_replacement_info(deg_signal)
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
        
        h2_results_aggregates['Stack Rated Power Consumed [kWh]'] = p_consumed_max
        h2_results_aggregates['Stack Rated H2 Production [kg/hr]'] = rated_h2_hr
        h2_results_aggregates['Cluster Rated H2 Production [kg/yr]'] = rated_h2_hr*len(input_power_kw)*self.max_stacks
        h2_results_aggregates['Avg [hrs] until Replacement Per Stack'] = self.time_between_replacements
        h2_results_aggregates['Number of Lifetime Cluster Replacements'] = nsr_life
        h2_results_aggregates['PEM Capacity Factor'] = pem_cf
        
        h2_results_aggregates['Total H2 Production [kg]'] =np.sum(h2_kg_hr_system)
        h2_results_aggregates['Total Input Power [kWh]'] =np.sum(input_external_power_kw)
        h2_results_aggregates['Total kWh/kg'] =np.sum(input_external_power_kw)/np.sum(h2_kg_hr_system)
        h2_results_aggregates['Total Uptime [sec]'] = np.sum(self.cluster_status * self.dt)
        h2_results_aggregates['Total Off-Cycles'] = np.sum(self.off_cycle_cnt)
        h2_results_aggregates['Final Degradation [V]'] =self.cumulative_Vdeg_per_hr_sys[-1]
        h2_results_aggregates['IV curve coeff'] = self.curve_coeff

        h2_results['Stacks on'] = self.n_stacks_op
        h2_results['Power Per Stack [kW]'] = power_per_stack
        h2_results['Stack Current [A]'] = stack_current
        h2_results['V_cell No Deg'] = V_init
        h2_results['V_cell With Deg'] = V_cell
        h2_results['System Degradation [V]']=self.cumulative_Vdeg_per_hr_sys
        
      
        []
        return h2_results, h2_results_aggregates

        
    def find_equivalent_input_power_4_deg(self,power_in_kW,V_init,V_deg):
        '''this function corrects the current for degradation
        when the electrolyzer is degraded, it (in the past) would consume more power
        than input. Now, it finds the equivalent current so that power consumed when degraded
        is about equal to power input. Without this function, h2 production is the same 
        at BOL and EOL.'''
        E_cell=self.calc_reversible_cell_voltage(self.T_C)
        I_in = calc_current((power_in_kW,self.T_C), *self.curve_coeff)
        
        P_consumed_kW = I_in*(V_init + V_deg)*self.N_cells/1000 #power actuall consumed
    
        P_consumed_kW =np.where(P_consumed_kW>=power_in_kW,P_consumed_kW,power_in_kW) #added 3/16
        P_consumed_kW=P_consumed_kW*self.cluster_status
        
        #not the best way to do it, but it works for now
        power_diff_error_kW = P_consumed_kW - power_in_kW
        P_equiv = power_in_kW - power_diff_error_kW
        I_equiv = calc_current((P_equiv,self.T_C), *self.curve_coeff)
        
        I_equiv =np.where(I_equiv >0,I_equiv,0)

        clust_stat_new=I_equiv/I_equiv #unused - primarily a debug variable
        clust_stat_new=np.nan_to_num(clust_stat_new)
        
        V_act_equiv =self.calc_V_act(self.T_C,I_equiv,self.cell_active_area)
        V_ohm_equiv =self.calc_V_ohmic(self.T_C,I_equiv,self.cell_active_area,self.membrane_thickness)
        V_cell_equiv = E_cell + V_act_equiv + V_ohm_equiv + V_deg
        P_equiv_cons = V_cell_equiv*I_equiv*self.N_cells/1000 #debug variable
        
        data=[I_equiv,P_equiv,V_cell_equiv,clust_stat_new,P_equiv_cons,P_consumed_kW]
        keys=['I_equiv','P_equiv','V_cell_equiv','ClusterStatus_equiv','P_equiv_cons','P_consumed_kW_init']
        self.output_dict['Equivalent Current Calculation']=dict(zip(keys,data))
        
        return I_equiv
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
            # n_f=self.faradaic_efficiency(I_max)
            # I_max_check=(self.dt/1000)*kg_h2_per_stack*2*self.F*self.moles_per_g_h2/(self.N_cells*n_f)
            # V_max = self.cell_design(self.T_C,I_max)
            # P_bol_max_kW= I_max*V_max*self.N_cells/1000 #consumes
            # I_from_IV=calc_current((P_bol_max_kW,self.T_C),*self.curve_coeff) #could be used to double check

            # error_h2_from_Imax=self.h2_production_rate(I_max,1)-kg_h2_per_stack
            # error_h2_from_check=self.h2_production_rate(I_max_check,1)-kg_h2_per_stack
            # error_h2_from_IV=self.h2_production_rate(I_from_IV,1)-kg_h2_per_stack
            # h2_errors=[error_h2_from_Imax,error_h2_from_check,error_h2_from_IV]
            # I_max_vals=[I_max,I_max_check,I_from_IV]
            # idx_min_error=np.argmin(h2_errors)
            # I_max=I_max_vals[idx_min_error]
            


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
        #d_eol=0.7212 #end of life (eol) degradation value [V]
        #t_sim_sec = len(deg_signal) * self.dt 
        d_sim = deg_signal[-1] #[V] dgradation at end of simulation
        #t_eod=(self.d_eol/d_sim)*(t_sim_sec/3600) #time between replacement [hrs]
        stack_operational_time_sec=np.sum(self.cluster_status * self.dt)
        #[below] revised on 03/27 to be based on operational hours
        #rather than simulation length
        t_eod = (self.d_eol/d_sim)*(stack_operational_time_sec/3600) 
         #time until death [hrs] for all stacks in a cluster
        self.time_between_replacements=t_eod

        plant_life_hrs=self.plant_life_years*8760
        #TODO - remove below, is unnecessary
        num_clusterrep=plant_life_hrs/t_eod #number of lifetime cluster replacements
        return num_clusterrep
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

    def approx_fatigue_degradation(self,voltage_signal):
        #should not use voltage values when voltage_signal = 0
        #aka - should only be counted when electrolyzer is on
        # import rainflow
        
        
        dt_fatigue_calc_hrs = 24*7#calculate per week
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
        e_h2=39.41 #kWh/kg - HHV
        system_power_in_kw=P_sys #self.input_dict['P_input_external_kW'] #all stack input power
        system_h2_prod_rate=self.h2_production_rate(I,self.n_stacks_op)
        system_eff=(e_h2 * system_h2_prod_rate)/system_power_in_kw
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
        R_cell = (delta_cm / sigma) #ionic resistance [ohms]
        R_elec=3.5*(10 ** (-5)) # [ohms] from Table 1 in  https://journals.utm.my/jurnalteknologi/article/view/5213/3557
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
        nsr_life=self.calc_stack_replacement_info(deg_signal)

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
        
        h2_results_aggregates['Stack Rated Power Consumed [kWh]'] = p_consumed_max
        h2_results_aggregates['Stack Rated H2 Production [kg/hr]'] = rated_h2_hr
        h2_results_aggregates['Cluster Rated H2 Production [kg/yr]'] = rated_h2_hr*len(power_input_signal)*self.max_stacks
        h2_results_aggregates['Avg [hrs] until Replacement Per Stack'] = self.time_between_replacements
        h2_results_aggregates['Number of Lifetime Cluster Replacements'] = nsr_life
        h2_results_aggregates['PEM Capacity Factor'] = pem_cf
        
        h2_results_aggregates['Total H2 Production [kg]'] =np.sum(h2_kg_hr_system)
        h2_results_aggregates['Total Input Power [kWh]'] =np.sum(power_input_signal)
        h2_results_aggregates['Total kWh/kg'] =np.sum(power_input_signal)/np.sum(h2_kg_hr_system)
        h2_results_aggregates['Total Uptime [sec]'] = np.sum(self.cluster_status * self.dt)
        h2_results_aggregates['Total Off-Cycles'] = np.sum(self.off_cycle_cnt)
        h2_results_aggregates['Final Degradation [V]'] =self.cumulative_Vdeg_per_hr_sys[-1]
        h2_results_aggregates['IV curve coeff'] = self.curve_coeff

        h2_results['Stacks on'] = self.n_stacks_op
        h2_results['Power Per Stack [kW]'] = power_per_stack
        h2_results['Stack Current [A]'] = current_signal
        h2_results['V_cell No Deg'] = V_init
        h2_results['V_cell With Deg'] = V_cell_deg
        h2_results['System Degradation [V]']=self.cumulative_Vdeg_per_hr_sys
        
      
        []
        return h2_results, h2_results_aggregates



    # def cell_design(self, Stack_T, Stack_Current):
    #     """

    #     Please note that this method is currently not used in the model. It
    #     will be used once the electrolyzer model is expanded to variable
    #     voltage supply as well as implementation of the self.system_design()
    #     method

    #     Motivation:

    #     The most common representation of the electrolyzer performance is the
    #     polarization curve that represents the relation between the current density
    #     and the voltage (V):
    #     Source: https://www.sciencedirect.com/science/article/pii/S0959652620312312

    #     V = N_c(E_cell + V_Act,c + V_Act,a + iR_cell)

    #     where N_c is the number of electrolyzer cells,E_cell is the open circuit
    #     voltage VAct,and V_Act,c are the anode and cathode activation over-potentials,
    #     i is the current density and iRcell is the electrolyzer cell resistance
    #     (ohmic losses).

    #     Use this to make a V vs. A (Amperes/cm2) graph which starts at 1.23V because
    #     thermodynamic reaction of water formation/splitting dictates that standard
    #     electrode potential has a ∆G of 237 kJ/mol (where: ∆H = ∆G + T∆S)

    #     10/31/2022
    #     ESG: https://www.sciencedirect.com/science/article/pii/S0360319906000693
    #     -> calculates cell voltage to make IV curve (called by iv_curve)
    #     Another good source for the equations used in this function: 
    #     https://www.sciencedirect.com/science/article/pii/S0360319918309017

    #     """

    #     # Cell level inputs:

    #     E_rev0 = 1.229  # (in Volts) Reversible potential at 25degC - Nerst Equation (see Note below)
    #     #E_th = 1.48  # (in Volts) Thermoneutral potential at 25degC - No longer used

    #     T_K=Stack_T+ 273.15  # in Kelvins
    #     # E_cell == Open Circuit Voltage - used to be a static variable, now calculated
    #     # NOTE: E_rev is unused right now, E_rev0 is the general nerst equation for operating at 25 deg C at atmospheric pressure
    #     # (whereas we will be operating at higher temps). From the literature above, it appears that E_rev0 is more correct
    #     # https://www.sciencedirect.com/science/article/pii/S0360319911021380 
    #     E_rev = 1.5184 - (1.5421 * (10 ** (-3)) * T_K) + \
    #              (9.523 * (10 ** (-5)) * T_K * math.log(T_K)) + \
    #              (9.84 * (10 ** (-8)) * (T_K ** 2))
        
    #     # Calculate partial pressure of H2 at the cathode: 
    #     # Uses Antoine formula (see link below)
    #     # p_h2o_sat calculation taken from compression efficiency calculation
    #     # https://www.omnicalculator.com/chemistry/vapour-pressure-of-water#antoine-equation
    #     A = 8.07131
    #     B = 1730.63
    #     C = 233.426
        
    #     p_h2o_sat_mmHg = 10 ** (A - (B / (C + Stack_T)))  #vapor pressure of water in [mmHg] using Antoine formula
    #     p_h20_sat_atm=p_h2o_sat_mmHg*self.mmHg_2_atm #convert mmHg to atm

    #     # could also use Arden-Buck equation (see below). Arden Buck and Antoine equations give barely different pressures 
    #     # for the temperatures we're looking, however, the differences between the two become more substantial at higher temps
    
    #     # p_h20_sat_pa=((0.61121*math.exp((18.678-(Stack_T/234.5))*(Stack_T/(257.14+Stack_T))))*1e+3) #ARDEN BUCK
    #     # p_h20_sat_atm=p_h20_sat_pa/self.patmo

    #     # Cell reversible voltage kind of explain in Equations (12)-(15) of below source
    #     # https://www.sciencedirect.com/science/article/pii/S0360319906000693
    #     # OR see equation (8) in the source below
    #     # https://www.sciencedirect.com/science/article/pii/S0360319917309278?via%3Dihub
    #     E_cell=E_rev0 + ((self.R*T_K)/(2*self.F))*(np.log((1-p_h20_sat_atm)*math.sqrt(1-p_h20_sat_atm))) #1 value is atmoshperic pressure in atm
    #     i = Stack_Current/self.cell_active_area #i is cell current density

    #     # Following coefficient values obtained from Yigit and Selamet (2016) -
    #     # https://www.sciencedirect.com/science/article/pii/S0360319916318341?via%3Dihub
    #     a_a = 2  # Anode charge transfer coefficient
    #     a_c = 0.5  # Cathode charge transfer coefficient
    #     i_o_a = 2 * (10 ** (-7)) #anode exchange current density
    #     i_o_c = 2 * (10 ** (-3)) #cathode exchange current density

    #     #below is the activation energy for anode and cathode - see  https://www.sciencedirect.com/science/article/pii/S0360319911021380 
    #     V_act = (((self.R * T_K) / (a_a * self.F)) * np.arcsinh(i / (2 * i_o_a))) + (
    #             ((self.R * T_K) / (a_c * self.F)) * np.arcsinh(i / (2 * i_o_c)))
        
    #     # equation 13 and 12 for lambda_water_content and sigma: from https://www.sciencedirect.com/science/article/pii/S0360319917309278?via%3Dihub         
    #     lambda_water_content = ((-2.89556 + (0.016 * T_K)) + 1.625) / 0.1875
    #     delta = 0.018 # [cm] reasonable membrane thickness of 180-µm NOTE: this will likely decrease in the future 
    #     sigma = ((0.005139 * lambda_water_content) - 0.00326) * math.exp(
    #         1268 * ((1 / 303) - (1 / T_K)))   # membrane proton conductivity [S/cm]
        
    #     R_cell = (delta / sigma) #ionic resistance [ohms]
    #     R_elec=3.5*(10 ** (-5)) # [ohms] from Table 1 in  https://journals.utm.my/jurnalteknologi/article/view/5213/3557
    #     V_cell = E_cell + V_act + (i *( R_cell + R_elec)) #cell voltage [V]
    #     # NOTE: R_elec is to account for the electronic resistance measured between stack terminals in open-circuit conditions
    #     # Supposedly, removing it shouldn't lead to large errors 
    #     # calculation for it: http://www.electrochemsci.org/papers/vol7/7043314.pdf

    #     #V_stack = self.N_cells * V_cell  # Stack operational voltage -> this is combined in iv_calc for power rather than here

    #     return V_cell
    # def max_eff(self):
    #     e_h2=39.41 #kWh/kg
    #     P_min = 0.1*self.stack_rating_kW
    #     I_min = calc_current((P_min,self.T_C),*self.curve_coeff)
    #     V_min = self.cell_design(self.T_C,I_min)
    #     h2_stack_kg= self.h2_production_rate(I_min,1)
    #     maximum_eff_perc = (e_h2*h2_stack_kg)/P_min
    #     max_eff_kWhperkg = P_min/h2_stack_kg
    #     return maximum_eff_perc,max_eff_kWhperkg

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
