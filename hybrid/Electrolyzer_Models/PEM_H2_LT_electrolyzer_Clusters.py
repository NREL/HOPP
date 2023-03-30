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

    def __init__(self, cluster_size_mw=1000,include_degradation_penalty=True,output_dict={},dt=3600):
        #self.input_dict = input_dict
        # print('RUNNING CLUSTERS PEM')
        self.include_deg_penalty = include_degradation_penalty
        self.input_dict={}
        self.output_dict = output_dict
        self.dt=dt
        # array of input power signal
        # self.input_dict['P_input_external_kW'] = electrical_generation_ts#input_dict['P_input_external_kW']
        #self.electrolyzer_system_size_MW = n_stacks_op#input_dict['electrolyzer_system_size_MW']
        self.max_stacks = cluster_size_mw
        # self.input_dict['voltage_type'] = 'variable'  # not yet implemented
        self.input_dict['voltage_type'] = 'constant'
        self.stack_input_voltage_DC = 250

        # Assumptions:
        self.min_V_cell = 1.62  # Only used in variable voltage scenario
        self.p_s_h2_bar = 31  # H2 outlet pressure
        self.stack_input_current_lower_bound = 400 #[A] any current below this amount (10% rated) will saturate the H2 production to zero, used to be 500 (12.5% of rated)
        self.stack_rating_kW = 1000  # 1 MW
        self.cell_active_area = 1920#1250 #[cm^2]
        self.N_cells = 130
        self.max_cell_current=2*self.cell_active_area #PEM electrolyzers have a max current density of approx 2 A/cm^2 so max current is 2*cell_area

        # Constants:
        self.moles_per_g_h2 = 0.49606 #[1/weight_h2]
        self.V_TN = 1.48  # Thermo-neutral Voltage (Volts) in standard conditions
        self.F = 96485.34  # Faraday's Constant (C/mol) or [As/mol]
        self.R = 8.314  # Ideal Gas Constant (J/mol/K)

        #Additional Constants
        self.T_C = 80 #stack temperature in [C]
        self.mmHg_2_Pa = 133.322 #convert between mmHg to Pa
        self.patmo = 101325 #atmospheric pressure [Pa]
        self.mmHg_2_atm = self.mmHg_2_Pa/self.patmo #convert from mmHg to atm

        
        self.curve_coeff=self.iv_curve() #this initializes the I-V curve to calculate current
        
    def run(self,input_external_power_kw):
        startup_time=600 #[sec]
        startup_ratio = 1-(startup_time/self.dt)
        input_power_kw = self.external_power_supply(input_external_power_kw)
        self.cluster_status = self.system_design(input_power_kw,self.max_stacks)
        cluster_cycling = [self.cluster_status[0]] + list(np.diff(self.cluster_status))
        cluster_cycling = np.array(cluster_cycling)
        # cluster_turnon=
        h2_multiplier = np.where(cluster_cycling > 0, startup_ratio, 1)

        self.n_stacks_op = self.max_stacks*self.cluster_status
        #n_stacks_op is now either number of pem per cluster or 0 if cluster is off!

        #self.external_power_supply(electrical_generation_ts,n_stacks_op) 
        power_per_stack = np.where(self.n_stacks_op>0,input_power_kw/self.n_stacks_op,0)
        stack_current =calc_current((power_per_stack,self.T_C), *self.curve_coeff)
        stack_current =np.where(stack_current >self.stack_input_current_lower_bound,stack_current,0)

        if self.include_deg_penalty:
            self.simple_degradation()
            V_init=self.cell_design(self.T_C,stack_current)
            # single_stack_deg = self.total_Vdeg_per_hr_sys/self.n_stacks_op
            # single_stack_deg = [0 if np.isinf(t) else t for t in single_stack_deg]
            # single_stack_deg = [0 if np.isnan(t) else t for t in single_stack_deg]

            # V_cell= V_init+np.array(single_stack_deg)
            V_cell= V_init+self.cumulative_Vdeg_per_hr_sys
        else:
            V_cell=self.cell_design(self.T_C,stack_current) #+self.total_Vdeg_per_hr_sys
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
        avg_hrs_til_replace=self.simple_degradation()
        maximum_eff_perc,max_eff_kWhperkg = self.max_eff()
        total_eff = 39.41*np.sum(h2_kg_hr_system)/np.sum(input_external_power_kw)
        h2_results={}
        h2_results_aggregates={}
        h2_results['Input Power [kW]'] = input_external_power_kw
        h2_results['hydrogen production no start-up time']=h2_kg_hr_system_init
        h2_results['hydrogen_hourly_production']=h2_kg_hr_system
        h2_results['water_hourly_usage_gal'] =h20_gal_used_system
        h2_results['electrolyzer_total_efficiency_perc'] = efficiency
        h2_results['kwh_per_kgH2'] = input_power_kw / h2_kg_hr_system
        h2_results['Power Consumed [kWh]'] = system_power_consumed
        
        h2_results_aggregates['Stack Rated Power Consumed [kW]'] = p_consumed_max
        h2_results_aggregates['Stack Rated H2 Production [kg/hr]'] = rated_h2_hr
        h2_results_aggregates['Avg [hrs] until Replacement Per Stack'] = avg_hrs_til_replace
        h2_results_aggregates['PEM Capacity Factor'] = pem_cf
        h2_results_aggregates['Efficiency at Min Power [%]'] = maximum_eff_perc
        h2_results_aggregates['Efficiency at Min Power [kWh/kgH2]'] =max_eff_kWhperkg
        h2_results_aggregates['Final Efficiency [%]'] =total_eff
        h2_results_aggregates['Total H2 Production [kg]'] =np.sum(h2_kg_hr_system)
        h2_results_aggregates['Total Input Power [kWh]'] =np.sum(input_external_power_kw)
        h2_results_aggregates['Total kWh/kg'] =np.sum(input_external_power_kw)/np.sum(h2_kg_hr_system)
        h2_results_aggregates['Total Uptime [sec]'] = np.sum(self.cluster_status * self.dt)
        h2_results_aggregates['Total Off-Cycles'] = np.sum(self.off_cycle_cnt)
        h2_results_aggregates['Final Degradation [V]'] =self.cumulative_Vdeg_per_hr_sys[-1]


        h2_results['Stacks on'] = self.n_stacks_op
        h2_results['Power Per Stack [kW]'] = power_per_stack
        h2_results['Stack Current [A]'] = stack_current
        h2_results['V_cell No Deg'] = V_init
        h2_results['V_cell With Deg'] = V_cell
        h2_results['System Degradation [V]']=self.cumulative_Vdeg_per_hr_sys
        
      
        []
        return h2_results, h2_results_aggregates

        
    def simple_degradation(self):
        #import rainflow
        #.rate_fatigue = 3.33330244e-07 #multiple by rf_track
        #rf_cycles = rainflow.count_cycles(voltage_signal, nbins=10)
        # rf_sum = np.sum([pair[0] * pair[1] for pair in rf_cycles])
        t_sim = len(self.n_stacks_op) * self.dt #[sec]
        d_eol=0.7212 #end of life (eol) degradation value [V]
        operating_voltage=2 #V
        change_stack=np.diff(self.cluster_status)
        cycle_cnt = np.where(change_stack < 0, -1*change_stack, 0)
        cycle_cnt = np.array([0] + list(cycle_cnt))
        steady_deg_rate=1.41737929e-10 #[V/s] - from Zack
        onoff_deg_rate=1.47821515e-04 #[V/off-cycle]
        self.off_cycle_cnt = cycle_cnt
        steady_deg_per_hr=self.dt*steady_deg_rate*operating_voltage*self.cluster_status
        self.output_dict['System Steady Degradation [V]'] = steady_deg_per_hr
        self.output_dict['Off-Cycles'] = cycle_cnt
        self.output_dict['Total Off-Cycles'] = np.sum(cycle_cnt)
        self.output_dict['Total Uptime [sec]'] = np.sum(self.cluster_status * self.dt)
        # stack_off_deg= onoff_deg_rate*np.sum(cycle_cnt)
        stack_off_deg_per_hr= onoff_deg_rate*cycle_cnt
        self.output_dict['System Cycle Degradation [V]'] = stack_off_deg_per_hr

        # self.total_Vdeg_per_hr_sys = steady_deg_per_hr + stack_off_deg_per_hr
        total_Vdeg_per_hr_sys = steady_deg_per_hr + stack_off_deg_per_hr
        self.cumulative_Vdeg_per_hr_sys = np.cumsum(total_Vdeg_per_hr_sys)
        d_sim=self.cumulative_Vdeg_per_hr_sys[-1]
        #total_sys_deg=np.sum(steady_deg_per_hr) + stack_off_deg
        # total_system_deg = np.cumsum(self.total_Vdeg_per_hr_sys) 
        #avg_deg_per_stack = total_sys_deg / np.max(self.n_stacks_op)
        # total_avg_deg = (steady_deg_per_hr + stack_off_deg_per_hr)/ self.n_stacks_op
        # total_avg_deg_per_stack = [0 if np.isnan(t) else t for t in total_avg_deg]
        # total_avg_deg_per_stack = [0 if np.isinf(t) else t for t in total_avg_deg_per_stack]
        # avg_deg_per_stack = total_system_deg[-1]/self.max_stacks
        # self.output_dict['Average Vdeg/Stack']=avg_deg_per_stack
        # d_sim = np.cumsum(avg_deg_per_stack)
        # d_sim = np.cumsum(total_avg_deg_per_stack)
        # t_eod = ((d_eol/avg_deg_per_stack)*t_sim)/3600 #avg time [hrs] until death!
        t_eod = (d_eol/d_sim)*t_sim/3600 

        return t_eod #time until death [hrs] for all stacks in a cluster
    def full_degradation(self,voltage_signal,use_fatigue=True):
        V_deg_uptime = self.calc_uptime_degradation()
        V_deg_onoff = self.calc_onoff_degradation()
        V_signal = voltage_signal + np.cumsum(V_deg_uptime) + np.cumsum(V_deg_onoff)
        if use_fatigue:
            V_fatigue=self.approx_fatigue_degradation(V_signal)
        else:
            V_fatigue=np.zeros(len(voltage_signal))
        deg_signal = np.cumsum(V_deg_uptime) + np.cumsum(V_deg_onoff) + V_fatigue
        
        
        
    def calc_stack_replacement_info(self,plant_life_yrs):
        d_eol=0.7212 #end of life (eol) degradation value [V]
        plant_life_hrs=plant_life_yrs*8760
        t_sim = len(self.n_stacks_op) * self.dt 
    def calc_uptime_degradation(self,use_uptime_deg=True):
        steady_deg_rate=1.41737929e-10 #[V/s] 
        operating_voltage=2 #V
        
        steady_deg_per_hr=self.dt*steady_deg_rate*operating_voltage*self.cluster_status
        cumulative_Vdeg=np.cumsum(steady_deg_per_hr)
        self.output_dict['Total Uptime [sec]'] = np.sum(self.cluster_status * self.dt)
        self.output_dict['Total Uptime Degradation [V]'] = cumulative_Vdeg[-1]

        return steady_deg_per_hr
        
    def calc_onoff_degradation(self,use_onoff_deg=True):
        onoff_deg_rate=1.47821515e-04 #[V/off-cycle]
        change_stack=np.diff(self.cluster_status)
        cycle_cnt = np.where(change_stack < 0, -1*change_stack, 0)
        cycle_cnt = np.array([0] + list(cycle_cnt))
        self.off_cycle_cnt = cycle_cnt
        stack_off_deg_per_hr= onoff_deg_rate*cycle_cnt
        self.output_dict['System Cycle Degradation [V]'] = np.cumsum(stack_off_deg_per_hr)[-1]
        self.output_dict['Off-Cycles'] = cycle_cnt
        return stack_off_deg_per_hr

    def approx_fatigue_degradation(self,voltage_signal):
        import rainflow
        rate_fatigue = 3.33330244e-07 #multiply by rf_track
        dt_fatigue_calc_hrs = 24*7#calculate per week
        t_calc=np.arange(0,len(voltage_signal)+dt_fatigue_calc_hrs ,dt_fatigue_calc_hrs ) 
        # n_fatigue_calcs = 48
        # time_between_calcs = np.floor(len(voltage_signal)/n_fatigue_calcs)
        rf_cycles = rainflow.count_cycles(voltage_signal, nbins=10)
        rf_sum = np.sum([pair[0] * pair[1] for pair in rf_cycles])
        lifetime_fatigue_deg=rf_sum*rate_fatigue
        self.output_dict['Approx Total Fatigue Degradation [V]'] = lifetime_fatigue_deg
        rf_track=0
        V_fatigue_track=0
        V_fatigue_ts=np.zeros(len(voltage_signal))
        for i in range(len(t_calc)-1):
            rf_cycles=rainflow.count_cycles(voltage_signal[t_calc[i]:t_calc[i+1]], nbins=10)
            rf_sum=np.sum([pair[0] * pair[1] for pair in rf_cycles])
            rf_track+=rf_sum
            V_fatigue_ts[t_calc[i]:t_calc[i+1]]=rf_track*rate_fatigue
            #already cumulative!
        self.output_dict['Sim End RF Track'] = rf_track
        self.output_dict['Total Actial Fatigue Degradation [V]'] = V_fatigue_ts[-1]

        return V_fatigue_ts #already cumulative!



    # def calc_simulation_fatigue(self,voltage_signal):
    #     import rainflow
    #     rate_fatigue = 3.33330244e-07 #multiply by rf_track
    #     rf_cycles = rainflow.count_cycles(voltage_signal, nbins=10)
    #     rf_sum = np.sum([pair[0] * pair[1] for pair in rf_cycles])
    #     fatigue_deg=rf_sum*rate_fatigue
    # def calc_number_life_cluster_rep(self,plant_life_yrs,V_cell,use_fatigue=True):
    #     d_eol=0.7212 #end of life (eol) degradation value [V]
    #     Vdeg_init=self.cumulative_Vdeg_per_hr_sys[-1]
    #     plant_life_hrs=plant_life_yrs*8760
    #     if use_fatigue:
    #         V_fatigue=self.calc_simulation_fatigue(V_cell)
    #         d_sim=Vdeg_init + V_fatigue
    #     else:
    #         d_sim = Vdeg_init
    #     t_eod = (d_eol/d_sim)*len(V_cell) #time between replacement in hrs
    #     n_cluster_lifetimerep=plant_life_hrs/t_eod
    #     return n_cluster_lifetimerep,t_eod


    def system_efficiency(self,P_sys,I):
        e_h2=39.41 #kWh/kg
        system_power_in_kw=P_sys #self.input_dict['P_input_external_kW'] #all stack input power
        system_h2_prod_rate=self.h2_production_rate(I,self.n_stacks_op)
        system_eff=(e_h2 * system_h2_prod_rate)/system_power_in_kw
        return system_eff
    def rated_h2_prod(self):
        I_max = calc_current((self.stack_rating_kW,self.T_C),*self.curve_coeff)
        V_max = self.cell_design(self.T_C,I_max)
        P_consumed_stack_kw = I_max*V_max*self.N_cells/1000
        max_h2_stack_kg= self.h2_production_rate(I_max,1)
        return P_consumed_stack_kw,max_h2_stack_kg
    def max_eff(self):
        e_h2=39.41 #kWh/kg
        P_min = 0.1*self.stack_rating_kW
        I_min = calc_current((P_min,self.T_C),*self.curve_coeff)
        V_min = self.cell_design(self.T_C,I_min)
        h2_stack_kg= self.h2_production_rate(I_min,1)
        maximum_eff_perc = (e_h2*h2_stack_kg)/P_min
        max_eff_kWhperkg = P_min/h2_stack_kg
        return maximum_eff_perc,max_eff_kWhperkg

        

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
        if self.input_dict['voltage_type'] == 'constant':
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
        """

        Please note that this method is currently not used in the model. It
        will be used once the electrolyzer model is expanded to variable
        voltage supply as well as implementation of the self.system_design()
        method

        Motivation:

        The most common representation of the electrolyzer performance is the
        polarization curve that represents the relation between the current density
        and the voltage (V):
        Source: https://www.sciencedirect.com/science/article/pii/S0959652620312312

        V = N_c(E_cell + V_Act,c + V_Act,a + iR_cell)

        where N_c is the number of electrolyzer cells,E_cell is the open circuit
        voltage VAct,and V_Act,c are the anode and cathode activation over-potentials,
        i is the current density and iRcell is the electrolyzer cell resistance
        (ohmic losses).

        Use this to make a V vs. A (Amperes/cm2) graph which starts at 1.23V because
        thermodynamic reaction of water formation/splitting dictates that standard
        electrode potential has a ∆G of 237 kJ/mol (where: ∆H = ∆G + T∆S)

        10/31/2022
        ESG: https://www.sciencedirect.com/science/article/pii/S0360319906000693
        -> calculates cell voltage to make IV curve (called by iv_curve)
        Another good source for the equations used in this function: 
        https://www.sciencedirect.com/science/article/pii/S0360319918309017

        """

        # Cell level inputs:

        E_rev0 = 1.229  # (in Volts) Reversible potential at 25degC - Nerst Equation (see Note below)
        #E_th = 1.48  # (in Volts) Thermoneutral potential at 25degC - No longer used

        T_K=Stack_T+ 273.15  # in Kelvins
        # E_cell == Open Circuit Voltage - used to be a static variable, now calculated
        # NOTE: E_rev is unused right now, E_rev0 is the general nerst equation for operating at 25 deg C at atmospheric pressure
        # (whereas we will be operating at higher temps). From the literature above, it appears that E_rev0 is more correct
        # https://www.sciencedirect.com/science/article/pii/S0360319911021380 
        E_rev = 1.5184 - (1.5421 * (10 ** (-3)) * T_K) + \
                 (9.523 * (10 ** (-5)) * T_K * math.log(T_K)) + \
                 (9.84 * (10 ** (-8)) * (T_K ** 2))
        
        # Calculate partial pressure of H2 at the cathode: 
        # Uses Antoine formula (see link below)
        # p_h2o_sat calculation taken from compression efficiency calculation
        # https://www.omnicalculator.com/chemistry/vapour-pressure-of-water#antoine-equation
        A = 8.07131
        B = 1730.63
        C = 233.426
        
        p_h2o_sat_mmHg = 10 ** (A - (B / (C + Stack_T)))  #vapor pressure of water in [mmHg] using Antoine formula
        p_h20_sat_atm=p_h2o_sat_mmHg*self.mmHg_2_atm #convert mmHg to atm

        # could also use Arden-Buck equation (see below). Arden Buck and Antoine equations give barely different pressures 
        # for the temperatures we're looking, however, the differences between the two become more substantial at higher temps
    
        # p_h20_sat_pa=((0.61121*math.exp((18.678-(Stack_T/234.5))*(Stack_T/(257.14+Stack_T))))*1e+3) #ARDEN BUCK
        # p_h20_sat_atm=p_h20_sat_pa/self.patmo

        # Cell reversible voltage kind of explain in Equations (12)-(15) of below source
        # https://www.sciencedirect.com/science/article/pii/S0360319906000693
        # OR see equation (8) in the source below
        # https://www.sciencedirect.com/science/article/pii/S0360319917309278?via%3Dihub
        E_cell=E_rev0 + ((self.R*T_K)/(2*self.F))*(np.log((1-p_h20_sat_atm)*math.sqrt(1-p_h20_sat_atm))) #1 value is atmoshperic pressure in atm
        i = Stack_Current/self.cell_active_area #i is cell current density

        # Following coefficient values obtained from Yigit and Selamet (2016) -
        # https://www.sciencedirect.com/science/article/pii/S0360319916318341?via%3Dihub
        a_a = 2  # Anode charge transfer coefficient
        a_c = 0.5  # Cathode charge transfer coefficient
        i_o_a = 2 * (10 ** (-7)) #anode exchange current density
        i_o_c = 2 * (10 ** (-3)) #cathode exchange current density

        #below is the activation energy for anode and cathode - see  https://www.sciencedirect.com/science/article/pii/S0360319911021380 
        V_act = (((self.R * T_K) / (a_a * self.F)) * np.arcsinh(i / (2 * i_o_a))) + (
                ((self.R * T_K) / (a_c * self.F)) * np.arcsinh(i / (2 * i_o_c)))
        
        # equation 13 and 12 for lambda_water_content and sigma: from https://www.sciencedirect.com/science/article/pii/S0360319917309278?via%3Dihub         
        lambda_water_content = ((-2.89556 + (0.016 * T_K)) + 1.625) / 0.1875
        delta = 0.018 # [cm] reasonable membrane thickness of 180-µm NOTE: this will likely decrease in the future 
        sigma = ((0.005139 * lambda_water_content) - 0.00326) * math.exp(
            1268 * ((1 / 303) - (1 / T_K)))   # membrane proton conductivity [S/cm]
        
        R_cell = (delta / sigma) #ionic resistance [ohms]
        R_elec=3.5*(10 ** (-5)) # [ohms] from Table 1 in  https://journals.utm.my/jurnalteknologi/article/view/5213/3557
        V_cell = E_cell + V_act + (i *( R_cell + R_elec)) #cell voltage [V]
        # NOTE: R_elec is to account for the electronic resistance measured between stack terminals in open-circuit conditions
        # Supposedly, removing it shouldn't lead to large errors 
        # calculation for it: http://www.electrochemsci.org/papers/vol7/7043314.pdf

        #V_stack = self.N_cells * V_cell  # Stack operational voltage -> this is combined in iv_calc for power rather than here

        return V_cell

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
        #Fixed unit bug between mmHg and Pa
        
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
