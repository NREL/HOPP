import os
import sys
sys.path.append('')
# from dotenv import load_dotenv
import pandas as pd
# from PEM_H2_LT_electrolyzer_ESGBasicClusters import PEM_electrolyzer_LT as PEMClusters
from hopp.simulation.technologies.hydrogen.electrolysis.PEM_H2_LT_electrolyzer_Clusters import PEM_H2_Clusters as PEMClusters
from hopp.add_custom_modules.custom_wind_floris import Floris
from hopp.to_organize.H2_Analysis.simple_dispatch import SimpleDispatch
# from PEM_H2_LT_electrolyzer_Clusters import PEM_H2_Clusters as PEMClusters
import numpy as np
from numpy import savetxt #ESG
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import warnings
import math
import scipy
import time
from scipy import interpolate
warnings.filterwarnings("ignore")

"""
Perform a LCOH analysis for an offshore wind + Hydrogen PEM system

1. Offshore wind site locations and cost details (4 sites, $1300/kw capex + BOS cost which will come from Orbit Runs)~
2. Cost Scaling Based on Year (Have Weiser et. al report with cost scaling for fixed and floating tech, will implement)
3. Cost Scaling Based on Plant Size (Shields et. Al report)
4. Future Model Development Required:
- Floating Electrolyzer Platform
"""
# 
#---------------------------
# 
class turbine_power_electronics:
    '''Add description and stuff :)'''
    # def __init__(self,electrical_power_signal,nturbs_per_cable,nturbs_dist2_load,n_distances,wind_plant,n_turbs):
    def __init__(self,n_turbs,verbose=True):

        # self.cluster_cap_mw = np.round(system_size_mw/num_clusters)
        self.power_data=pd.Series()
        self.loss_data=pd.Series()
        self.ts_data = pd.DataFrame()
        # power_keys = ['Initial Power [MW] (unlimited)','Initial Power [MW] (saturated to rated)','Inital Power [MW] with Default Losses','Power after Transformer [MW]','Power After Distribution [MW]','Power After Rectifier [MW]']
        self.component_list={}
        self.n_turbs = n_turbs

        # self.num_clusters = num_clusters
        # self.tot_system_size_mw = system_size_mw

        # self.stack_rating_kw = 1000
        # self.cluster_cap_mw = system_size_mw//num_clusters
        # # self.cluster_cap_mw = num_clusters*self.stack_rating_kw/(1e3)
        # self.stack_min_power_kw=0.1*self.stack_rating_kw
        # #self.num_available_pem=interconnection_size_mw
        # self.input_power_kw=electrical_power_signal
        # self.cluster_min_power = self.stack_min_power_kw * self.cluster_cap_mw
        # self.cluster_max_power = self.stack_rating_kw * self.cluster_cap_mw
        pysam_default_keys =['avail_bop','avail_grid','avail_turb','ops_env','ops_grid','ops_load','elec_eff','elec_parasitic','env_deg','env_exp','icing']
        pysam_default_vals = [0.5,1.5,3.58,1,0.84,0.99,1.91,0.1,1.8,0.4,0.21]
        prelim_losses = dict(zip(pysam_default_keys,pysam_default_vals))
        self.prelim_losses = pd.Series(prelim_losses)

        # if floris:
        #     self.run_power_elec_floris(nturbs_per_cable,nturbs_dist2_load,n_distances,wind_plant)
        # else:
        #     self.run_power_elec_pysam(electrical_power_signal,nturbs_per_cable,nturbs_dist2_load,n_distances,wind_plant)
        
        
    def run_distributed_layout_power_floris(self,wind_plant):
        #need floris configuration!
        x_load_percent = np.linspace(0.0,1.0,11)
        
        #ac2ac_transformer_eff=np.array([90.63, 93.91, 95.63, 96.56, 97.19, 97.50, 97.66, 97.66, 97.66, 97.50])
        ac2dc_rectification_eff=np.array([96,96.54, 98.12, 98.24, 98.6, 98.33, 98.03, 97.91, 97.43, 97.04, 96.687])/100
        dc2dc_rectification_eff=np.array([91,91.46, 95.16, 96.54, 97.13, 97.43, 97.61,97.61,97.73,97.67,97.61])/100
        rect_eff = ac2dc_rectification_eff*dc2dc_rectification_eff
        f=interpolate.interp1d(x_load_percent,rect_eff)
        start_idx = 0
        end_idx = 8760
        nTurbs = self.num_clusters
        clusters = self.create_clusters()
        available_turbs = wind_plant._system_model.nTurbs
        # power_turbines = np.zeros((available_turbs, 8760))
        power_turbines = np.zeros((8760,available_turbs))
        power_to_clusters = np.zeros((8760,self.num_clusters))
        ac2dc_rated_power_kw = wind_plant.turb_rating
        
        # power_turbines[:, start_idx:end_idx] = wind_plant._system_model.fi.get_turbine_powers().reshape((available_turbs, end_idx - start_idx))/1000
        # print(np.sum(power_turbines))
        power_turbines[:, start_idx:end_idx] = wind_plant._system_model.fi.get_turbine_powers().reshape((end_idx - start_idx,available_turbs))/1000
        power_turbines = np.where(power_turbines > ac2dc_rated_power_kw,ac2dc_rated_power_kw,power_turbines)
        power_to_clusters = (power_turbines)*(f(power_turbines/ac2dc_rated_power_kw))
        approx_power_loss =  np.sum(power_to_clusters)-np.sum(power_turbines)
        approx_perc_power_loss = 100*(approx_power_loss/np.sum(power_turbines))
        if self.tot_system_size_mw % nTurbs !=0:#self.num_clusters < available_turbs:
            turb_power_mw = wind_plant.turb_rating/1e3
            available_turb_mw = available_turbs*turb_power_mw
            
            residual_cluster_cap_mw = self.tot_system_size_mw % self.num_clusters
            resid_cluster = self.create_filler_cluster(residual_cluster_cap_mw)
            nturbs_extra = np.ceil(residual_cluster_cap_mw/turb_power_mw)
            resid_turb_power =power_turbines[len(clusters)-1] 
            resid_turb_power = np.where(resid_turb_power > residual_cluster_cap_mw,residual_cluster_cap_mw*1e3,resid_turb_power)
            power_to_clusters[len(clusters)-1] = (resid_turb_power)*(f(resid_turb_power/(residual_cluster_cap_mw*1e3)))
            clusters.extend(resid_cluster)
            nTurbs = nTurbs + nturbs_extra



        # turb_power_mw = wind_plant.turb_rating/1e3
        # residual_cluster_cap = self.tot_system_size_mw % turb_power_mw 
        
        # power_farm *((100 - 12.83)/100) / 1000

        
       
        h2_df_ts = pd.DataFrame()
        h2_df_tot = pd.DataFrame()
        # h2_dict_ts={}
        # h2_dict_tot={}
        #TODO check the size of pwoer to clusters!
        #TODO ADD PIPELINE H2 LOSSES
        
        col_names = []
        start=time.perf_counter()
        for ci,cluster in enumerate(clusters):
            cl_name = 'Cluster #{}'.format(ci)
            col_names.append(cl_name)
            h2_ts,h2_tot = clusters[ci].run(power_to_clusters[ci])
            # h2_dict_ts['Cluster #{}'.format(ci)] = h2_ts
            
            h2_ts_temp = pd.Series(h2_ts,name = cl_name)
            h2_tot_temp = pd.Series(h2_tot,name = cl_name)
            if len(h2_df_tot) ==0:
                # h2_df_ts=pd.concat([h2_df_ts,h2_ts_temp],axis=0,ignore_index=False)
                h2_df_tot=pd.concat([h2_df_tot,h2_tot_temp],axis=0,ignore_index=False)
                h2_df_tot.columns = col_names

                h2_df_ts=pd.concat([h2_df_ts,h2_ts_temp],axis=0,ignore_index=False)
                h2_df_ts.columns = col_names
            else:
                # h2_df_ts = h2_df_ts.join(h2_ts_temp)
                h2_df_tot = h2_df_tot.join(h2_tot_temp)
                h2_df_tot.columns = col_names

                h2_df_ts = h2_df_ts.join(h2_ts_temp)
                h2_df_ts.columns = col_names

        end=time.perf_counter()
        if self.verbose:
            print('Took {} sec to run the distributed PEM case function'.format(round(end-start,3)))
            print('########################')
            print('Approximate Power Loss of {} kW ({} percent of generated power)'.format(round(approx_power_loss),round(approx_perc_power_loss,2)))
            print('########################')
        
        return h2_df_ts, h2_df_tot
        []
    
    def get_fully_centralized_power_losses(self,wind_plant,cable_info):
        print("Calculating power losses for centralized system ... this may take a while")
        start=time.perf_counter()
        component_list={}
        losses_list={}
        power_keys = ['Initial Power [MW] (unlimited)','Initial Power [MW] (saturated to rated)','Power after Transformer [MW]','Power After Distribution [MW]','Power After Rectifier [MW]']
        
        loss_keys = ['Transformer Loss [kW]','Transformer Loss [%]','Cable Loss [kW]','Cable Loss [%]','Rectifier Loss [kW]','Rectifier Loss [%]','Total Losses [kW]','Total Losses [%]']
        power_data=[]
        
        # ts_keys = ['Initial Power (saturated)','Power to Cables','Power to Rectifier','Power to PEM']
        available_turbs = wind_plant._system_model.nTurbs
        start_idx = 0
        end_idx = 8760
        x_load_percent = np.linspace(0.0,1.0,11)
        ac2ac_transformer_eff=np.array([90,90.63, 93.91, 95.63, 96.56, 97.19, 97.50, 97.66, 97.66, 97.66, 97.50])/100
        ac2dc_rectification_eff=np.array([96,96.54, 98.12, 98.24, 98.6, 98.33, 98.03, 97.91, 97.43, 97.04, 96.687])/100
        dc2dc_rectification_eff=np.array([91,91.46, 95.16, 96.54, 97.13, 97.43, 97.61,97.61,97.73,97.67,97.61])/100
        rect_eff = ac2dc_rectification_eff*dc2dc_rectification_eff
        f_ac2ac=interpolate.interp1d(x_load_percent,ac2ac_transformer_eff)
        f_ac2dc=interpolate.interp1d(x_load_percent,rect_eff)
        Vline_turb = 480
        P_rated_turb=cable_info['Turb Power [kW]']#*1e3
        nturbs_percable = cable_info['Turbs/Cable']
        TN=cable_info['Transformer Turns']
        Iline_turb = cable_info['Primary Turb Current']
        R_cable = cable_info['Cable Resistance']
        n_cables = cable_info['n_cables']
        l_cable = cable_info['Cable Lengths']
        V_cable = cable_info['V_cable']
        d_turb = cable_info['Distance Between Turbs']
        T1_rated_power_kw = 1.2*P_rated_turb
        component_keys = ['Component','Use','Number of Components','Rated Power [kW]','Input Rating','Output Rating']
        transformer_data = ['Step-Up Transformer','Turb->Distribution Cable',available_turbs,T1_rated_power_kw,str(Vline_turb) + ' V', str(V_cable) + ' V']
        component_list.update({'Transformer':dict(zip(component_keys,transformer_data))})
        component_list.update({'Cables':cable_info})


        # power_turbines = np.zeros((available_turbs, 8760))
        power_turbines = np.zeros((8760,available_turbs))
        power_to_cables = np.zeros(( 8760,len(l_cable)))
        power_loss_cables = np.zeros(( 8760,len(l_cable)))
        power_from_cables = np.zeros(( 8760,len(l_cable)))
        voltage_drop = np.zeros(( 8760,len(l_cable)))
        # power_loss_cables = np.zeros((len(l_cable), 8760))
        # power_from_cables = np.zeros((len(l_cable), 8760))
        # voltage_drop = np.zeros((len(l_cable), 8760))

        #hybrid_plant.wind._system_model.nTurbs
        # rot_diam = wind_plant.rotor_diameter
        # turb2turb_dist = 5*rot_diam
        # kiloft_to_km = 0.3048
        # nmax_cables = 10
        
        # power_turbines[:, start_idx:end_idx] = wind_plant._system_model.fi.get_turbine_powers().reshape((available_turbs, end_idx - start_idx))/1000
        power_turbines[:, start_idx:end_idx] = wind_plant._system_model.fi.get_turbine_powers().reshape((end_idx - start_idx,available_turbs))/1000
        # print('Annual Energy (ESG) [kWh]: {}'.format(round(np.sum(power_turbines),3)))
        power_data.append(np.sum(power_turbines)/1000)
        power_turbines = np.where(power_turbines > cable_info['Turb Power [kW]'],cable_info['Turb Power [kW]'],power_turbines)
        power_data.append(np.sum(power_turbines)/1000)
        turbpower_to_cable = (power_turbines)*(f_ac2ac(power_turbines/T1_rated_power_kw))
        power_data.append(np.sum(turbpower_to_cable)/1000)
        turbpower_to_cable = np.hsplit(turbpower_to_cable,len(l_cable))
        nt=0
        for i,t2c in enumerate(turbpower_to_cable):
            # d_tot = l_cable[i]
            #all axis = 1 used to be axis=0
            power_to_cable = np.cumsum(t2c,axis=1)[:,-1]
            d_row2load = l_cable[i] - d_turb*(nturbs_percable-1)
            turb_current = (t2c*1e3)/(np.sqrt(3)*V_cable) #after transformer
            cumulative_i = np.cumsum(turb_current,axis=1)
            vdrop_segment = cumulative_i*R_cable*d_turb            
            vdrop_segment[:,-1] = (vdrop_segment[:,-1]/d_turb)*d_row2load

            p_loss_segment = np.sqrt(3)*(cumulative_i**2)*R_cable*d_turb
            p_loss_segment[:,-1]=(p_loss_segment[:,-1]/d_turb)*d_row2load
            p_loss_cable_kW = np.cumsum(p_loss_segment,axis=1)[:,-1]/1000
            
            voltage_drop[:,i] = np.cumsum(vdrop_segment,axis=1)[:,-1]
            # voltage_drop[i] = np.cumsum(vdrop_segment,axis=1)[-1]
            power_to_cables[:,i]=power_to_cable
            power_loss_cables[:,i] = p_loss_cable_kW
            power_from_cables[:,i] = power_to_cable - p_loss_cable_kW
        []
        
        power_to_allrectifier = np.sum(power_from_cables,axis=1)
        power_data.append(np.sum(power_to_allrectifier)/1000)
        rectifier_rated_size_kw = 125*1e3
        n_rectifier = 8
        power_per_rectifier = power_to_allrectifier/n_rectifier
        rectpower_to_pem = (power_per_rectifier)*(f_ac2dc(power_per_rectifier/rectifier_rated_size_kw))
        final_power_to_pem = rectpower_to_pem*n_rectifier
        power_data.append(np.sum(final_power_to_pem)/1000)
        rectifier_data = ['Rectifier','AC->DC',n_rectifier,rectifier_rated_size_kw,str(V_cable) + ' Vac','1500 Vdc']
        pem_info={'n_clusters':n_rectifier,'cluster_cap_mw':rectifier_rated_size_kw/1000}
        component_list.update({'Rectifier':dict(zip(component_keys,rectifier_data))})
        component_list.update({'PEM organization':pem_info})
        loss_data = np.array([np.sum(turbpower_to_cable)-np.sum(power_turbines),100*(np.sum(turbpower_to_cable)-np.sum(power_turbines))/np.sum(power_turbines),\
            -1*np.sum(power_loss_cables),-100*np.sum(power_loss_cables)/np.sum(turbpower_to_cable),np.sum(final_power_to_pem)-np.sum(power_to_allrectifier),\
                100*(np.sum(final_power_to_pem)-np.sum(power_to_allrectifier))/np.sum(final_power_to_pem),np.sum(final_power_to_pem)-np.sum(power_turbines),\
                    100*(np.sum(final_power_to_pem)-np.sum(power_turbines))/np.sum(power_turbines)])
        ts_data = {'Initial Power [kW](saturated)':np.sum(power_turbines,axis=1),'Power to Cables [kW]':np.sum(turbpower_to_cable,axis=1),
        'Power to Rectifier [kW]':power_to_allrectifier,'Power to PEM [kW]':final_power_to_pem} 
        losses_list.update({'Annual Power Per Step [MW]':dict(zip(power_keys,power_data))})
        losses_list.update({'Annual Losses [kW]':dict(zip(loss_keys,loss_data))})
        losses_list.update({'Time-Series Data [kW]':ts_data})
        #assuming we can just combine all the power from the lines
        #on the bus so the bus power is 
        #dynapower high power rectifier
        #AC input: 69 kV, 3 phase, 50-60 Hz
        #DC output 100,000 Adc and 1500 Vdc = 150 MW
        #what if 8 PEM clusters each of 125 MW capacity
        
        end=time.perf_counter()
        print('Took {} sec to calculate the centralized case power losses'.format(round(end-start,3)))

        return losses_list,component_list,final_power_to_pem, n_rectifier
    def run_power_elec_pysam(self,electrical_generation_timeseries,nturbs_per_cable,nturbs_dist2_load,n_distances,wind_plant):
        # avg_power_per_turb = electrical_generation_timeseries/self.n_turbs
        start=time.perf_counter()
        # n_turbs = wind_plant._system_model.nTurbs
        
        cable_info = self.find_best_cable(nturbs_per_cable,nturbs_dist2_load,n_distances,wind_plant)
        self.component_list.update({'Cables':cable_info})
        P_rated_turb=cable_info['Turb Power [kW]']#*1e3
        Vline_turb = 480
        nturbs_percable = cable_info['Turbs/Cable']
        # TN=cable_info['Transformer Turns']
        # Iline_turb = cable_info['Primary Turb Current']
        R_cable = cable_info['Cable Resistance']
        n_cables = cable_info['n_cables']
        l_cable = cable_info['Cable Lengths']
        V_cable = cable_info['V_cable']
        d_turb = cable_info['Distance Between Turbs']
        T1_rated_power_kw = 1.2*P_rated_turb

        # power = self.get_floris_power_and_saturate(wind_plant,P_rated_turb)
        power = self.split_pysam_power_and_adjust(electrical_generation_timeseries,P_rated_turb)
        power = self.do_transformer_losses(power,T1_rated_power_kw,l_cable,self.n_turbs,Vline_turb,V_cable)
        power = self.do_cable_losses(power,l_cable,R_cable,d_turb,V_cable,nturbs_percable,n_cables)
        power,n_clusters = self.do_rectifier_losses(power,V_cable)
        end=time.perf_counter()
        print('Took {} sec to calculate the centralized case power losses'.format(round(end-start,3)))
        return power, n_clusters
        

    def run_power_elec_floris(self,nturbs_per_cable,nturbs_dist2_load,n_distances,wind_plant):
        start=time.perf_counter()
        n_turbs = wind_plant._system_model.nTurbs
        
        cable_info = self.find_best_cable(nturbs_per_cable,nturbs_dist2_load,n_distances,wind_plant)
        self.component_list.update({'Cables':cable_info})
        P_rated_turb=cable_info['Turb Power [kW]']#*1e3
        Vline_turb = 480
        nturbs_percable = cable_info['Turbs/Cable']
        # TN=cable_info['Transformer Turns']
        # Iline_turb = cable_info['Primary Turb Current']
        R_cable = cable_info['Cable Resistance']
        n_cables = cable_info['n_cables']
        l_cable = cable_info['Cable Lengths']
        V_cable = cable_info['V_cable']
        d_turb = cable_info['Distance Between Turbs']
        T1_rated_power_kw = 1.2*P_rated_turb
        
        
        power = self.get_floris_power_and_saturate(wind_plant,P_rated_turb)
        power = self.do_transformer_losses(power,T1_rated_power_kw,l_cable,n_turbs,Vline_turb,V_cable)
        power = self.do_cable_losses(power,l_cable,R_cable,d_turb,V_cable,nturbs_percable,n_cables)
        power,n_clusters = self.do_rectifier_losses(power,V_cable)
        end=time.perf_counter()
        tot_diff=self.power_data['Power After Additional Default Load Bus Losses [MW]']-self.power_data['Inital Power [MW] with Default Losses']
        totpdiff = 100*tot_diff/self.power_data['Inital Power [MW] with Default Losses']
        self.loss_data = pd.concat([self.loss_data,pd.Series(dict(zip(['Total Loss [MW]','Total Loss [%]'],np.array([tot_diff,totpdiff]))))])
        print('Took {} sec to calculate the centralized case power losses'.format(round(end-start,3)))
        print('########################')
        print('(Centralized): Approximate Power Loss of {} MW ({} percent of generated power)'.format(round(tot_diff),round(totpdiff,2)))
        print('########################')
        return power, n_clusters

    def split_pysam_power_and_adjust(self,electrical_generation_timeseries,turb_rating_kw):
        tot_pwr_keys=['Initial PySam Power [MW]','Initial Power [MW] Without Losses','Inital Power [MW] with Default Losses']
        power_data=[]
        power_data.append(np.sum(electrical_generation_timeseries)/1000)
        avg_power_per_turb = electrical_generation_timeseries/self.n_turbs
        add_in_losses = np.sum(self.prelim_losses.values)
        avg_power_per_turb = avg_power_per_turb/((100-add_in_losses)/100)
        
        avg_power_per_turb = np.where(avg_power_per_turb > turb_rating_kw,turb_rating_kw,avg_power_per_turb)
        self.num_col = len(electrical_generation_timeseries)
        power_turbines = np.repeat(avg_power_per_turb,self.n_turbs).reshape(self.n_turbs,self.num_col)
        power_data.append(np.sum(power_turbines)/1000)

        additional_losses = np.sum(self.prelim_losses[['avail_turb','ops_env','env_deg','env_exp','icing']].values)
        power_turbines = power_turbines*((100-additional_losses)/100)
        power_data.append(np.sum(power_turbines)/1000)

        self.power_data = pd.concat([self.power_data,pd.Series(dict(zip(tot_pwr_keys,power_data)))])
        loss_keys = ['Turbine Loss [MW]','Turbine Loss [%]']
        loss_vals = np.array([power_data[1]-power_data[-1],100*(power_data[1]-power_data[-1])/power_data[-1]])
        self.loss_data = pd.concat([self.loss_data,pd.Series(dict(zip(loss_keys,loss_vals)))])
        self.ts_data = pd.concat([self.ts_data,{'Init Power [kW]':np.sum(power_turbines,axis=1)}],ignore_index=False)

        return power_turbines




    def get_floris_power_and_saturate(self,wind_plant,turb_rating_kw):
        self.num_col = 8760
        tot_pwr_keys=['Initial Power [MW] (unlimited)','Initial Power [MW] (saturated to rated)','Inital Power [MW] with Default Losses']
        power_data=[]
        start_idx = 0
        end_idx = 8760
        available_turbs = self.n_turbs
        # available_turbs = wind_plant._system_model.nTurbs
        additional_losses = np.sum(self.prelim_losses[['avail_turb','ops_env','env_deg','env_exp','icing']].values)
        power_turbines = np.zeros((8760,available_turbs))

        power_turbines[:, start_idx:end_idx] = wind_plant._system_model.fi.get_turbine_powers().reshape((end_idx - start_idx,available_turbs))/1000
        power_data.append(np.sum(power_turbines)/1000)

        power_turbines = np.where(power_turbines > turb_rating_kw,turb_rating_kw,power_turbines)
        power_data.append(np.sum(power_turbines)/1000)

        power_turbines = power_turbines*((100-additional_losses)/100)
        power_data.append(np.sum(power_turbines)/1000)
        self.power_data = pd.concat([self.power_data,pd.Series(dict(zip(tot_pwr_keys,power_data)))])
        loss_keys = ['Turbine Loss [MW]','Turbine Loss [%]']
        loss_vals = np.array([power_data[-1]-power_data[0],100*(power_data[-1]-power_data[0])/power_data[0]])
        self.loss_data = pd.concat([self.loss_data,pd.Series(dict(zip(loss_keys,loss_vals)))])
        self.ts_data = pd.concat([self.ts_data,pd.DataFrame({'Init Power [kW]':np.sum(power_turbines,axis=1)})],ignore_index=False)
        return power_turbines
        []
    def do_transformer_losses(self,turbpower_to_transformer,T1_rated_power_kw,l_cable,n_turbs,Vline_turb,V_cable):
        x_load_percent = np.linspace(0.0,1.0,11)
        ac2ac_transformer_eff=np.array([90,90.63, 93.91, 95.63, 96.56, 97.19, 97.50, 97.66, 97.66, 97.66, 97.50])/100
        f_ac2ac=interpolate.interp1d(x_load_percent,ac2ac_transformer_eff)
        turbpower_to_cable = (turbpower_to_transformer)*(f_ac2ac(turbpower_to_transformer/T1_rated_power_kw))
        power_data=[(np.sum(turbpower_to_cable)/1000)]
        tot_pwr_keys=['Power after Transformer [MW]']
        self.power_data = pd.concat([self.power_data,pd.Series(dict(zip(tot_pwr_keys,power_data)))])
        self.ts_data = pd.concat([self.ts_data,pd.DataFrame({'Power after Transformer [kW]':np.sum(turbpower_to_cable,axis=1)})],axis=1)
        turbpower_to_cable = np.hsplit(turbpower_to_cable,len(l_cable))
        loss_keys = ['Transformer Loss [MW]','Transformer Loss [%]']
        loss_vals = np.array([power_data[0]-(np.sum(turbpower_to_transformer)/1000),100*(power_data[0]-(np.sum(turbpower_to_transformer)/1000))/(np.sum(turbpower_to_transformer)/1000)])
        self.loss_data = pd.concat([self.loss_data,pd.Series(dict(zip(loss_keys,loss_vals)))])
        # self.ts_data = pd.concat([self.ts_data,pd.DataFrame({'Power after Transformer [kW]':np.sum(turbpower_to_cable,axis=1)})])
        component_keys = ['Component','Use','Number of Components','Rated Power [kW]','Input Rating','Output Rating']
        transformer_data = ['Step-Up Transformer','Turb->Distribution Cable',n_turbs,T1_rated_power_kw,str(Vline_turb) + ' V', str(V_cable) + ' V']
        self.component_list.update({'Transformer':dict(zip(component_keys,transformer_data))})

        return turbpower_to_cable

    def do_cable_losses(self,turbpower_to_cable,l_cable,R_cable,d_turb,V_cable,nturbs_percable,n_cables):
        additional_losses = np.sum(self.prelim_losses[['avail_grid','ops_grid']].values)
        tot_pwr_keys =['Power After Cable Losses [MW]','Power After Additional Default Distribution Losses [MW]']
        power_data=[]

        power_to_cables = np.zeros(( self.num_col,len(l_cable)))
        power_loss_cables = np.zeros(( self.num_col,len(l_cable)))
        power_from_cables = np.zeros(( self.num_col,len(l_cable)))
        voltage_drop = np.zeros(( self.num_col,len(l_cable)))
        for i,t2c in enumerate(turbpower_to_cable):
            # d_tot = l_cable[i]
            #all axis = 1 used to be axis=0
            power_to_cable = np.cumsum(t2c,axis=1)[:,-1]
            d_row2load = l_cable[i] - d_turb*(nturbs_percable-1)
            turb_current = (t2c*1e3)/(np.sqrt(3)*V_cable) #after transformer
            cumulative_i = np.cumsum(turb_current,axis=1)
            vdrop_segment = cumulative_i*R_cable*d_turb            
            vdrop_segment[:,-1] = (vdrop_segment[:,-1]/d_turb)*d_row2load

            # p_loss_segment = np.sqrt(3)*(cumulative_i**2)*R_cable*d_turb
            p_loss_segment = np.sqrt(3)*((cumulative_i/n_cables)**2)*R_cable*d_turb*n_cables
            p_loss_segment[:,-1]=(p_loss_segment[:,-1]/d_turb)*d_row2load
            p_loss_cable_kW = np.cumsum(p_loss_segment,axis=1)[:,-1]/1000
            
            voltage_drop[:,i] = np.cumsum(vdrop_segment,axis=1)[:,-1]
            # voltage_drop[i] = np.cumsum(vdrop_segment,axis=1)[-1]
            power_to_cables[:,i]=power_to_cable
            power_loss_cables[:,i] = p_loss_cable_kW
            power_from_cables[:,i] = power_to_cable - p_loss_cable_kW
        power_data.append(np.sum(power_from_cables)/1000)
        power_from_cables = power_from_cables*((100-additional_losses)/100)
        
        #TODO add something
        power_to_allrectifier = np.sum(power_from_cables,axis=1)
        power_data.append(np.sum(power_to_allrectifier)/1000)
        self.power_data = pd.concat([self.power_data,pd.Series(dict(zip(tot_pwr_keys,power_data)))])
        loss_keys = ['Cable Loss [MW]','Cable Loss [%]']
        loss_vals = np.array([power_data[1]-(np.sum(turbpower_to_cable)/1000),100*(power_data[1]-(np.sum(turbpower_to_cable)/1000))/(np.sum(turbpower_to_cable)/1000)])
        self.loss_data = pd.concat([self.loss_data,pd.Series(dict(zip(loss_keys,loss_vals)))])
        self.ts_data = pd.concat([self.ts_data,pd.DataFrame({'Power after Cable [kW]':power_to_allrectifier})],axis=1)
        return power_to_allrectifier
        []
    def do_rectifier_losses(self,power_to_allrectifier,V_cable):
        tot_pwr_keys =['Power After Rectifier [MW]','Power After Additional Default Load Bus Losses [MW]']
        power_data=[]
        additional_losses = np.sum(self.prelim_losses[['ops_load','avail_bop','elec_parasitic']].values)
        x_load_percent = np.linspace(0.0,1.0,11)
        ac2dc_rectification_eff=np.array([96,96.54, 98.12, 98.24, 98.6, 98.33, 98.03, 97.91, 97.43, 97.04, 96.687])/100
        dc2dc_rectification_eff=np.array([91,91.46, 95.16, 96.54, 97.13, 97.43, 97.61,97.61,97.73,97.67,97.61])/100
        rect_eff = ac2dc_rectification_eff*dc2dc_rectification_eff
        f_ac2dc=interpolate.interp1d(x_load_percent,rect_eff)
        
        # power_data.append(np.sum(power_to_allrectifier)/1000)
        rectifier_rated_size_kw = 125*1e3
        n_rectifier = 8
        power_per_rectifier = power_to_allrectifier/n_rectifier
        rectpower_to_pem = (power_per_rectifier)*(f_ac2dc(power_per_rectifier/rectifier_rated_size_kw))
        final_power_to_pem = rectpower_to_pem*n_rectifier
        power_data.append(np.sum(final_power_to_pem)/1000)
        final_power_to_pem = final_power_to_pem*((100-additional_losses)/100)
        power_data.append(np.sum(final_power_to_pem)/1000)
        self.power_data = pd.concat([self.power_data,pd.Series(dict(zip(tot_pwr_keys,power_data)))])

        component_keys = ['Component','Use','Number of Components','Rated Power [kW]','Input Rating','Output Rating']
        rectifier_data = ['Rectifier','AC->DC',n_rectifier,rectifier_rated_size_kw,str(V_cable) + ' Vac','1500 Vdc']
        pem_info={'n_clusters':n_rectifier,'cluster_cap_mw':rectifier_rated_size_kw/1000}
        self.component_list.update({'Rectifier':dict(zip(component_keys,rectifier_data))})
        self.component_list.update({'PEM organization':pem_info})
        loss_keys = ['Rectifier Loss [MW]','Rectifier Loss [%]']
        loss_vals = np.array([power_data[1]-(np.sum(power_to_allrectifier)/1000),100*(power_data[1]-(np.sum(power_to_allrectifier)/1000))/(np.sum(power_to_allrectifier)/1000)])
        self.loss_data = pd.concat([self.loss_data,pd.Series(dict(zip(loss_keys,loss_vals)))])
        self.ts_data = pd.concat([self.ts_data,pd.DataFrame({'Power after Rectifier [kW]':final_power_to_pem})],axis=1)

        #TODO add something
        []
        return final_power_to_pem, n_rectifier
 




    def find_best_cable(self,nturbs_per_cable,nturbs_dist2_load,n_distances,wind_plant):
        rot_diam = wind_plant.rotor_diameter
        turb2turb_dist = 5*rot_diam
        cable_lengths=self.get_cable_length(nturbs_per_cable,turb2turb_dist,nturbs_dist2_load,n_distances)
        kiloft_to_km = 0.3048
        nmax_cables = 10
        Vline_turb = 480
        turb_cap_kw = wind_plant.turb_rating
        Iline_turb=(turb_cap_kw*1e3)/(np.sqrt(3)*Vline_turb)
        v_cable = 34.5*1000
        T1_turns = v_cable/Vline_turb
        i_turb = Iline_turb/T1_turns
        cable_resistance_per_kft = np.array([0.12,0.25,0.02,0.01,0.009])
        cbl_names = np.array(["AWG 1/0","AWG 4/0","MCM 500","MCM 1000","MCM 1250"])
        cable_resistance_per_m = (cable_resistance_per_kft *(1/kiloft_to_km))/(1e3)
        cable_ampacity = np.array([150,230,320,455,495])
        cable_cost_per_m = np.array([61115.1602528554,72334.3683802817,96358.26769213431,104330.7086713996,115964.28690974298])/1000

        i_to_cable = i_turb * nturbs_per_cable
        cable_required_power = (turb_cap_kw*1e3)*nturbs_per_cable
        n_cables = np.ceil(i_to_cable/cable_ampacity)
        p_line_max=np.sqrt(3)*cable_ampacity*n_cables*v_cable
        cb_idx = np.argwhere((p_line_max >= cable_required_power) & (n_cables<=nmax_cables))
        cb_idx = cb_idx.reshape(len(cb_idx))

        i_per_cable = i_to_cable/n_cables[cb_idx]
        i_rated_cable = cable_ampacity[cb_idx]
        cable_r = cable_resistance_per_m[cb_idx]
        n_cables_okay = n_cables[cb_idx]
        names = cbl_names[cb_idx]
        p_loss_per_cable = (i_per_cable**2)*cable_r
        p_loss_tot_per_m = p_loss_per_cable*n_cables[cb_idx]
        idx_lowestloss = np.argmin(p_loss_tot_per_m)

        r_cable = cable_r[idx_lowestloss]
        num_cables = n_cables_okay[idx_lowestloss]
        i_cable = i_rated_cable[idx_lowestloss]
        cable_type = names[idx_lowestloss]
        cost = cable_cost_per_m[cb_idx]
        cost = cost[idx_lowestloss]
        cable_info = {'V_cable':v_cable,'Cable Ampacity':i_cable,'Cable Name':cable_type,'n_cables':num_cables,
        'Cable Resistance':r_cable,'Cable Lengths':cable_lengths,'Transformer Turns':T1_turns, 'Cost/m':cost,
        'Primary Turb Current':Iline_turb,'Turbs/Cable':nturbs_per_cable,'Turb Power [kW]':turb_cap_kw,'Distance Between Turbs':turb2turb_dist}
        
        return cable_info

    def get_cable_length(self,nturbs_per_cable,t2t_dist,nturbs_dist2_load,n_distances):
        # within_row_length  = np.arange(0,nturbs_per_cable,1)*t2t_dist
        
        within_row_length = [(nturbs_per_cable-1)*t2t_dist]
        cable_lengths = []
        for d_row2load in nturbs_dist2_load:
            turbs2load = np.arange(0.5,d_row2load+0.5,1)
            # turbs2load = np.arange(0,d_row2load,1)
        # turbs2load = np.arange(0,nturbs_dist2_load,1)
        # row_2_load_length = nturbs_dist2_load*t2t_dist
            row_2_load_length = turbs2load *t2t_dist
            
            for row_length in within_row_length:
                l = row_length + row_2_load_length
                cable_lengths.extend(list(l))
            # cable_lengths = within_row_length + row_2_load_length
        all_cable_lengths = cable_lengths*n_distances
        return np.array(all_cable_lengths)

class dc_component_power_electronics:
    '''Add description and stuff :)'''
    # def __init__(self,electrical_power_signal,nturbs_per_cable,nturbs_dist2_load,n_distances,wind_plant,n_turbs):
    def __init__(self,hybrid_plant,dc_component_info,external_power_kw):
        #https://www.energy.gov/eere/solar/articles/solar-plus-storage-101
        # self.pv_default_inv_eff=hybrid_plant.pv._system_model.SystemDesign.inv_eff
        
        # self.pv_output_ac=hybrid_plant.pv._system_model.Outputs.ac
        self.pv_output_dc=hybrid_plant.pv._system_model.Outputs.dc
        self.pv_rating_kw = hybrid_plant.pv._system_model.SystemDesign.system_capacity
        self.pv_default_losses = hybrid_plant.pv._system_model.SystemDesign.losses #DC losses PySam [%]
        # dc_component_info['Co-located Components']
        # dc_component_info['DC Component Distance to Load']
        self.flex_batpe_sizes=dc_component_info['Flexible Battery Charge Rate'] 
        # dc_component_info['Co-located Component Num Clusters']
        self.load_demand_kw=dc_component_info['Rated Load [kW]']
        
        # if dc_component_info['Co-located Component Loc'] ==dc_component_info['Load Loc']:
        #     self.run_colocated_dc_bus(hybrid_plant,external_power_kw)
    def run_colocated_dc_bus(self,hybrid_plant,external_power_kw):
        #pv -> PEM needs dc/dc converter
        #pv -> battery needs dc/dc converter
        #battery -> PEM needs dc/dc converter
        # powers_keys=['Init','Init (saturated)','Bat->PEM','PV->PEM','Final to PEM']
        loss_keys = ['Unidirect (PV->PEM)','Bidirect (PV->Bat)','Bidirect (Bat->PEM)','Total']
        
        power=[]
        losses={}
        pv_power_keys = ['Initial [kWdc]','Saturated','PV to PEM (init)','PV to Battery (init)',\
            'PV to Battery (final)','PV to PEM (final)','Battery to PEM (init)','Battery to PEM (final)']
        self.set_up_pv(hybrid_plant)
        power.append(np.sum(self.pv_output_kWdc))
        self.set_up_battery(hybrid_plant)
        pv_init_kWdc = np.where(self.pv_output_kWdc>self.pv_rating_kWdc,self.pv_rating_kWdc,self.pv_output_kWdc)
        pe_sizes=self.size_dc_components(self.flex_batpe_sizes,external_power_kw,self.load_demand_kw,self.pv_rating_kWdc)
        power.append(np.sum(pv_init_kWdc))
        
        load=self.load_demand_kw*np.ones(len(pv_init_kWdc))
        wind_shortfall = [x - y for x, y in
                             zip(load,external_power_kw)]
        wind_shortfall = [x if x > 0 else 0 for x in wind_shortfall]
        solar_curtailment = [x - y for x, y in
                                zip(pv_init_kWdc,wind_shortfall)]
        solar_curtailment=[x if x > 0 else 0 for x in solar_curtailment]
        solar_2_pem = pv_init_kWdc - np.array(solar_curtailment)
        power.append(np.sum(solar_2_pem))
        gen_pwr_2_pem = solar_2_pem + external_power_kw
        wind_solar_shortfall = [x - y for x, y in
                             zip(load,gen_pwr_2_pem)]
        wind_solar_shortfall = [x if x > 0 else 0 for x in wind_solar_shortfall]
        battery_used, excess_energy, battery_SOC=self.run_dispatch(solar_curtailment,wind_solar_shortfall)

        bat_getting_charged = np.diff(battery_SOC)
        pv2bat= np.where(bat_getting_charged>0,bat_getting_charged,0)
        power.append(np.sum(pv2bat))
        
        pv_power2pem=self.dc2dc_unidirect_transformer(pe_sizes['PV [DC/DC] Uni-direct'],solar_2_pem )
        pv2pem_transformer_loss_kw = np.sum(pv_power2pem) - np.sum(solar_2_pem)
        pv2pem_transformer_loss_perc = 100*pv2pem_transformer_loss_kw/np.sum(solar_2_pem)

        pv_power2bat=self.dc2dc_bidirect_converter(pe_sizes['Battery [DC/DC] Bi-direct'],pv2bat)
        battery_used_eff, excess_energy, battery_SOC_temp=self.run_dispatch(pv_power2bat,wind_solar_shortfall)
        bat2pem = np.array(battery_used_eff)
        pv2bat_bidirectconv_loss_kw = np.sum(pv_power2bat) - np.sum(pv2bat)
        pv2bat_bidirectconv_loss_perc = 100*pv2bat_bidirectconv_loss_kw/np.sum(pv2bat)

        # battery_used, excess_energy, battery_SOC=self.run_dispatch(solar_curtailment,wind_solar_shortfall)
        bat_power2pem=self.dc2dc_bidirect_converter(pe_sizes['Battery [DC/DC] Bi-direct'],bat2pem)
        final_power_to_pem = external_power_kw + pv_power2pem + bat_power2pem
        bat2pem_bidirectconv_loss_kw = np.sum(bat_power2pem) - np.sum(bat2pem)
        bat2pem_bidirectconv_loss_perc = 100*bat2pem_bidirectconv_loss_kw /np.sum(bat2pem)
        tot_loss_kw =  (np.sum(pv_power2pem) + np.sum(bat_power2pem))-np.sum(self.pv_output_kWdc)
        tot_loss_perc = 100*tot_loss_kw/np.sum(self.pv_output_kWdc)
        pwr_to_pem_tot = (np.sum(pv_power2pem) + np.sum(bat_power2pem))
        mw_losses = np.array([pv2pem_transformer_loss_kw,pv2bat_bidirectconv_loss_kw,bat2pem_bidirectconv_loss_kw,tot_loss_kw])/1000
        perc_losses = [pv2pem_transformer_loss_perc,pv2bat_bidirectconv_loss_perc,bat2pem_bidirectconv_loss_perc,tot_loss_perc]
        losses['MW']=pd.Series(dict(zip(loss_keys,mw_losses)),name='PV-Cent')
        losses['%']=pd.Series(dict(zip(loss_keys,perc_losses)),name='PV-Cent')

        power.extend([np.sum(pv_power2bat),np.sum(pv_power2pem),np.sum(battery_used_eff),np.sum(bat_power2pem)])
        power_infolosses=pd.Series(dict(zip(pv_power_keys,power)))
        ts_data = pd.DataFrame({'Wind':external_power_kw,'PV':pv_power2pem,'Battery':bat_power2pem})
        # pe_sizes['Battery [DC/DC] Bi-direct']
        # pe_sizes['PV [DC/DC] Uni-direct']
        # combined_pv_wind_storage_power_production_hopp = combined_pv_wind_power_production_hopp + battery_used
        # return final_power_to_pem,losses,ts_data
        return power_infolosses,losses,ts_data
    def run_distributed_pv_central_bat(self,hybrid_plant,external_power_kw,n_pv_panels,V_cable,cable_lengths):
        pv_power={}
        losses={}
        powers_keys=['Initial [kWdc]','Inverter','AC/AC Transformer','Cables','Rectifier','PV to PEM (final)','PV to Battery (final)','Battery to PEM (final)','Final to PEM']
        loss_keys = ['Inverter','AC/AC Transformer','Cables','Rectifier','Unidirect (PV->PEM)','Bidirect (PV->Bat)','Bidirect (Bat->PEM)','Total']
        pv_info={'n_panels':n_pv_panels}
        self.set_up_pv(hybrid_plant)
        self.set_up_battery(hybrid_plant)
        power_per_pv = self.pv_output_kWdc/n_pv_panels
        pv_power.update({'Power/Panel (init-kWdc)':power_per_pv})
        pv_panel_rating_dc = self.pv_rating_kWdc/n_pv_panels
        pv_info.update({'panel cap [kWdc]':pv_panel_rating_dc})
        pe_sizes=self.size_dc_components(False,external_power_kw,self.load_demand_kw,pv_panel_rating_dc)
        pv_panel_kWdc = np.where(power_per_pv >pv_panel_rating_dc ,pv_panel_rating_dc ,power_per_pv )
        #Step 1: convert DC-> AC with inverter
        pv_power.update({'Power/Panel (init-kWdc)':power_per_pv})
        
        pv_panel_kWac = self.dc2ac_inverter(pe_sizes['PV [DC/AC]'],pv_panel_kWdc)
        pv_power.update({'Power/Panel After Inverter [kWac]':pv_panel_kWac})
        # ['Inverter Loss [MW]','Inverter Loss [%]']
        inv_pwr = np.sum(pv_panel_kWac)
        inv_loss_kw = np.sum(pv_panel_kWac)-np.sum(pv_panel_kWdc)
        inv_loss_perc = 100*inv_loss_kw/np.sum(pv_panel_kWdc)
        inv_loss_kw=inv_loss_kw*n_pv_panels
        inv_pwr_tot = np.sum(pv_panel_kWac)*n_pv_panels

        pv_panel_kWac = self.ac2ac_transformer(pe_sizes['PV [AC/AC]'],pv_panel_kWac)
        pv_power.update({'Power/Panel After AC-Transformer [kWac]':pv_panel_kWac})
        trans_loss_kw = np.sum(pv_panel_kWac)-inv_pwr
        trans_loss_perc = 100*trans_loss_kw/inv_pwr
        trans_loss_kw = trans_loss_kw*n_pv_panels
        trans_pwr_tot = np.sum(pv_panel_kWac)*n_pv_panels

        pv_Vac_l2l= 480
        r_cable,pv_cableinfo=self.find_pv_cable(V_cable,pv_panel_rating_dc ,pv_Vac_l2l)
        pv_Icable=(pv_panel_kWac*1000)/(np.sqrt(3)*V_cable)
        n_cables=pv_cableinfo['n_cables']
        cable_loss_pv_panels = np.zeros((len(pv_panel_kWac),len(cable_lengths)))
        pv_power_endcable = np.zeros((len(pv_panel_kWac),len(cable_lengths)))
        pv_power_post_rect = np.zeros((len(pv_panel_kWac),len(cable_lengths)))
        for ci,l_cable in enumerate(cable_lengths):
            #TODO: fix cable losses in turbine to be like below.
            cable_loss_pv_panels[:,ci] = l_cable*r_cable*((pv_Icable/n_cables)**2)*n_cables
            pv_power_endcable[:,ci] = ((pv_panel_kWac*1000) - cable_loss_pv_panels[:,ci])/1000
            pv_power_post_rect[:,ci]= self.ac2dc_rectifier(pe_sizes['PV [DC/AC]'],pv_power_endcable[:,ci])
        cable_loss_kw = np.sum(pv_power_endcable) - (n_pv_panels*np.sum(pv_panel_kWac))
        cable_loss_perc = 100*cable_loss_kw/(n_pv_panels*np.sum(pv_panel_kWac))
        cable_power_tot = np.sum(pv_power_endcable) 

        pv_power.update({'Tot PV Power After Cables [kWac]':np.sum(pv_power_endcable,axis=1)})
        pv_4load = np.sum(pv_power_post_rect,axis=1)
        rect_loss_kw = np.sum(pv_power_post_rect)-np.sum(pv_power_endcable)
        rect_loss_perc = 100*rect_loss_kw/np.sum(pv_power_endcable)
        rect_power_tot = np.sum(pv_power_post_rect)

        pv_power.update({'Tot PV Power After Rectifier':pv_4load})
        load=self.load_demand_kw*np.ones(len(pv_panel_kWdc))
        wind_shortfall = [x - y for x, y in
                             zip(load,external_power_kw)]
        wind_shortfall = [x if x > 0 else 0 for x in wind_shortfall]
        solar_curtailment = [x - y for x, y in
                                zip(pv_4load ,wind_shortfall)]
        solar_curtailment=[x if x > 0 else 0 for x in solar_curtailment]
        solar_2_pem = pv_4load - np.array(solar_curtailment)
        gen_pwr_2_pem = solar_2_pem + external_power_kw
        wind_solar_shortfall = [x - y for x, y in
                             zip(load,gen_pwr_2_pem)]
        wind_solar_shortfall = [x if x > 0 else 0 for x in wind_solar_shortfall]
        battery_used, excess_energy, battery_SOC=self.run_dispatch(solar_curtailment,wind_solar_shortfall)
        bat_getting_charged = np.diff(battery_SOC)
        pv2bat= np.where(bat_getting_charged>0,bat_getting_charged,0)
        # power.append(np.sum(pv2bat))
        pv_power2pem=self.dc2dc_unidirect_transformer(pe_sizes['PV [DC/DC] Uni-direct']*n_pv_panels,solar_2_pem )
        pv2pem_transformer_loss_kw = np.sum(pv_power2pem) - np.sum(solar_2_pem)
        pv2pem_transformer_loss_perc = 100*pv2pem_transformer_loss_kw/np.sum(solar_2_pem)

        pv_power2bat=self.dc2dc_bidirect_converter(pe_sizes['Battery [DC/DC] Bi-direct']*n_pv_panels,pv2bat)
        pv2bat_bidirectconv_loss_kw = np.sum(pv_power2bat) - np.sum(pv2bat)
        pv2bat_bidirectconv_loss_perc = 100*pv2bat_bidirectconv_loss_kw/np.sum(pv2bat)

        battery_used_eff, excess_energy, battery_SOC_temp=self.run_dispatch(pv_power2bat,wind_solar_shortfall)
        bat2pem = np.array(battery_used_eff)
        bat_power2pem=self.dc2dc_bidirect_converter(pe_sizes['Battery [DC/DC] Bi-direct']*n_pv_panels,bat2pem)
        bat2pem_bidirectconv_loss_kw = np.sum(bat_power2pem) - np.sum(bat2pem)
        bat2pem_bidirectconv_loss_perc = 100*bat2pem_bidirectconv_loss_kw /np.sum(bat2pem)
        tot_loss_kw =  (np.sum(pv_power2pem) + np.sum(bat_power2pem))-np.sum(self.pv_output_kWdc)
        tot_loss_perc = 100*tot_loss_kw/np.sum(self.pv_output_kWdc)
        pwr_to_pem_tot = (np.sum(pv_power2pem) + np.sum(bat_power2pem))
        power_vals = [np.sum(self.pv_output_kWdc),inv_pwr_tot,trans_pwr_tot,cable_power_tot,rect_power_tot,np.sum(pv_power2pem),np.sum(pv_power2bat),np.sum(bat_power2pem),pwr_to_pem_tot]
        mw_losses = np.array([inv_loss_kw,trans_loss_kw,cable_loss_kw,rect_loss_kw,pv2pem_transformer_loss_kw,pv2bat_bidirectconv_loss_kw,bat2pem_bidirectconv_loss_kw,tot_loss_kw])/1000
        perc_losses = [inv_loss_perc,trans_loss_perc,cable_loss_perc,rect_loss_perc,pv2pem_transformer_loss_perc,pv2bat_bidirectconv_loss_perc,bat2pem_bidirectconv_loss_perc,tot_loss_perc]
        losses['MW']=pd.Series(dict(zip(loss_keys,mw_losses)),name='PV-Dist')
        losses['%']=pd.Series(dict(zip(loss_keys,perc_losses)),name='PV-Dist')
        power_along_path = pd.Series(dict(zip(powers_keys,power_vals)),name='Power [kW]')
        # pv_power.update({'Tot PV Power direct to PEM (init)':solar_2_pem })
        # pv_power.update({'Tot PV Power into Battery (init)':pv2bat})
        # pv_power.update({'Bat discharge to PEM (init)':bat2pem})
        # pv_power.update({'Tot PV Power direct to PEM':pv_power2pem})
        # pv_power.update({'Tot PV Power into Battery':pv_power2bat})
        # pv_power.update({'Bat discharge to PEM':bat_power2pem})
        ts_power = {'Wind':external_power_kw,'Solar':pv_power2pem,'Battery':bat_power2pem}
        return ts_power,losses,power_along_path


    # def clean_up_outputs(self,ts_data_dict,n_panels):
    #     keys,vals = zip(*ts_data_dict.items())
    #     keys_copy = keys
    #     pdiff={}
    #     mw_diff={}
    #     found_keys=[]
    #     if n_panels !=0:
    #         for i,k in enumerate(keys):
    #             if '/Panel' in k:
    #                 newkey = k.replace('Power/Panel','Tot')
    #                 ts_data_dict.update({newkey:vals[i]*n_panels})
    #             if '(init)' in k:
    #                 found_keys.append(k)
    #                 keys_copy.remove(k)
    #                 oldkey = k.replace(' (init)','')
    #                 found_keys.append(oldkey)
    #                 keys_copy.remove(oldkey)
    #                 kW_diff=np.sum(ts_data_dict[oldkey]) - np.sum(ts_data_dict[k])
    #                 mw_diff.update({oldkey:kW_diff/1000})
    #                 percdiff=100*kW_diff/np.sum(ts_data_dict[k])
    #                 pdiff.update({oldkey:percdiff})
            






        #Step 2: step-up to cable voltage
        #Step 3: cable power at load
        #Step 4: Rectifier to battery
        # Step 5: Rectifier to PEM
        #inverter, step-up transformer, cable, rectifier to bat, rectifier to pem


    def find_pv_cable(self,cable_voltage,power_to_cable_kWac,pv_Vacl2l):
        kiloft_to_km = 0.3048
        nmax_cables=10
        cable_resistance_per_kft = np.array([0.12,0.25,0.02,0.01,0.009])
        cbl_names = np.array(["AWG 1/0","AWG 4/0","MCM 500","MCM 1000","MCM 1250"])
        cable_resistance_per_m = (cable_resistance_per_kft *(1/kiloft_to_km))/(1e3)
        cable_ampacity = np.array([150,230,320,455,495])
        cable_cost_per_m = np.array([61115.1602528554,72334.3683802817,96358.26769213431,104330.7086713996,115964.28690974298])/1000
        ac2ac_turns = cable_voltage/pv_Vacl2l
        pv_Icable=power_to_cable_kWac*1000/(np.sqrt(3)*cable_voltage)
        cable_required_power=power_to_cable_kWac*1000
        n_cables = np.ceil(pv_Icable/cable_ampacity)
        p_line_max=np.sqrt(3)*cable_ampacity*n_cables*cable_voltage
        cb_idx = np.argwhere((p_line_max >= cable_required_power) & (n_cables<=nmax_cables))
        cb_idx = cb_idx.reshape(len(cb_idx))

        i_per_cable = pv_Icable/n_cables[cb_idx]
        i_rated_cable = cable_ampacity[cb_idx]
        cable_r = cable_resistance_per_m[cb_idx]
        n_cables_okay = n_cables[cb_idx]
        names = cbl_names[cb_idx]
        p_loss_per_cable = (i_per_cable**2)*cable_r
        p_loss_tot_per_m = p_loss_per_cable*n_cables[cb_idx]
        idx_lowestloss = np.argmin(p_loss_tot_per_m)

        r_cable = cable_r[idx_lowestloss]
        num_cables = n_cables_okay[idx_lowestloss]
        i_cable = i_rated_cable[idx_lowestloss]
        cable_type = names[idx_lowestloss]
        cost = cable_cost_per_m[cb_idx]
        cost = cost[idx_lowestloss]
        cable_info = {'V_cable':cable_voltage,'Cable Ampacity':i_cable,'Cable Name':cable_type,'n_cables':num_cables,
        'Cable Resistance':r_cable,'Transformer Turns':ac2ac_turns, 'Cost/m':cost,
        'PV Cable Current':pv_Icable,'PV Power [kW]':power_to_cable_kWac}
        
        return r_cable, cable_info


    def ac2dc_rectifier(self,ac2dcrect_size_kw,input_power_kWac):
        x_load_percent = np.linspace(0.0,1.0,11)
        ac2dc_rectification_eff=np.array([96,96.54, 98.12, 98.24, 98.6, 98.33, 98.03, 97.91, 97.43, 97.04, 96.687])/100
        dc2dc_rectification_eff=np.array([91,91.46, 95.16, 96.54, 97.13, 97.43, 97.61,97.61,97.73,97.67,97.61])/100
        rect_eff = ac2dc_rectification_eff*dc2dc_rectification_eff
        f_ac2dc=interpolate.interp1d(x_load_percent,rect_eff)
        power_load_percent = input_power_kWac/ac2dcrect_size_kw
        output_power_kWdc = input_power_kWac*(f_ac2dc(power_load_percent))
        return output_power_kWdc

    def ac2ac_transformer(self,ac2ac_transformer_rating_kw,input_power_kWac):
        x_load_percent = np.linspace(0.0,1.0,11)
        ac2ac_transformer_eff=np.array([90,90.63, 93.91, 95.63, 96.56, 97.19, 97.50, 97.66, 97.66, 97.66, 97.50])/100
        f_ac2ac=interpolate.interp1d(x_load_percent,ac2ac_transformer_eff)
        power_load_percent = input_power_kWac/ac2ac_transformer_rating_kw
        output_power_kWac = input_power_kWac*(f_ac2ac(power_load_percent))
        return output_power_kWac

    def run_dispatch(self,curtailed_power,shortfall):
        bat_model = SimpleDispatch()
        bat_model.Nt = len(shortfall)
        if len(curtailed_power)<bat_model.Nt:
            curtailed_power=np.append(np.array([0]),curtailed_power)
        bat_model.curtailment = curtailed_power
        bat_model.shortfall = shortfall
        bat_model.charge_rate=self.bat_chargerate_kw
        bat_model.discharge_rate=self.bat_chargerate_kw
        bat_model.battery_storage=self.bat_capacity_kwh
        battery_used, excess_energy, battery_SOC = bat_model.run()
        return battery_used,excess_energy,battery_SOC

    
    def set_up_pv(self,hybrid_plant):
        #cited by ATB [2021]: https://www.nrel.gov/docs/fy22osti/80694.pdf
        #Table 3 has utility-scale PV info (100MW PV)
        #Figure 11: ($0.04/Wdc cost for an inverter and 0.33$/Wdc for solar module)
        #dc-coupled with single bidirect (Table 10 page 39)
        #$0.89/Wdc or $1.14/Wac Table ES-2
        # 
        pv_ref_size_kw = 100*1000 #[kWdc] (100MW)
        pv_ref_volt = 1500 #[Vdc]
        dc_2_ac_ratio_fixed = 1.31
        # self.pv_Vdc_mpp = 240
        V_inv_output_l2l = 480 #480/277 Wy
        I_inv_output = 64 #[A]
        S_inv_output = np.sqrt(3)*V_inv_output_l2l*I_inv_output
        dc_2_ac_ratio_1axis = 1.28 #[DC-AC Inverter Ratio from Table 3]
        #hybrid_plant.pv._system_model.SystemDesign.dc_ac_ratio
        self.dc2ac_loading_ratio = 1.3 #used for AC and DC coupled systems
        self.inv2bat_storage_ratio = 1.67 #only used for AC coupled-systems
        self.pv_output_kWdc=np.array(hybrid_plant.pv._system_model.Outputs.dc)/1000
        self.pv_rating_kWdc = hybrid_plant.pv._system_model.SystemDesign.system_capacity
        self.pv_default_losses = hybrid_plant.pv._system_model.SystemDesign.losses
    def size_dc_components(self,resize_bat,external_power_kw,rated_load_kw,pv_rating_kwDc):
        dc_pe={}
        #from NREL paper, inverter is 60% of PV rating
        # dc2ac_inverter_rating_kWac = (0.6)*self.pv_rating_kWdc
        #https://www.nrel.gov/docs/fy17osti/68737.pdf
        #need bidirect DC invert for DC connected
        #need bidirect + PV inverter for AC connected
        dc2ac_pv_inverter_rating_kWac = (1/self.dc2ac_loading_ratio)*pv_rating_kwDc
        # pv2bat_dc2dc_unidirect_rating_kWac = []
        ac2ac_pv_transformer_rating_kWac = 1.2*dc2ac_pv_inverter_rating_kWac
        pv2pem_dc2dc_unidirect_rating_kWac = pv_rating_kwDc#self.pv_rating_kWdc
        # bat2pem_dc2dc
        if resize_bat:
            raw_pv_wind = self.pv_output_kWdc + external_power_kw
            load_dmd = rated_load_kw*np.ones(len(raw_pv_wind))
            gen_load_diff = raw_pv_wind - load_dmd
            curt_gen = np.where(gen_load_diff>0,gen_load_diff,0)
            max_bat_pwr=np.max(curt_gen)
            bidirect4bat_rating_kw = np.max([max_bat_pwr,self.bat_chargerate_kw])
        else:
            bidirect4bat_rating_kw =self.bat_chargerate_kw

        dc_pe = {'PV [DC/AC]': dc2ac_pv_inverter_rating_kWac,'Battery [DC/DC] Bi-direct':bidirect4bat_rating_kw,
        'PV [DC/DC] Uni-direct':pv2pem_dc2dc_unidirect_rating_kWac,'PV [AC/AC]':ac2ac_pv_transformer_rating_kWac}
        return dc_pe

    def set_up_battery(self,hybrid_plant):
        #ATB: https://atb.nrel.gov/electricity/2022/utility-scale_battery_storage
        #cited by ATB [2021]: https://www.nrel.gov/docs/fy22osti/80694.pdf
        #Table 8 has utility-scale lithium-ion battery info
        #inverter loading ratio of 1.3 and inverter/storage size ratio of 1.67
        #b_size is 60MWdc to match 100MW PV system
        #24 inverters rated at 2.4MW
        #footnote 16 on page 34 is good!
        #Figure 23 has layout configurations
        #Figure 17 says uses transformer to step-up 480 V inverter output to 12-66kV
    
        self.bat_chargerate_kw=hybrid_plant.battery.system_capacity_kw
        self.bat_capacity_kwh=hybrid_plant.battery.system_capacity_kwh

    def dc2dc_unidirect_transformer(self,dc2dc_transformer_rating_kw,input_power_kw):
        #unidirectional dc converter step-up (buck-boost)
        #https://ieeexplore.ieee.org/document/9767750
        #https://www.sciencedirect.com/science/article/pii/S0306261916312879
        x_load_percent=np.array([0,10.1, 13.2,  17.1,  20. ,  25.1,  29.9,  35. ,  40.1,  50.1,  60. , 69.9,  79.9,  90.1, 100. ])
        dc2dc_transformer_eff=np.array([80,80.9,84.1, 88.1, 90. , 93.2, 95. , 96. , 96.3, 96.2, 95.9, 95.4, 95.3, 95.3, 95.2])/100
        f_dc2dc=interpolate.interp1d(x_load_percent,dc2dc_transformer_eff)
        power_load_percent = input_power_kw/dc2dc_transformer_rating_kw
        output_power_kWdc = input_power_kw*(f_dc2dc(power_load_percent))
        return output_power_kWdc
    def dc2dc_bidirect_converter(self,dc2dc_bidirect_rating_kw,input_power_kw):
        #https://www.sciencedirect.com/science/article/pii/S0306261916312879
        x_load_percent = np.linspace(0.0,1.0,11)
        dc2dc_bidirectconverter_eff=np.array([91,91.44, 95.19, 96.56, 97.18, 97.50, 97.62, 97.68, 97.77, 97.68, 97.68])/100
        f_dc2dc_biC=interpolate.interp1d(x_load_percent,dc2dc_bidirectconverter_eff)
        power_load_percent = input_power_kw/dc2dc_bidirect_rating_kw
        output_power_kWdc = input_power_kw*(f_dc2dc_biC(power_load_percent))
        # self.pv_output_dc
        return output_power_kWdc
    def dc2ac_inverter(self,dc2ac_inverter_rating_kw,input_power_kw):
        dc2ac_ratio = 1.3 #output capacity of PV compared to processing capacity of inverter
        x_load_percent = np.linspace(0.0,1.0,11)
        ac2dc_generation_eff=np.array([96,96.53, 98.00, 98.12, 98.29, 98.03, 97.91, 97.74, 97.15, 96.97, 96.48])/100
        dc2dc_generation_eff=np.array([90,91.44, 95.19, 96.56, 97.18, 97.50, 97.62, 97.68, 97.77, 97.68, 97.68])/100
        inverter_eff = ac2dc_generation_eff*dc2dc_generation_eff
        f_dc2ac=interpolate.interp1d(x_load_percent,inverter_eff)
        saturated_input_power=np.where(input_power_kw>dc2ac_inverter_rating_kw,dc2ac_inverter_rating_kw,input_power_kw)
        power_load_percent = saturated_input_power/dc2ac_inverter_rating_kw
        output_power_kWac = saturated_input_power*(f_dc2ac(power_load_percent))
        return output_power_kWac
class battery_power_electronics:
    def __init__(self,hybrid_plant,battery):
        self.bat_chargerate_kw=hybrid_plant.battery.system_capacity_kw
        self.bat_capacity_kw=hybrid_plant.battery.system_capacity_kw
        hybrid_plant.battery.system_capacity_voltage
        hybrid_plant.battery.system_voltage_volts_kw
        hybrid_plant.battery._system_model.StatePack.I_chargeable
        hybrid_plant.battery._system_model.StatePack.I_dischargeable
        hybrid_plant.battery._system_model.StatePack.P_chargeable
        hybrid_plant.battery._system_model.StatePack.V
        hybrid_plant.battery._system_model.ParamsCell.Vcut #Vnom, Vexp,Vfull,resistance
        hybrid_plant.battery._system_model.ParamsPack.nominal_voltage
        #need curtailment & shortfall
#ref_capac_kw = 1000 #[kW]
        # ref_dimension_sqft = 72000 #[ft^2]
        # ref_v_mpp = 240 #[V]
        # ref_i_mpp = 1152 #[Amp]
        # ref_pAc_mpp = 898564 #[Wac]
        # ref_pDc_mpp = 1000350 #[Wac]
        # ref_cost = 1254000 #[$]
    # def min_deg_cntrl(self):
    #     #run as few as possible
    #     []
    # def create_clusters(self):
    #     start=time.perf_counter()
    #     stacks=[]
    #     # TODO fix the power input - don't make it required!
    #     # in_dict={'dt':3600}
    #     for i in range(self.num_clusters):
    #         stacks.append(PEMClusters(cluster_size_mw = self.cluster_cap_mw))
    #     end=time.perf_counter()
    #     print('Took {} sec to run the create clusters'.format(round(end-start,3)))
    #     return stacks
    # def create_filler_cluster(self,cluster_size_mw):
    #     # start=time.perf_counter()
    #     stacks=[]
    #     # TODO fix the power input - don't make it required!
    #     # in_dict={'dt':3600}
        
    #     stacks.append(PEMClusters(cluster_size_mw=cluster_size_mw))
    #     # end=time.perf_counter()
    #     # print('Took {} sec to run the create clusters'.format(round(end-start,3)))
    #     return stacks


# if __name__=="__main__":

#     system_size_mw = 1000
#     num_clusters = 20
#     cluster_cap_mw = system_size_mw/num_clusters
#     stack_rating_kw = 1000
#     cluster_min_power_kw = 0.1*stack_rating_kw*cluster_cap_mw
#     num_steps = 200
#     power_rampup = np.arange(cluster_min_power_kw,system_size_mw*stack_rating_kw,cluster_min_power_kw)
    
#     # power_rampup = np.linspace(cluster_min_power_kw,system_size_mw*1000,num_steps)
#     power_rampdown = np.flip(power_rampup)
#     power_in = np.concatenate((power_rampup,power_rampdown))
#     pem=run_PEM_clusters(power_in,system_size_mw,num_clusters)

#     h2_ts,h2_tot = pem.run()
#     []
    # def run(self):
        
    #     clusters = self.create_clusters()
    #     power_to_clusters = self.even_split_power()
    #     h2_df_ts = pd.DataFrame()
    #     h2_df_tot = pd.DataFrame()
    #     # h2_dict_ts={}
    #     # h2_dict_tot={}
        
    #     col_names = []
    #     start=time.perf_counter()
    #     for ci,cluster in enumerate(clusters):
    #         cl_name = 'Cluster #{}'.format(ci)
    #         col_names.append(cl_name)
    #         h2_ts,h2_tot = clusters[ci].run(power_to_clusters[ci])
    #         # h2_dict_ts['Cluster #{}'.format(ci)] = h2_ts
            
    #         h2_ts_temp = pd.Series(h2_ts,name = cl_name)
    #         h2_tot_temp = pd.Series(h2_tot,name = cl_name)
    #         if len(h2_df_tot) ==0:
    #             # h2_df_ts=pd.concat([h2_df_ts,h2_ts_temp],axis=0,ignore_index=False)
    #             h2_df_tot=pd.concat([h2_df_tot,h2_tot_temp],axis=0,ignore_index=False)
    #             h2_df_tot.columns = col_names

    #             h2_df_ts=pd.concat([h2_df_ts,h2_ts_temp],axis=0,ignore_index=False)
    #             h2_df_ts.columns = col_names
    #         else:
    #             # h2_df_ts = h2_df_ts.join(h2_ts_temp)
    #             h2_df_tot = h2_df_tot.join(h2_tot_temp)
    #             h2_df_tot.columns = col_names

    #             h2_df_ts = h2_df_ts.join(h2_ts_temp)
    #             h2_df_ts.columns = col_names

    #     end=time.perf_counter()
    #     print('Took {} sec to run the RUN function'.format(round(end-start,3)))
    #     return h2_df_ts, h2_df_tot
    #     # return h2_dict_ts, h2_df_tot

    # def even_split_power(self):
    #     start=time.perf_counter()
    #     #determine how much power to give each cluster
    #     num_clusters_on = np.floor(self.input_power_kw/self.cluster_min_power)
    #     num_clusters_on = np.where(num_clusters_on > self.num_clusters, self.num_clusters,num_clusters_on)
    #     power_per_cluster = [self.input_power_kw[ti]/num_clusters_on[ti] if num_clusters_on[ti] > 0 else 0 for ti, pwr in enumerate(self.input_power_kw)]
        
    #     power_per_to_active_clusters = np.array(power_per_cluster)
    #     power_to_clusters = np.zeros((len(self.input_power_kw),self.num_clusters))
    #     for i,cluster_power in enumerate(power_per_to_active_clusters):#np.arange(0,self.n_stacks,1):
    #         clusters_off = self.num_clusters - int(num_clusters_on[i])
    #         no_power = np.zeros(clusters_off)
    #         with_power = cluster_power * np.ones(int(num_clusters_on[i]))
    #         tot_power = np.concatenate((with_power,no_power))
    #         power_to_clusters[i] = tot_power

    #     # power_to_clusters = np.repeat([power_per_cluster],self.num_clusters,axis=0)
    #     end=time.perf_counter()
    #     print('Took {} sec to run basic_split_power function'.format(round(end-start,3)))
    #     #rows are power, columns are stacks [300 x n_stacks]
