import os
import sys
sys.path.append('')
# from dotenv import load_dotenv
import pandas as pd
# from PEM_H2_LT_electrolyzer_ESGBasicClusters import PEM_electrolyzer_LT as PEMClusters
from hybrid.Electrolyzer_Models.PEM_H2_LT_electrolyzer_Clusters import PEM_H2_Clusters as PEMClusters
from hybrid.add_custom_modules.custom_wind_floris import Floris
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
class run_PEM_power_electronics:
    '''Add description and stuff :)'''
    # def __init__(self,electrical_power_signal,nturbs_per_cable,nturbs_dist2_load,n_distances,wind_plant,n_turbs):
    def __init__(self,n_turbs):

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
        # n_cables = cable_info['n_cables']
        l_cable = cable_info['Cable Lengths']
        V_cable = cable_info['V_cable']
        d_turb = cable_info['Distance Between Turbs']
        T1_rated_power_kw = 1.2*P_rated_turb

        # power = self.get_floris_power_and_saturate(wind_plant,P_rated_turb)
        power = self.split_pysam_power_and_adjust(electrical_generation_timeseries,P_rated_turb)
        power = self.do_transformer_losses(power,T1_rated_power_kw,l_cable,self.n_turbs,Vline_turb,V_cable)
        power = self.do_cable_losses(power,l_cable,R_cable,d_turb,V_cable,nturbs_percable)
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
        # n_cables = cable_info['n_cables']
        l_cable = cable_info['Cable Lengths']
        V_cable = cable_info['V_cable']
        d_turb = cable_info['Distance Between Turbs']
        T1_rated_power_kw = 1.2*P_rated_turb
        
        
        power = self.get_floris_power_and_saturate(wind_plant,P_rated_turb)
        power = self.do_transformer_losses(power,T1_rated_power_kw,l_cable,n_turbs,Vline_turb,V_cable)
        power = self.do_cable_losses(power,l_cable,R_cable,d_turb,V_cable,nturbs_percable)
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

    def do_cable_losses(self,turbpower_to_cable,l_cable,R_cable,d_turb,V_cable,nturbs_percable):
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

            p_loss_segment = np.sqrt(3)*(cumulative_i**2)*R_cable*d_turb
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

