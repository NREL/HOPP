import os
import sys
sys.path.append('')
# from dotenv import load_dotenv
import pandas as pd
# from PEM_H2_LT_electrolyzer_ESGBasicClusters import PEM_electrolyzer_LT as PEMClusters
from greenheart.simulation.technologies.hydrogen.electrolysis.PEM_H2_LT_electrolyzer_Clusters import PEM_H2_Clusters as PEMClusters
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
#from PyOMO import ipOpt !! FOR SANJANA!!
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
class run_PEM_clusters:
    '''Add description and stuff :)'''
    def __init__(self,electrical_power_signal,system_size_mw,num_clusters,verbose=True):

        self.cluster_cap_mw = np.round(system_size_mw/num_clusters)
        self.num_clusters = num_clusters

        self.stack_rating_kw = 1000
        self.stack_min_power_kw=0.1*self.stack_rating_kw
        #self.num_available_pem=interconnection_size_mw
        self.input_power_kw=electrical_power_signal
        self.cluster_min_power = self.stack_min_power_kw * self.cluster_cap_mw
        self.cluster_max_power = self.stack_rating_kw * self.cluster_cap_mw

        self.verbose = verbose

    def run(self):
        
        clusters = self.create_clusters()
        power_to_clusters = self.even_split_power()
        h2_df_ts = pd.DataFrame()
        h2_df_tot = pd.DataFrame()
        # h2_dict_ts={}
        # h2_dict_tot={}
        
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
            print('Took {} sec to run the RUN function'.format(round(end-start,3)))
        return h2_df_ts, h2_df_tot
        # return h2_dict_ts, h2_df_tot
    def optimize_power_split(self):
        #Inputs: power signal, number of stacks, cost of switching (assumed constant)
        #install PyOMO
        #!!! Insert Sanjana's Code !!!
        #
        power_per_stack = []
        return power_per_stack #size
    def even_split_power(self):
        start=time.perf_counter()
        #determine how much power to give each cluster
        num_clusters_on = np.floor(self.input_power_kw/self.cluster_min_power)
        num_clusters_on = np.where(num_clusters_on > self.num_clusters, self.num_clusters,num_clusters_on)
        power_per_cluster = [self.input_power_kw[ti]/num_clusters_on[ti] if num_clusters_on[ti] > 0 else 0 for ti, pwr in enumerate(self.input_power_kw)]
        
        power_per_to_active_clusters = np.array(power_per_cluster)
        power_to_clusters = np.zeros((len(self.input_power_kw),self.num_clusters))
        for i,cluster_power in enumerate(power_per_to_active_clusters):#np.arange(0,self.n_stacks,1):
            clusters_off = self.num_clusters - int(num_clusters_on[i])
            no_power = np.zeros(clusters_off)
            with_power = cluster_power * np.ones(int(num_clusters_on[i]))
            tot_power = np.concatenate((with_power,no_power))
            power_to_clusters[i] = tot_power

        # power_to_clusters = np.repeat([power_per_cluster],self.num_clusters,axis=0)
        end=time.perf_counter()
        if self.verbose:
            print('Took {} sec to run basic_split_power function'.format(round(end-start,3)))
        #rows are power, columns are stacks [300 x n_stacks]


        return np.transpose(power_to_clusters)
    def run_distributed_layout_power(self,wind_plant):
        #need floris configuration!
        x_load_percent = np.linspace(0.1,1.0,10)
        
        #ac2ac_transformer_eff=np.array([90.63, 93.91, 95.63, 96.56, 97.19, 97.50, 97.66, 97.66, 97.66, 97.50])
        ac2dc_rectification_eff=np.array([96.54, 98.12, 98.24, 98.6, 98.33, 98.03, 97.91, 97.43, 97.04, 96.687])/100
        dc2dc_rectification_eff=np.array([91.46, 95.16, 96.54, 97.13, 97.43, 97.61,97.61,97.73,97.67,97.61])/100
        rect_eff = ac2dc_rectification_eff*dc2dc_rectification_eff
        f=interpolate.interp1d(x_load_percent,rect_eff)
        start_idx = 0
        end_idx = 8760
        nTurbs = self.num_clusters
        power_turbines = np.zeros((nTurbs, 8760))
        power_to_clusters = np.zeros((8760,self.num_clusters))
        ac2dc_rated_power_kw = wind_plant.turb_rating
        
        power_turbines[:, start_idx:end_idx] = wind_plant._system_model.fi.get_turbine_powers().reshape((nTurbs, end_idx - start_idx))/1000
        power_to_clusters = (power_turbines)*(f(power_turbines/ac2dc_rated_power_kw))
        
        # power_farm *((100 - 12.83)/100) / 1000

        clusters = self.create_clusters()
       
        h2_df_ts = pd.DataFrame()
        h2_df_tot = pd.DataFrame()
        # h2_dict_ts={}
        # h2_dict_tot={}
        
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
        return h2_df_ts, h2_df_tot
        []
    
    def max_h2_cntrl(self):
        #run as many at lower power as possible
        []
    def min_deg_cntrl(self):
        #run as few as possible
        []
    def create_clusters(self):
        start=time.perf_counter()
        stacks=[]
        # TODO fix the power input - don't make it required!
        # in_dict={'dt':3600}
        for i in range(self.num_clusters):
            stacks.append(PEMClusters(cluster_size_mw = self.cluster_cap_mw))
        end=time.perf_counter()
        if self.verbose:
            print('Took {} sec to run the create clusters'.format(round(end-start,3)))
        return stacks


if __name__=="__main__":

    system_size_mw = 1000
    num_clusters = 20
    cluster_cap_mw = system_size_mw/num_clusters
    stack_rating_kw = 1000
    cluster_min_power_kw = 0.1*stack_rating_kw*cluster_cap_mw
    num_steps = 200
    power_rampup = np.arange(cluster_min_power_kw,system_size_mw*stack_rating_kw,cluster_min_power_kw)
    
    # power_rampup = np.linspace(cluster_min_power_kw,system_size_mw*1000,num_steps)
    power_rampdown = np.flip(power_rampup)
    power_in = np.concatenate((power_rampup,power_rampdown))
    pem=run_PEM_clusters(power_in,system_size_mw,num_clusters)

    h2_ts,h2_tot = pem.run()
    []