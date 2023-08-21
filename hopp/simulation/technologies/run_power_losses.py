import os
import sys
sys.path.append('')
# from dotenv import load_dotenv
import pandas as pd
# from PEM_H2_LT_electrolyzer_ESGBasicClusters import PEM_electrolyzer_LT as PEMClusters
from hopp.simulation.technologies.hydrogen.electrolysis.PEM_H2_LT_electrolyzer_Clusters import PEM_H2_Clusters as PEMClusters
from hopp.add_custom_modules.custom_wind_floris import Floris
from hopp.simulation.technologies.calc_power_losses import turbine_power_electronics
from hopp.simulation.technologies.calc_power_losses import dc_component_power_electronics
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

class run_power_electronics:
    def __init__(self,hybrid_plant,electrolysis_scale,system_config,floris,electrolyzer_size=1000):
        if electrolysis_scale == 'Centralized':
            self.run_centralized_wind
        elif electrolysis_scale=='Distributed':
            print('distributed not included right now')
            # self.run_distributed_wind
        if not floris:
            hybrid_plant,nTurbs = self.sloppy_floris_workaround()
        else:
            nTurbs=hybrid_plant.wind._system_model.nTurbs
        if system_config =='Compare':
            loss_info,generation_info=self.compare_worst_2_best(hybrid_plant,nTurbs,floris,electrolyzer_size=1000)
            self.losses_dict = loss_info
            self.generation_data = generation_info
        # if system_config == 'Worst Case':
        #     self.run_worst_hybrid_plant(hybrid_plant,nTurbs,floris,electrolyzer_size=1000)
        # elif system_config =='Best Case':
        #     self.run_best_hybrid_plant(hybrid_plant,nTurbs,floris,electrolyzer_size=1000)
    def compare_worst_2_best(self,hybrid_plant,nTurbs,floris,electrolyzer_size=1000):
        turb_power_to_pem,n_clusters=self.run_centralized_wind(hybrid_plant,nTurbs,floris)
        dc_component_info={}
        dc_component_info['Rated Load [kW]']=electrolyzer_size*1000
        # if hybrid_plant.battery.system_capacity_kw>0:
        dc_component_info['Flexible Battery Charge Rate']=True
        # else:
        #     dc_component_info['Flexible Battery Charge Rate']
        
        dc_pe=dc_component_power_electronics(hybrid_plant,dc_component_info,turb_power_to_pem)
        best_power_info, best_losses, best_ts_power=dc_pe.run_colocated_dc_bus(hybrid_plant,turb_power_to_pem)
        cbl_l,cbl_cnt=np.unique(self.turb_components['Cables']['Cable Lengths'],return_counts=True)
        n_pv_panels = len(cbl_l)
        cable_lengths=cbl_l#self.turb_components['Cables']['Cable Lengths'][0:n_pv_panels]
        worst_ts_power,worst_losses,worst_power_info=dc_pe.run_distributed_pv_central_bat(hybrid_plant,turb_power_to_pem,n_pv_panels,self.turb_components['Cables']['V_cable'],cable_lengths)
        self.n_clusters = n_clusters
        # pd.concat([worst_losses['%'],best_losses['%']],axis=1)
        # pd.concat([worst_power_info,best_power_info],axis=1)
        generation_data={}
        loss_data={}
        loss_data['PV MW Losses'] =pd.concat([worst_losses['MW'],best_losses['MW']],axis=1)
        loss_data['PV Percent Losses']=pd.concat([worst_losses['%'],best_losses['%']],axis=1)
        loss_data['PV Power Per Step']=pd.concat([worst_power_info,best_power_info],axis=1)
        generation_data['Central PV']=best_ts_power
        generation_data['Distributed PV']=pd.DataFrame(worst_ts_power)
        return loss_data,generation_data
        []

    def run_best_hybrid_plant(self,hybrid_plant,nTurbs,floris,electrolyzer_size=1000):
        turb_power_to_pem,n_clusters=self.run_centralized_wind(hybrid_plant,nTurbs,floris)
        dc_component_info={}
        # dc_component_info['Co-located Components'] = ['Battery','PV']
        dc_component_info['Co-located Component Loc'] = 'Centralized'
        dc_component_info['Load Loc'] = 'Centralized'
        # dc_component_info['Co-located Component Num Clusters'] = [n_clusters,n_clusters]
        # dc_component_info['Minimum Load Requirement [kW] per Cluster'] = 0.1*(electrolyzer_size/n_clusters)*1000
        # dc_component_info['Maximum Load Requirement [kW] per Cluster'] =(electrolyzer_size/n_clusters)*1000
        # dc_component_info['Flexible Battery Charge Rate'] = True
        # dc_component_info['DC Component Distance to Load'] = [0,0]
        dc_component_info['Rated Load [kW]']=electrolyzer_size*1000
        dc_pe=dc_component_power_electronics(hybrid_plant,dc_component_info,turb_power_to_pem)
        tot_power_to_pem, power_info, ts_power=dc_pe.run_colocated_dc_bus(hybrid_plant,turb_power_to_pem)

    def run_worst_hybrid_plant(self,hybrid_plant,nTurbs,floris,electrolyzer_size=1000):
        []
        self.turb_components['Cables']['V_cable']
        self.turb_components['Cables']['Cable Ampacity']
        self.turb_components['Cables']['n_cables']
        n_pv_panels = len(self.turb_components['Cables']['Cable Lengths'])
        pv_ac_line_length = []
        #run_distributed_pv_central_bat

    def sloppy_floris_workaround(self,hybrid_plant):
        if hybrid_plant.wind.rotor_diameter == 185:
            hybrid_plant.wind.turb_rating = 4000
            nTurbs = 255
        elif hybrid_plant.wind.rotor_diameter == 255:
            hybrid_plant.wind.turb_rating = 8000
            nTurbs = 126
        elif hybrid_plant.wind.rotor_diameter == 196:
            hybrid_plant.wind.turb_rating = 6000
            nTurbs = 168
        hybrid_plant.wind._system_model.nTurbs = nTurbs
        return hybrid_plant,nTurbs


    def run_centralized_wind(self,hybrid_plant,nTurbs,floris,electrolyzer_size=1000):
        # from hybrid.Electrolyzer_Models.run_h2_clusters import run_PEM_clusters
        power_elec=turbine_power_electronics(nTurbs)
        
        if nTurbs ==255:
            nturbs_per_cable = 15
            nturbs_dist2_load = [8,9]
            n_distances = 1

        elif nTurbs == 126:
            nturbs_per_cable = 6
            nturbs_dist2_load = [7]
            n_distances = 3

        elif nTurbs == 168:
            nturbs_per_cable = 6
            nturbs_dist2_load = [7]
            n_distances = 4
        if floris:
            power_2_pem,n_clusters=power_elec.run_power_elec_floris(nturbs_per_cable,nturbs_dist2_load,n_distances,hybrid_plant.wind)
        else:
            electrical_generation_timeseries = np.array(hybrid_plant.wind.generation_profile)
            power_2_pem,n_clusters=power_elec.run_power_elec_pysam(electrical_generation_timeseries,nturbs_per_cable,nturbs_dist2_load,n_distances,hybrid_plant.wind)
        power_data = power_elec.power_data
        loss_data = power_elec.loss_data
        self.wind_loss_data = loss_data
        self.wind_power_data = power_data
        time_series_data = power_elec.ts_data
        power_components = power_elec.component_list
        self.turb_components = power_components
        #TODO REMOVE BELOW
        # pem= run_PEM_clusters(power_2_pem,electrolyzer_size,n_clusters)
        # h2_ts,h2_tot = pem.run()
        return power_2_pem,n_clusters
            