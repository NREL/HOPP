import os
import sys

sys.path.append("")
# from dotenv import load_dotenv
import pandas as pd

from greenheart.simulation.technologies.hydrogen.electrolysis.PEM_H2_LT_electrolyzer_Clusters import PEM_H2_Clusters as PEMClusters
# from PEM_H2_LT_electrolyzer_Clusters import (
#     PEM_H2_Clusters as PEMClusters,
# )

import numpy as np
from numpy import savetxt  # ESG
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import warnings
import math
import scipy
import time
from scipy import interpolate
from pyomo.environ import *
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from greenheart.simulation.technologies.hydrogen.electrolysis.optimization_utils_linear import optimize
import time

# from PyOMO import ipOpt !! FOR SANJANA!!
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
# ---------------------------
#
class run_PEM_clusters:
    """Inputs:
    `electrical_power_signal`: plant power signal in kWh
    `system_size_mw`: total installed electrolyzer capacity (for green steel this is 1000 MW)
    `num_clusters`: number of PEM clusters that can be run independently
    ->ESG note: I have been using num_clusters = 8 for centralized cases
    Nomenclature:
    `cluster`: cluster is built up of 1MW stacks
    `stack`: must be 1MW (because of current PEM model)
    """

    def __init__(
        self,
        electrical_power_signal,
        system_size_mw,
        num_clusters,
        electrolyzer_direct_cost_kw,
        useful_life,
        user_defined_electrolyzer_params,
        degradation_penalty,
        turndown_ratio,
        verbose=True
    ):
        # nomen
        self.cluster_cap_mw = np.round(system_size_mw / num_clusters)
        # capacity of each cluster, must be a multiple of 1 MW
        self.num_clusters = num_clusters
        self.user_params = (
            user_defined_electrolyzer_params["Modify EOL Degradation Value"],
            user_defined_electrolyzer_params["EOL Rated Efficiency Drop"],
            user_defined_electrolyzer_params["Modify BOL Eff"],
            user_defined_electrolyzer_params["BOL Eff [kWh/kg-H2]"],
        )
        self.plant_life_yrs = useful_life
        self.use_deg_penalty = degradation_penalty
        self.turndown_ratio = turndown_ratio
        # Do not modify stack_rating_kw or stack_min_power_kw
        # these represent the hard-coded and unmodifiable
        # PEM model basecode
        self.stack_rating_kw = 1000  # single stack rating - DO NOT CHANGE
        self.stack_min_power_kw = turndown_ratio * self.stack_rating_kw
        # self.stack_min_power_kw = 0.1 * self.stack_rating_kw
        self.input_power_kw = electrical_power_signal
        self.cluster_min_power = self.stack_min_power_kw * self.cluster_cap_mw
        self.cluster_max_power = self.stack_rating_kw * self.cluster_cap_mw

        # For the optimization problem:
        self.T = len(self.input_power_kw)
        self.farm_power = 1e9
        self.switching_cost = (electrolyzer_direct_cost_kw*0.15*self.cluster_cap_mw * 1000)*(1.48e-4)/(0.26586)
        self.verbose=verbose

    def run_grid_connected_pem(self,system_size_mw,hydrogen_production_capacity_required_kgphr):
        pem=PEMClusters(
                    system_size_mw,
                    self.plant_life_yrs,
                    *self.user_params,
                    self.use_deg_penalty
                )

        power_timeseries,stack_current=pem.grid_connected_func(hydrogen_production_capacity_required_kgphr)
        h2_ts, h2_tot =pem.run_grid_connected_workaround(power_timeseries,stack_current)
        #h2_ts, h2_tot = pem.run(power_timeseries)
        h2_df_ts=pd.Series(h2_ts,name='Cluster #0')
        h2_df_tot=pd.Series(h2_tot,name='Cluster #0')
        # h2_df_ts = pd.DataFrame(h2_ts, index=list(h2_ts.keys()), columns=['Cluster #0'])
        # h2_df_tot = pd.DataFrame(h2_tot, index=list(h2_tot.keys()), columns=['Cluster #0'])
        []
        return pd.DataFrame(h2_df_ts),pd.DataFrame(h2_df_tot)



    def run(self, optimize=False):
        # TODO: add control type as input!
        clusters = self.create_clusters()  # initialize clusters
        if optimize:
            power_to_clusters = self.optimize_power_split()  # run Sanjana's code
        else:
            power_to_clusters = self.even_split_power()
        h2_df_ts = pd.DataFrame()
        h2_df_tot = pd.DataFrame()

        col_names = []
        start = time.perf_counter()
        for ci, cluster in enumerate(clusters):
            cl_name = "Cluster #{}".format(ci)
            col_names.append(cl_name)
            h2_ts, h2_tot = clusters[ci].run(power_to_clusters[ci])
            # h2_dict_ts['Cluster #{}'.format(ci)] = h2_ts

            h2_ts_temp = pd.Series(h2_ts, name=cl_name)
            h2_tot_temp = pd.Series(h2_tot, name=cl_name)
            if len(h2_df_tot) == 0:
                # h2_df_ts=pd.concat([h2_df_ts,h2_ts_temp],axis=0,ignore_index=False)
                h2_df_tot = pd.concat(
                    [h2_df_tot, h2_tot_temp], axis=0, ignore_index=False
                )
                h2_df_tot.columns = col_names

                h2_df_ts = pd.concat([h2_df_ts, h2_ts_temp], axis=0, ignore_index=False)
                h2_df_ts.columns = col_names
            else:
                # h2_df_ts = h2_df_ts.join(h2_ts_temp)
                h2_df_tot = h2_df_tot.join(h2_tot_temp)
                h2_df_tot.columns = col_names

                h2_df_ts = h2_df_ts.join(h2_ts_temp)
                h2_df_ts.columns = col_names

        end = time.perf_counter()
        self.clusters = clusters
        if self.verbose:
            print("Took {} sec to run the RUN function".format(round(end - start, 3)))
        return h2_df_ts, h2_df_tot
        # return h2_dict_ts, h2_df_tot

    def optimize_power_split(self):
        number_of_stacks = self.num_clusters
        rated_power = self.cluster_cap_mw * 1000
        tf = 96
        n_times_to_run = int(np.ceil(self.T / tf))
        df = pd.DataFrame({"Wind + PV Generation": self.input_power_kw})
        P_ = None
        I_ = None
        Tr_ = None
        AC = 1
        F_tot = 1
        diff = 0

        for start_time in range(n_times_to_run):
            print(
                f"Optimizing {number_of_stacks} stacks tarting {start_time*tf}hr/{self.T}hr"
            )
            if start_time == 0:

                df["Wind + PV Generation"].replace(0, np.NaN, inplace=True)
                df = df.interpolate()

            P_wind_t = df["Wind + PV Generation"][
                (start_time * tf) : ((start_time * tf) + tf)
            ].values
            start = time.time()
            if P_ is not None:
                P_ = P_[: len(P_wind_t), :]
                I_ = I_[: len(P_wind_t), :]
                Tr_ = Tr_[: len(P_wind_t), :]
            P_tot_opt, P_, H2f, I_, Tr_, P_wind_t, AC, F_tot = optimize(
                P_wind_t,
                T=(len(P_wind_t)),
                n_stacks=(number_of_stacks),
                c_wp=0,
                c_sw=self.switching_cost,
                rated_power=rated_power,
                P_init=P_,
                I_init=I_,
                T_init=Tr_,
                AC_init=AC,
                F_tot_init=F_tot,
            )

            diff += time.time() - start
            if type(AC).__module__ != "numpy":
                AC = np.array(AC)
                F_tot = np.array(F_tot)
            if start_time == 0:
                P_tot_opt_full = P_tot_opt
                P_full = P_
                P_wind_t_full = P_wind_t
                H2f_full = H2f
                I_full = I_
                Tr_full = np.sum(Tr_, axis=0)
                AC_full = AC
                F_tot_full = F_tot

            else:

                P_full = np.vstack((P_full, P_))
                H2f_full = np.vstack((H2f_full, H2f))
                I_full = np.vstack((I_full, I_))
                Tr_full = np.vstack((Tr_full, np.sum(Tr_, axis=0)))
                AC_full = np.vstack((AC_full, (AC)))
                F_tot_full = np.vstack((F_tot_full, (F_tot)))

        return np.transpose(P_full)

    def even_split_power(self):
        start = time.perf_counter()
        # determine how much power to give each cluster
        num_clusters_on = np.floor(self.input_power_kw / self.cluster_min_power)
        num_clusters_on = np.where(
            num_clusters_on > self.num_clusters, self.num_clusters, num_clusters_on
        )
        power_per_cluster = [
            self.input_power_kw[ti] / num_clusters_on[ti]
            if num_clusters_on[ti] > 0
            else 0
            for ti, pwr in enumerate(self.input_power_kw)
        ]

        power_per_to_active_clusters = np.array(power_per_cluster)
        power_to_clusters = np.zeros((len(self.input_power_kw), self.num_clusters))
        for i, cluster_power in enumerate(
            power_per_to_active_clusters
        ):  # np.arange(0,self.n_stacks,1):
            clusters_off = self.num_clusters - int(num_clusters_on[i])
            no_power = np.zeros(clusters_off)
            with_power = cluster_power * np.ones(int(num_clusters_on[i]))
            tot_power = np.concatenate((with_power, no_power))
            power_to_clusters[i] = tot_power

        # power_to_clusters = np.repeat([power_per_cluster],self.num_clusters,axis=0)
        end = time.perf_counter()
        
        if self.verbose:
            print(
                "Took {} sec to run even_split_power function".format(
                    round(end - start, 3)
                )
            )
        # rows are power, columns are stacks [300 x n_stacks]

        return np.transpose(power_to_clusters)

    def max_h2_cntrl(self):
        # run as many at lower power as possible
        []

    def min_deg_cntrl(self):
        # run as few as possible
        []

    def create_clusters(self):
        start = time.perf_counter()
        stacks = []
        # TODO fix the power input - don't make it required!
        # in_dict={'dt':3600}
        for i in range(self.num_clusters):
            # stacks.append(PEMClusters(cluster_size_mw = self.cluster_cap_mw))
            stacks.append(
                PEMClusters(
                    self.cluster_cap_mw,
                    self.plant_life_yrs,
                    *self.user_params,
                    self.use_deg_penalty,
                    self.turndown_ratio,
                )
            )
        end = time.perf_counter()
        if self.verbose:
            print("Took {} sec to run the create clusters".format(round(end - start, 3)))
        return stacks


if __name__ == "__main__":

    system_size_mw = 1000
    num_clusters = 20
    cluster_cap_mw = system_size_mw / num_clusters
    stack_rating_kw = 1000
    cluster_min_power_kw = 0.1 * stack_rating_kw * cluster_cap_mw
    num_steps = 200
    power_rampup = np.arange(
        cluster_min_power_kw, system_size_mw * stack_rating_kw, cluster_min_power_kw
    )

    plant_life = 30
    deg_penalty = True
    user_defined_electrolyzer_EOL_eff_drop = False
    EOL_eff_drop = 13
    user_defined_electrolyzer_BOL_kWh_per_kg = False
    BOL_kWh_per_kg = []
    electrolyzer_model_parameters = {
        "Modify BOL Eff": user_defined_electrolyzer_BOL_kWh_per_kg,
        "BOL Eff [kWh/kg-H2]": BOL_kWh_per_kg,
        "Modify EOL Degradation Value": user_defined_electrolyzer_EOL_eff_drop,
        "EOL Rated Efficiency Drop": EOL_eff_drop,
    }
    # power_rampup = np.linspace(cluster_min_power_kw,system_size_mw*1000,num_steps)
    power_rampdown = np.flip(power_rampup)
    power_in = np.concatenate((power_rampup, power_rampdown))
    pem = run_PEM_clusters(
        power_in,
        system_size_mw,
        num_clusters,
        plant_life,
        electrolyzer_model_parameters,
        deg_penalty,
    )

    h2_ts, h2_tot = pem.run()
    # pem.clusters[0].cell_design(80,1920*2)
    []
