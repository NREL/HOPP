import pyomo.environ as pyomo
from pyomo.network import Port, Arc
from pyomo.environ import units as u
import numpy as np
import matplotlib.pyplot as plt


def controls_analysis(grid_info: dict,
                        control_info: dict):
        """ Function to execute the controls analysis based on the control case specified in 
                the dictionary control_info

        Args:
               :param grid_info: :class:`hybrid.grid.Grid`
                     Grid class from hybrid simulation

               control_info (Dictionary): Control case information input from user
                        currently can only use the keys 'baseload' and 'frequency_regulation' control cases
        """
        
        if 'baseload' in control_info.keys():
                percent_met, overall_power_met = baseload_sim(grid_info, control_info)

                return percent_met, overall_power_met
        
        if 'frequency_regulation' in control_info.keys():
                percent_met, frequency_regulation_hrs = frequency_regulation_sim(grid_info, control_info)

                return percent_met, frequency_regulation_hrs


def baseload_sim(grid_info,
                        control_info):

        """
        Baseload analysis for firm power operating case: determines the percentage of time that the power output of the hybrid 
                plant fulfills the firm power requirement (here, assumed to be a constant value).  Also determines the percentage
                of power provided for the firm power requrirement overall
        Args:
               grid_info (Dictionary): Grid class from hybrid simulation
               control_info (Dictionary): Control case information input from user
                        in control_info, need:'baseload_limit': constant value of firm power you want to provide (kW)
                                              'baseload_percent' : percent error allowed in firm power requirement (between 0 and 100)

        :returns: percent_met, overall_power_met        

        """
        # Setting given values
        baseload_value_kw = control_info['baseload']['baseload_limit']
        baseload_percent = control_info['baseload']['baseload_percent'] / 100
        N_hybrid = len(grid_info.generation_profile)

        final_power_production = grid_info.generation_profile
        hybrid_power = [(x - (baseload_value_kw*0.95 * baseload_percent)) for x in final_power_production]

        baseload_met = len([i for i in hybrid_power if i  >= 0])
        percent_met = 100 * baseload_met/N_hybrid

        final_power_array = np.array(final_power_production)
        power_met = np.where(final_power_array > baseload_value_kw, baseload_value_kw, final_power_array)
        overall_power_met = np.sum(power_met) / (N_hybrid * baseload_value_kw) * 100

        # ## plotting first 12 days for validation ##
        # power_scale = 1/1000
        # hybrid_power_mw = [x*power_scale for x in final_power_production]
        # plt.figure(figsize=(8, 5))
        # plt.plot(hybrid_power_mw[0:96*3], 'm-x', label='Wind+Solar+Storage')
        # plt.plot(range(0,96*3), np.ones(96*3)*baseload_value_kw*power_scale, 'k', label='Baseload Goal')
        # plt.xlabel('Time [hr]')
        # plt.ylabel('Power [MW]')
        # plt.legend()
        # plt.show()
        
        
        return percent_met, overall_power_met

def frequency_regulation_sim(grid_info,
                        control_info):

        """
        Frequency regulation function for essential reliability services (ERS) availability operating case:
                Finds how many hours (in the group specified group size above the specified minimum
                power requirement) that the system has available to extra power that could be used to 
                provide ERS
        Args:
               grid_info (Dictionary): Grid class from hybrid simulation
               control_info (Dictionary): Control case information input from user
                in control_info, need:
                                'baseload_limit': constant value of firm power you want to provide (kW)
                                'baseload_percent' : percent error allowed in firm power requirement (between 0 and 100)
                                'min_regulation_hours': minimum size of hours in a group to be considered for ERS (>= 1)
                                'min_regulation_power': minimum power available over the whole group of hours to be 
                                        considered for ERS (> 0, in kW)

        :returns: percent_met, total_number_hours

        """
        # Setting given variables
        baseload_value_kw = control_info['frequency_regulation']['baseload_limit']
        baseload_percent = control_info['frequency_regulation']['baseload_percent'] / 100
        min_regulation_hours = control_info['frequency_regulation']['min_regulation_hours']
        min_regulation_power = control_info['frequency_regulation']['min_regulation_power']
        N_hybrid = len(grid_info.generation_profile)

        # Determining baseload power
        final_power_production = grid_info.generation_profile
        hybrid_power = [(x - (baseload_value_kw*0.95 * baseload_percent)) for x in final_power_production]

        baseload_met = len([i for i in hybrid_power if i  >= 0])
        percent_met = 100 * baseload_met/N_hybrid

        # Performing frequency regulation analysis:
        #    finding how many groups of hours satisfiy the ERS minimum power requirement
        frequency_power = [(x - (baseload_value_kw)) for x in final_power_production]
        frequency_power_array = np.array(frequency_power)
        frequency_test = np.where(frequency_power_array > min_regulation_power, frequency_power_array, 0)
        mask = (frequency_test!=0).astype(np.int)
        padded_mask = np.pad(mask,(1,), "constant")
        edge_mask = padded_mask[1:] - padded_mask[:-1]  # finding the difference between each array value

        group_starts = np.where(edge_mask == 1)[0]
        group_stops = np.where(edge_mask == -1)[0]

        # Find groups and drop groups that are too small
        groups = [group for group in zip(group_starts,group_stops) if ((group[1]-group[0]) >= min_regulation_hours)]
        group_lengths = [len(final_power_production[group[0]:group[1]]) for group in groups]
        total_number_hours = sum(group_lengths)

        # ## plotting first 12 days for validation ##
        # power_scale = 1/1000
        # hybrid_power_mw = [x*power_scale for x in final_power_production]
        # plt.figure(figsize=(8, 5))
        # plt.plot(hybrid_power_mw[0:96*3], 'm-x', label='Wind+Solar+Storage')
        # plt.plot(range(0,96*3), np.ones(96*3)*baseload_value_kw*power_scale, 'k', label='Baseload Goal')
        # plt.xlabel('Time [hr]')
        # plt.ylabel('Power [MW]')
        # plt.legend()
        # plt.show()
       
        
        return percent_met, total_number_hours


