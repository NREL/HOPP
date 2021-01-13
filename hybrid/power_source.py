import logging
from typing import Optional, Sequence
from hybrid.sites import SiteInfo
import PySAM.Singleowner as Singleowner
import pandas as pd
import numpy as np
import os

from hybrid.log import hybrid_logger as logger
from tools.powerflow import plant_grid
from tools.powerflow import cable
from tools.powerflow import power_flow_solvers
from tools.powerflow import visualization
from random import random


class PowerSource:
    def __init__(self, name, site: SiteInfo, system_model, financial_model):
        """
        Abstract class for a renewable energy power plant simulation.
        """
        self.name = name
        self.site = site
        self.system_model = system_model
        self.financial_model = financial_model
        self.set_construction_financing_cost_per_kw(financial_model.FinancialParameters.construction_financing_cost \
                                                    / financial_model.FinancialParameters.system_capacity)


    @property
    def system_capacity_kw(self) -> float:
        raise NotImplementedError

    def get_total_installed_cost_dollars(self) -> float:
        return self.financial_model.SystemCosts.total_installed_cost

    def set_total_installed_cost_dollars(self, total_installed_cost_dollars: float):
        self.financial_model.SystemCosts.total_installed_cost = total_installed_cost_dollars
        logger.info("{} set total_installed_cost to ${}".format(self.name, total_installed_cost_dollars))

    def set_construction_financing_cost_per_kw(self, construction_financing_cost_per_kw):
        self._construction_financing_cost_per_kw = construction_financing_cost_per_kw

    def get_construction_financing_cost(self) -> float:
        return self._construction_financing_cost_per_kw * self.system_capacity_kw

    def simulate_powerflow(self, project_life: int = 25):
        if self.name == 'WindPlant':
            gen = self.generation_profile()
            print("From simulate, printing turbine coords: X = {}, Y = {}".
                  format(self.system_model.Farm.wind_farm_xCoordinates,
                         self.system_model.Farm.wind_farm_yCoordinates))
            grid_connect_coords = np.array([[int(self.system_model.Turbine.wind_turbine_rotor_diameter), 0, 0]])
            turbine_coordinates_list = [[int(x), 0, 0] for x in self.system_model.Farm.wind_farm_xCoordinates]
            turbine_coordinates = np.array(turbine_coordinates_list)
            node_labels = ['T{}'.format(node_num) for node_num in range(1, len(turbine_coordinates)+1)]
            power_flow_grid_model = plant_grid.PlantGridModel(grid_connect_coords, 'G')
            power_flow_grid_model.add_nodes(turbine_coordinates, node_labels)
            cable1 = cable.CableByGeometry(1000 / 1000 ** 2)
            node_connection_matrix = list()
            node_connection_matrix_hub = ['G', 'T1']
            node_connection_matrix_turbines = [['T{}'.format(turb_node_label_num), 'T{}'.format(turb_node_label_num+1)]
                                               for turb_node_label_num in range(1, len(turbine_coordinates))]
            node_connection_matrix.append(node_connection_matrix_hub)
            node_connection_matrix += node_connection_matrix_turbines
            power_flow_grid_model.add_connections(node_connection_matrix, cable1, length_method='direct')


            power_flow_grid_model.assign_nominal_quantities(2e6, 14e3)

            # Add summary details to dataframe
            df_powerflow_summary = pd.DataFrame()
            # hybrid_size_kw = 100
            # df_powerflow_summary.hybrid_size_kw = hybrid_size_kw
            df_powerflow_summary.node_labels = node_labels
            df_powerflow_summary.cable = cable1
            df_powerflow_summary.turbine_coords = turbine_coordinates

            # Create lists to save details at each timestep
            df_powerflow_details = pd.DataFrame()
            gen_at_time_list = list()
            P_turbs_list = list()
            P_losses_list = list()
            P_losses_pc_list = list()
            s_NR_list = list()
            i_NR_list = list()
            v_NR_list = list()
            comment_list = list()

            # iterate through each gen timestep
            turb_gen = [int(x / self.num_turbines) * 1000 for x in gen]  # Get rough generation per turbine
            count = 0
            for gen_at_time in turb_gen:
                count = count + 1  # Timestep count
                if gen_at_time > 0:
                    P_turbs = [[random() * gen_at_time + (random() * gen_at_time * 0.01e1j)] for _ in
                               range(self.num_turbines)]
                    power_flow_grid_model = plant_grid.PlantGridModel(grid_connect_coords, 'G')
                    power_flow_grid_model.add_nodes(turbine_coordinates, node_labels)
                    power_flow_grid_model.add_connections(node_connection_matrix, cable1, length_method='direct')
                    power_flow_grid_model.assign_nominal_quantities(2e6, 14e3)

                    if count == 2:
                        # Set some of the generation at different turbines to zero
                        P_turbs[3][0] = 0
                        P_turbs[4][0] = 0
                        P_turbs[5][0] = 0
                        P_turbs[6][0] = 0
                        comment = '(Turbine 4, 5, 6, 7 set to zero)'
                    elif count == 3:
                        #  Break one of the connections:
                        #  Node connection to node 11 gets removed here.
                        new_node_connection_matrix =[['G', 'T1'], ['T1', 'T2'], ['T2', 'T3'], ['T3', 'T4'],
                                                       ['T4', 'T5'], ['T5', 'T6'], ['T6', 'T7'], ['T7', 'T8'],
                                                       ['T8', 'T9'], ['T9', 'T10'], ['T10', 'T12'], ['T12', 'T13'],
                                                       ['T13', 'T14'], ['T14', 'T15'], ['T15', 'T16'], ['T16', 'T17'],
                                                       ['T17', 'T18'], ['T18', 'T19'], ['T19', 'T20']]
                        power_flow_grid_model.add_connections(new_node_connection_matrix, cable1, length_method='direct')
                        comment = '(Removed link between nodes 10-11-12)'
                    else:
                        comment = ''

                    P_turbs = np.array(P_turbs)

                    # Solve power flow equations
                    s_NR, v_NR, i_NR = power_flow_solvers.Netwon_Raphson_method(
                        power_flow_grid_model.admittance_matrix_pu,
                        P_turbs / power_flow_grid_model.nominal_properties['power'],
                        max_iterations=20,
                        quiet=False)
                    print('Voltages:', v_NR*power_flow_grid_model.nominal_properties['voltage'])
                    s_found = s_NR * power_flow_grid_model.nominal_properties['power']
                    P_losses = np.sum(s_found)
                    P_losses_pc = 100 * np.real(P_losses) / np.sum(np.real(P_turbs)) + \
                                  100j * np.imag(P_losses) / np.sum(np.imag(P_turbs))
                    print('Losses (real and reactive):', P_losses_pc, '%.')
                    print(count)
                    gen_at_time_list.append(gen_at_time)
                    P_turbs_list.append(P_turbs)
                    P_losses_list.append(P_losses)
                    P_losses_pc_list.append(P_losses_pc)
                    s_NR_list.append(s_NR)
                    i_NR_list.append(i_NR)
                    v_NR_list.append(v_NR)
                    comment_list.append(comment)

                    # Visualize
                    real_power = np.real(P_turbs)
                    real_power = [x[0] for x in real_power]
                    real_power = np.array(real_power)
                    ax, plt = visualization.grid_layout(power_flow_grid_model)
                    visualization.overlay_quantity(power_flow_grid_model, real_power, ax, 'Real Power (kW)',
                                                   'Real Power at Timestep {} {}'.format(count, comment))
                    plotname = '{}_Real Power{}'.format(count, '.jpg')
                    plt.savefig(os.path.join('results', plotname))
                    plt.close()

                    reactive_power = np.imag(P_turbs)
                    reactive_power = [x[0] for x in reactive_power]
                    reactive_power = np.array(reactive_power)
                    ax, plt = visualization.grid_layout(power_flow_grid_model)
                    visualization.overlay_quantity(power_flow_grid_model, reactive_power, ax, 'Reactive Power (kW)',
                                                   'Reactive Power at Timestep {} {}'.format(count, comment))
                    plotname = '{}_Reactive Power{}'.format(count, '.jpg')
                    plt.savefig(os.path.join('results', plotname))
                    plt.close()

                    real_voltage = np.real(v_NR)
                    real_voltage = [abs(x[0]) for x in real_voltage]
                    real_voltage = np.array(real_voltage)
                    ax, plt = visualization.grid_layout(power_flow_grid_model)
                    visualization.overlay_quantity(power_flow_grid_model, real_voltage, ax, 'Real Voltage (V)',
                                                   'Real Voltage at Timestep {} {}'.format(count, comment))
                    plotname = '{}_Real Voltage{}'.format(count, '.jpg')
                    plt.savefig(os.path.join('results', plotname))
                    plt.close()

                    reactive_voltage = np.imag(v_NR)
                    reactive_voltage = [abs(x[0]) for x in reactive_voltage]
                    reactive_voltage = np.array(reactive_voltage)
                    ax, plt = visualization.grid_layout(power_flow_grid_model)
                    visualization.overlay_quantity(power_flow_grid_model, reactive_voltage, ax, 'Reactive Voltage (V)',
                                                   'Reactive Voltage at Timestep {} {}'.format(count, comment))
                    plotname = '{}_Reactive Voltage{}'.format(count, '.jpg')
                    plt.savefig(os.path.join('results', plotname))
                    plt.close()

                else:
                    gen_at_time = 0
                    P_turbs = 0
                    P_losses = 0
                    P_losses_pc = 0
                    s_NR = 0
                    i_NR = 0
                    v_NR = 0
                    comment = 'No generation from turbines'
                    gen_at_time_list.append(gen_at_time)
                    P_turbs_list.append(P_turbs)
                    P_losses_list.append(P_losses)
                    P_losses_pc_list.append(P_losses_pc)
                    s_NR_list.append(s_NR)
                    i_NR_list.append(i_NR)
                    v_NR_list.append(v_NR)
                    comment_list.append(comment)

            powerflow_dict = {'Comment': comment_list, 'gen_at_time': gen_at_time_list, 'P_turbs': P_turbs_list,
                              'P_losses': P_losses_list, 'P_losses_percent': P_losses_pc_list,
                              's_NR': s_NR_list, 'i_NR': i_NR_list, 'v_NR': v_NR_list}

            df_powerflow_details = pd.DataFrame(powerflow_dict)
            df_powerflow_details.to_csv('powerflow_details.csv')

    def simulate(self, project_life: int = 25, power_flow_calculation: bool = False):
        """
        Run the system and financial model
        """
        if not self.system_model:
            return
        self.system_model.execute(0)
        if power_flow_calculation:
            self.simulate_powerflow()
        if not self.financial_model:
            return

        self.financial_model.value("construction_financing_cost", self.get_construction_financing_cost())
        self.financial_model.Revenue.ppa_soln_mode = 1
        self.financial_model.Lifetime.system_use_lifetime_output = 1
        self.financial_model.FinancialParameters.analysis_period = project_life
        single_year_gen = self.financial_model.SystemOutput.gen
        self.financial_model.SystemOutput.gen = list(single_year_gen) * project_life

        if self.name != "Grid":
            self.financial_model.SystemOutput.system_pre_curtailment_kwac = self.system_model.Outputs.gen * project_life
            self.financial_model.SystemOutput.annual_energy_pre_curtailment_ac = self.system_model.Outputs.annual_energy

        self.financial_model.execute(0)
        logger.info("{} simulation executed".format(self.name))

    def generation_profile(self) -> Sequence:
        if self.system_capacity_kw:
            return self.system_model.Outputs.gen
        else:
            return [0] * self.site.n_timesteps

    def copy(self):
        """
        :return: new instance
        """
        raise NotImplementedError
