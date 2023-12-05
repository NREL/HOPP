# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 09:06:19 2022

@author: ereznic2
"""
import os
import numpy as np
import pandas as pd

import hopp.to_organize.hydrogen_steel_pipe_cost_functions as hydrogen_steel_pipe_cost_functions

#parent_path = os.path.abspath('')

def hydrogen_steel_pipeline_cost_analysis(parent_path,turbine_model,hydrogen_max_hourly_production_kg,site_name):
    pipe_info_dir = parent_path + '/H2_Analysis/'

    pipeline_info = pd.read_csv(pipe_info_dir+'/Pipeline_info.csv',header = 0,sep=',')

    pipeline_info = pipeline_info.loc[pipeline_info['Site']==site_name].reset_index().drop(['index'],axis=1)

    total_plant_hydrogen_production_kgpsec = hydrogen_max_hourly_production_kg/3600

    pipe_diameter_min_mm = []
    for j in range(pipeline_info.shape[0]):
        #j=0
        flowrate_per_turbine = total_plant_hydrogen_production_kgpsec/pipeline_info.loc[j,'Total number of turbines']
        pipe_mass_flow_kgpsec = flowrate_per_turbine*pipeline_info.loc[j,'Number of turbines in each pipe']
        # Decide pipe inlet pressure based on where the pipe starts
        if pipeline_info.loc[j,'Pipe start point'] == 'Turbine':
            pressure_inlet_bar = 40
        elif pipeline_info.loc[j,'Pipe start point'] == 'Pipe':
            pressure_inlet_bar = 20

        if pipeline_info.loc[j,'Pipe end point'] == 'Central':
            pressure_outlet_bar =5
        elif pipeline_info.loc[j,'Pipe end point'] == 'Pipe':
            pressure_outlet_bar = 20

        pipe_diameter_min_mm.append(25.4*hydrogen_steel_pipe_cost_functions.get_diameter_of_pipe(pipeline_info.loc[j,'Length of Pipe in Arm (m)'],pipe_mass_flow_kgpsec,pressure_inlet_bar,pressure_outlet_bar))

    # Design specifications for the pipes
    grade = 'X52'
    design_option = 'B'
    location_class = 1
    joint_factor = 1
    design_pressure_asme = pressure_inlet_bar/10

    # Calculate minimum DN, schedule, and associated thickness
    pipe_min_DN = []
    pipe_outer_diameter = []
    pipe_inner_diameter = []
    pipe_min_schedule = []
    pipe_min_thickness = []

    # Loop through pipes in the network
    for min_inner_diameter in pipe_diameter_min_mm:

        dn,outer_diameter,schedule_minviable,thickness_of_schedule=hydrogen_steel_pipe_cost_functions.get_dn_schedule_of_pipe(pipe_info_dir,grade,design_option,location_class,joint_factor,min_inner_diameter,design_pressure_asme)

        inner_diameter = outer_diameter - 2*thickness_of_schedule

        pipe_min_DN.append(dn)
        pipe_outer_diameter.append(outer_diameter)
        pipe_inner_diameter.append(inner_diameter)
        pipe_min_schedule.append(schedule_minviable)
        pipe_min_thickness.append(thickness_of_schedule)

    # Economic analysis
    steel_costs_per_kg_all_grades = pd.read_csv(pipe_info_dir+'steel_costs_per_kg.csv',index_col = None,header = 0)
    steel_cost_per_kg = steel_costs_per_kg_all_grades.loc[steel_costs_per_kg_all_grades['Grade']==grade,'Price [$/kg]'].to_list()[0]
    density_steel = 7840

    cpi_2004 = 251.1
    cpi_2019 = 255.7

    cpi_ratio = cpi_2019/cpi_2004

    pipe_material_cost_bymass_USD = []
    pipe_material_cost_USD = []
    pipe_misc_cost_USD = []
    pipe_labor_cost_USD = []
    pipe_row_cost_USD = []
    pipe_total_cost_USD = []

    #   ANL costing coefficients for all regions. Material cost is ignored as internal methods are used
    anl_coefs_regional = {'GP':{'labor':[10406,0.20953,-0.08419],'misc':[4944,0.17351,-0.07621],'ROW':[2751,-0.28294,0.00731],'Material':[5813,0.31599,-0.00376]},
                            'NE':{'labor':[249131,-0.33162,-0.17892],'misc':[65990,-0.29673,-0.06856],'ROW':[83124,-0.66357,-0.07544],'Material':[10409,0.296847,-0.07257]},
                            'MA':{'labor':[43692,0.05683,-0.10108],'misc':[14616,0.16354,-0.16186],'ROW':[1942,0.17394,-0.01555],'Material':[9113,0.279875,-0.00840]},
                            'GL':{'labor':[58154,-0.14821,-0.10596],'misc':[41238,-0.34751,-0.11104],'ROW':[14259,-0.65318,0.06865],'Material':[8971,0.255012,-0.03138]},
                            'RM':{'labor':[10406,0.20953,-0.08419],'misc':[4944,0.17351,-0.07621],'ROW':[2751,-0.28294,0.00731],'Material':[5813,0.31599,-0.00376]},
                            'SE':{'labor':[32094,0.06110,-0.14828],'misc':[11270,0.19077,-0.13669],'ROW':[9531,-0.37284,0.02616],'Material':[6207,0.38224,-0.05211]},
                            'PN':{'labor':[32094,0.06110,-0.14828],'misc':[11270,0.19077,-0.13669],'ROW':[9531,-0.37284,0.02616],'Material':[6207,0.38224,-0.05211]},
                            'SW':{'labor':[95295,-0.53848,0.03070],'misc':[19211,-0.14178,-0.04697],'ROW':[72634,-1.07566,0.05284],'Material':[5605,0.41642,-0.06441]},
                            'CA':{'labor':[95295,-0.53848,0.03070],'misc':[19211,-0.14178,-0.04697],'ROW':[72634,-1.07566,0.05284],'Material':[5605,0.41642,-0.06441]}}

    region={'IN':'GL','TX':'SW','IA':'GP','MS':'SE','MN':'GP','WY':'RM'}
    anl_coefs = anl_coefs_regional[region[site_name]]
    small_positive = 1e-7 # This allows solution and file output writing when length of a given DN is set to zero, but usually this is a sign of an issue somewhere
    for j in range(len(pipe_min_DN)):
       # j = 0

        # Calculate pipe material cost based on pipe volume
        pipe_volume = np.pi/4*((0.001*pipe_outer_diameter[j])**2 - (0.001*pipe_inner_diameter[j])**2)*pipeline_info.loc[j,'Length of Pipe in Arm (m)']
        pipe_mass_kg = density_steel*pipe_volume
        pipe_material_cost_bymass_USD.append(steel_cost_per_kg*pipe_mass_kg*pipeline_info.loc[j,'Number of such pipes needed'])

        # Calculate costs using Parker 2004 data adjusted to 2019 (two methods)
        pipeline_length_miles = pipeline_info.loc[j,'Number of such pipes needed']*pipeline_info.loc[j,'Length of Pipe in Arm (m)']*0.621371/1000
        pipe_diam_in = pipe_outer_diameter[j]/25.4

        pipe_material_cost_USD.append(cpi_ratio*anl_coefs['Material'][0]*(pipe_diam_in**anl_coefs['Material'][1])*(pipeline_length_miles**anl_coefs['Material'][2]+small_positive)*pipe_diam_in*pipeline_length_miles)
        pipe_labor_cost_USD.append(cpi_ratio*anl_coefs['labor'][0]*(pipe_diam_in**anl_coefs['labor'][1])*(pipeline_length_miles**anl_coefs['labor'][2]+small_positive)*pipe_diam_in*pipeline_length_miles)
        pipe_misc_cost_USD.append(cpi_ratio*anl_coefs['misc'][0]*(pipe_diam_in**anl_coefs['misc'][1])*(pipeline_length_miles**anl_coefs['misc'][2]+small_positive)*pipe_diam_in*pipeline_length_miles)
        pipe_row_cost_USD.append( cpi_ratio*anl_coefs['ROW'][0]*(pipe_diam_in**anl_coefs['ROW'][1])*(pipeline_length_miles**anl_coefs['ROW'][2]+small_positive)*pipe_diam_in*pipeline_length_miles)

        pipe_total_cost_USD.append(pipe_material_cost_USD[j] + pipe_misc_cost_USD[j] + pipe_labor_cost_USD[j] + pipe_row_cost_USD[j])

    pipe_material_cost_USD = pd.DataFrame(pipe_material_cost_USD,columns = ['Total material cost ($)'])
    pipe_misc_cost_USD = pd.DataFrame(pipe_misc_cost_USD,columns = ['Total misc cost ($)'])
    pipe_labor_cost_USD = pd.DataFrame(pipe_labor_cost_USD,columns = ['Total labor cost ($)'])
    pipe_row_cost_USD = pd.DataFrame(pipe_row_cost_USD,columns = ['Total ROW cost ($)'])

    pipe_material_cost_bymass_USD = pd.DataFrame(pipe_material_cost_bymass_USD,columns = ['Total material cost ($)'])

    pipe_network_costs_USD = pipe_material_cost_USD.join(pipe_misc_cost_USD)
    pipe_network_costs_USD = pipe_network_costs_USD.join(pipe_labor_cost_USD)
    pipe_network_costs_USD = pipe_network_costs_USD.join(pipe_row_cost_USD)

    pipe_network_cost_total_USD = sum(pipe_total_cost_USD)

    return(pipe_network_cost_total_USD,pipe_network_costs_USD,pipe_material_cost_bymass_USD)