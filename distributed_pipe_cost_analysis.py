# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 09:06:19 2022

@author: ereznic2
"""
import os
import numpy as np 
import pandas as pd

import hydrogen_steel_pipe_cost_functions

#parent_path = os.path.abspath('')

def hydrogen_steel_pipeline_cost_analysis(parent_path,turbine_model,hydrogen_max_hourly_production_kg,site_name):
    pipe_info_dir = parent_path + '/examples/H2_Analysis/'
    
    pipeline_info = pd.read_csv(pipe_info_dir+'/Pipeline_info.csv',header = 0,sep=',')
    
    turbine_case = 'lbw_'+turbine_model
    
    pipeline_info = pipeline_info.loc[pipeline_info['Site']==turbine_case].reset_index().drop(['index'],axis=1)
    
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
    for j in range(len(pipe_min_DN)):
       # j = 0
       
        # Calculate pipe material cost based on pipe volume
        pipe_volume = np.pi/4*((0.001*pipe_outer_diameter[j])**2 - (0.001*pipe_inner_diameter[j])**2)*pipeline_info.loc[j,'Length of Pipe in Arm (m)']
        pipe_mass_kg = density_steel*pipe_volume
        pipe_material_cost_bymass_USD.append(steel_cost_per_kg*pipe_mass_kg*pipeline_info.loc[j,'Number of such pipes needed'])
        
        # Calculate costs using Parker 2004 data adjusted to 2019 (two methods)
        pipeline_length_miles = pipeline_info.loc[j,'Number of such pipes needed']*pipeline_info.loc[j,'Length of Pipe in Arm (m)']*0.621371/1000
        pipe_diam_in = pipe_outer_diameter[j]/25.4
        
        if site_name == 'IN':
            pipe_material_cost_USD.append(cpi_ratio*8971*(pipe_diam_in**0.255012)/(pipeline_length_miles**0.03138)*pipe_diam_in*pipeline_length_miles)
            
            pipe_labor_cost_USD.append(cpi_ratio*58154*(pipe_diam_in**0.14821)/(pipeline_length_miles**0.10596)*pipe_diam_in*pipeline_length_miles)
            
            pipe_misc_cost_USD.append(cpi_ratio*41238*(pipe_diam_in**0.34751)/(pipeline_length_miles**0.11104)*pipe_diam_in*pipeline_length_miles)
            
            pipe_row_cost_USD.append(cpi_ratio*14259*(pipe_diam_in**0.65318)/(pipeline_length_miles**0.06865)*pipe_diam_in*pipeline_length_miles)
            
        
        elif site_name == 'TX':
            pipe_material_cost_USD.append(cpi_ratio*5605*(pipe_diam_in**0.41642)/(pipeline_length_miles**0.06441)*pipe_diam_in*pipeline_length_miles)
            
            pipe_labor_cost_USD.append(cpi_ratio*95295*(pipe_diam_in**0.53848)/(pipeline_length_miles**0.03070)*pipe_diam_in*pipeline_length_miles)
            
            pipe_misc_cost_USD.append(cpi_ratio*19211*(pipe_diam_in**0.14178)/(pipeline_length_miles**0.04697)*pipe_diam_in*pipeline_length_miles)
            
            pipe_row_cost_USD.append(cpi_ratio*72634*(pipe_diam_in**1.07566)/(pipeline_length_miles**0.05284)*pipe_diam_in*pipeline_length_miles)
            
            
        elif site_name == 'IA' or site_name == 'WY':
            pipe_material_cost_USD.append(cpi_ratio*5813*(pipe_diam_in**0.31599)/(pipeline_length_miles**0.00376)*pipe_diam_in*pipeline_length_miles)
            
            pipe_labor_cost_USD.append(cpi_ratio*10406*(pipe_diam_in**0.20953)/(pipeline_length_miles**0.08419)*pipe_diam_in*pipeline_length_miles)
            
            pipe_misc_cost_USD.append(cpi_ratio*4944*(pipe_diam_in**0.17351)/(pipeline_length_miles**0.07261)*pipe_diam_in*pipeline_length_miles)
            
            pipe_row_cost_USD.append(cpi_ratio*2751*(pipe_diam_in**0.28294)/(pipeline_length_miles**0.00731)*pipe_diam_in*pipeline_length_miles)
            
            
        elif site_name == 'MS':
            pipe_material_cost_USD.append(cpi_ratio*6207*(pipe_diam_in**0.38224)/(pipeline_length_miles**0.05211)*pipe_diam_in*pipeline_length_miles)
            
            pipe_labor_cost_USD.append(cpi_ratio*32094*(pipe_diam_in**0.06110)/(pipeline_length_miles**0.14828)*pipe_diam_in*pipeline_length_miles)
            
            pipe_misc_cost_USD.append(cpi_ratio*11270*(pipe_diam_in**0.19077)/(pipeline_length_miles**0.13669)*pipe_diam_in*pipeline_length_miles)
            
            pipe_row_cost_USD.append(cpi_ratio*9531*(pipe_diam_in**0.37284)/(pipeline_length_miles**0.02616)*pipe_diam_in*pipeline_length_miles)
        
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