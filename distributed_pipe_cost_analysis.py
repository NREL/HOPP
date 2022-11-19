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

def hydrogen_steel_pipeline_cost_analysis(parent_path,turbine_model,hydrogen_max_hourly_production_kg):
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
    
    cpi_2004 = 188.9
    cpi_2019 = 255.7
    
    cpi_ratio = cpi_2019/cpi_2004
            
    pipe_material_cost_USD = []
    pipe_material_cost_Parker_2019_USD = []
    pipe_misc_cost_USD = []
    pipe_misc_cost_Parker_2019_USD = []
    pipe_labor_cost_USD = []
    pipe_labor_cost_Parker_2019_USD = []
    pipe_row_cost_USD = []
    pipe_row_cost_Parker_2019_USD = []
    pipe_total_cost_USD = []
    pipe_total_cost_Parker_2019_USD = []
    for j in range(len(pipe_min_DN)):
       # j = 0
       
        # Calculate pipe material cost based on pipe volume
        pipe_volume = np.pi/4*((0.001*pipe_outer_diameter[j])**2 - (0.001*pipe_inner_diameter[j])**2)*pipeline_info.loc[j,'Length of Pipe in Arm (m)']
        pipe_mass_kg = density_steel*pipe_volume
        pipe_material_cost_USD.append(steel_cost_per_kg*pipe_mass_kg*pipeline_info.loc[j,'Number of such pipes needed'])
        
        # Calculate costs using Parker 2004 data adjusted to 2019 (two methods)
        pipeline_length_miles = pipeline_info.loc[j,'Number of such pipes needed']*pipeline_info.loc[j,'Length of Pipe in Arm (m)']*0.621371/1000
        pipe_diam_in = pipe_outer_diameter[j]/25.4
        
        # Calculate 
        pipe_material_cost_Parker_2019_USD.append(cpi_ratio*((330.5 * pipe_diam_in**2 + 687 * pipe_diam_in + 26960) * pipeline_length_miles + 35000))
          
        pipe_misc_cost_USD.append(cpi_ratio*4944*(pipe_diam_in**0.17351)/(pipeline_length_miles**0.07261)*pipe_diam_in*pipeline_length_miles)
        pipe_misc_cost_Parker_2019_USD.append(cpi_ratio*((8417 * pipe_diam_in + 7324) * pipeline_length_miles + 95000))
        
        pipe_labor_cost_USD.append(cpi_ratio*10406*(pipe_diam_in**0.20953)/(pipeline_length_miles**0.08419)*pipe_diam_in*pipeline_length_miles)
        pipe_labor_cost_Parker_2019_USD.append(cpi_ratio*((343 * pipe_diam_in**2 + 2047 * pipe_diam_in + 170013) * pipeline_length_miles + 185000))
        
        pipe_row_cost_USD.append(cpi_ratio*2751*(pipe_diam_in**0.28294)/(pipeline_length_miles**0.00731)*pipe_diam_in*pipeline_length_miles)
        pipe_row_cost_Parker_2019_USD.append(cpi_ratio*((577 * pipe_diam_in + 29788) * pipeline_length_miles + 40000 ))
        
        pipe_total_cost_USD.append(pipe_material_cost_USD[j] + pipe_misc_cost_USD[j] + pipe_labor_cost_USD[j] + pipe_row_cost_USD[j])
        pipe_total_cost_Parker_2019_USD.append(pipe_material_cost_Parker_2019_USD[j] + pipe_misc_cost_Parker_2019_USD[j] + pipe_labor_cost_Parker_2019_USD[j] + pipe_row_cost_Parker_2019_USD[j])
        
        
    pipe_material_cost_USD = pd.DataFrame(pipe_material_cost_USD,columns = ['Total material cost ($)'])
    pipe_misc_cost_USD = pd.DataFrame(pipe_misc_cost_USD,columns = ['Total misc cost ($)'])
    pipe_labor_cost_USD = pd.DataFrame(pipe_labor_cost_USD,columns = ['Total labor cost ($)'])
    pipe_row_cost_USD = pd.DataFrame(pipe_row_cost_USD,columns = ['Total ROW cost ($)'])
    
    pipe_network_costs_USD = pipe_material_cost_USD.join(pipe_misc_cost_USD)
    pipe_network_costs_USD = pipe_network_costs_USD.join(pipe_labor_cost_USD)
    pipe_network_costs_USD = pipe_network_costs_USD.join(pipe_row_cost_USD)
    
    pipe_material_cost_Parker_2019_USD = pd.DataFrame(pipe_material_cost_Parker_2019_USD,columns = ['Total material cost ($)'])
    pipe_misc_cost_Parker_2019_USD = pd.DataFrame(pipe_misc_cost_Parker_2019_USD,columns = ['Total misc cost ($)'])
    pipe_labor_cost_Parker_2019_USD = pd.DataFrame(pipe_labor_cost_Parker_2019_USD,columns = ['Total labor cost ($)'])
    pipe_row_cost_Parker_2019_USD = pd.DataFrame(pipe_row_cost_Parker_2019_USD,columns = ['Total ROW cost ($)'])
    
    pipe_network_costs_Parker_2019_USD = pipe_material_cost_Parker_2019_USD.join(pipe_misc_cost_Parker_2019_USD)
    pipe_network_costs_Parker_2019_USD = pipe_network_costs_Parker_2019_USD.join(pipe_labor_cost_Parker_2019_USD)
    pipe_network_costs_Parker_2019_USD = pipe_network_costs_Parker_2019_USD.join(pipe_row_cost_Parker_2019_USD)  
    
    pipe_network_cost_total_USD = sum(pipe_total_cost_USD)
    pipe_network_cost_total_Parker_2019_USD = sum(pipe_total_cost_Parker_2019_USD)
    
    return(pipe_network_cost_total_USD,pipe_network_cost_total_Parker_2019_USD,pipe_network_costs_USD,pipe_network_costs_Parker_2019_USD)