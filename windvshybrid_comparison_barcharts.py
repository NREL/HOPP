import os
import sys
sys.path.append('')
from dotenv import load_dotenv
import pandas as pd

import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import sqlite3

parent_path = os.path.abspath('')

# Initialization and Global Settings
#Specify directory name
main_directory = 'examples/H2_Analysis/Phase1B/Fin_sum'
plot_directory = 'examples/H2_Analysis/Phase1B/Plots/'

retail_string = 'retail-flat'
plot_subdirectory = 'Wind_vs_hybrid_barcharts/'

# Read in the summary data from the database
conn = sqlite3.connect(main_directory+'/Default_summary.db')
financial_summary  = pd.read_sql_query("SELECT * From Summary",conn)

conn.commit()
conn.close()

# Fix steel and ammonia prices
financial_summary.loc[financial_summary['LCOH ($/kg)']<0,'Steel price: Hydrogen ($/tonne)']=-financial_summary['Steel price: Hydrogen ($/tonne)']
financial_summary.loc[financial_summary['LCOH ($/kg)']<0,'Steel price: Total ($/tonne)'] = financial_summary['Steel price: Total ($/tonne)']+2*financial_summary['Steel price: Hydrogen ($/tonne)']

financial_summary.loc[financial_summary['LCOH ($/kg)']<0,'Ammonia price: Hydrogen ($/kg)']=-financial_summary['Ammonia price: Hydrogen ($/kg)']
financial_summary.loc[financial_summary['LCOH ($/kg)']<0,'Ammonia price: Total ($/kg)'] = financial_summary['Ammonia price: Total ($/kg)']+2*financial_summary['Ammonia price: Hydrogen ($/kg)']

# Add property tax and insurance to steel and ammonia prices
financial_summary['Steel price: Property tax and insurance ($/tonne)'] = 0.02*financial_summary['Steel Plant Total CAPEX ($)']/financial_summary['Steel annual production (tonne/year)']
financial_summary['Ammonia price: Property tax and insurance ($/kg)'] = 0.02*financial_summary['Ammonia Plant Total CAPEX ($)']/financial_summary['Ammonia annual production (kg/year)']
financial_summary['Steel price: Remaining Financial ($/tonne)'] = financial_summary['Steel price: Remaining Financial ($/tonne)'] + financial_summary['Steel price: Property tax and insurance ($/tonne)']
financial_summary['Ammonia price: Remaining Financial ($/kg)'] = financial_summary['Ammonia price: Remaining Financial ($/kg)'] + financial_summary['Ammonia price: Property tax and insurance ($/kg)']
financial_summary['Steel price: Total ($/tonne)'] = financial_summary['Steel price: Total ($/tonne)'] + financial_summary['Steel price: Property tax and insurance ($/tonne)']
financial_summary['Ammonia price: Total ($/kg)'] = financial_summary['Ammonia price: Total ($/kg)'] + financial_summary['Ammonia price: Property tax and insurance ($/kg)']

# Order matrix by location
financial_summary.loc[financial_summary['Site']=='IN','Order']= 0
financial_summary.loc[financial_summary['Site']=='TX','Order']= 1
financial_summary.loc[financial_summary['Site']=='IA','Order']= 2
financial_summary.loc[financial_summary['Site']=='MS','Order']= 3
financial_summary.loc[financial_summary['Site']=='WY','Order']= 4

# Narrow down to retail price of interest
if retail_string == 'retail-flat':
    financial_summary = financial_summary.loc[(financial_summary['Grid case']!='grid-only-wholesale') & (financial_summary['Grid case']!='hybrid-grid-wholesale') & (financial_summary['Grid case'] != 'grid-only-retail-flat')]
elif retail_string == 'wholesale':
    financial_summary = financial_summary.loc[(financial_summary['Grid Case']!='grid-only-retail-flat') & (financial_summary['Grid Case']!='hybrid-grid-retail-flat')& (financial_summary['Grid case'] != 'grid-only-wholesale')]


# Hybri cases
ren_cases=['Wind','Wind+PV+bat']

# Policy option
policy_options = [
                'no-policy',
                #'base',
                'max'
                ]

# Electrolysis case 
electrolysis_case = 'Centralized'

electrolysis_cost_case = 'low'

locations = [
            'IN',
            'TX',
            'IA',
            'MS',
            'WY'
             ]


years = [
    '2020',
    '2025',
    '2030',
    '2035'
    ]

retail_string = 'retail-flat'

grid_cases = [
            'off-grid',
            'hybrid-grid-'+retail_string
            ]

control_method='Basic'
deg_string='deg-pen' #'no-deg'

save_plot = True

for year in years:
    for grid_case in grid_cases:
        for policy_option in policy_options:



            # Downselect to case of interest
            fin_sum_grid_year = financial_summary.loc[(financial_summary['Year']==year) & (financial_summary['Grid case']==grid_case) & (financial_summary['Electrolysis case']==electrolysis_case)&(financial_summary['Policy Option']==policy_option)]
            fin_sum_grid_year['Average stack \n life (yrs)']=fin_sum_grid_year['Average stack life (hrs)']/8760
            fin_sum_grid_year = fin_sum_grid_year.rename(columns = {'Hydrogen storage duration (hr)':'Hydrogen storage \n duration (hr)','Steel price: Total ($/tonne)':'Steel price: \n Total ($/tonne)','Ammonia price: Total ($/kg)':'Ammonia price: \n Total ($/kg)'})

            fin_sum_grid_year = fin_sum_grid_year.sort_values(by='Order',ignore_index=True)

            if grid_case == 'off-grid':
                grid_string='Off-Grid'
            else:
                grid_string='Hybrid-Grid'

            if policy_option=='no-policy':
                policy_string = 'No policy'
            elif policy_option =='base':
                policy_string='Base policy'
            elif policy_option =='max':
                policy_string='Max policy'

            title_desc='{}, {}, {}, {} Electrolyzer Control'.format(year,grid_string,policy_string,control_method)
            filename = '{}_EC-cost-{}_{}_{}_{}-control'.format(year,electrolysis_cost_case,grid_case,policy_option,control_method)

            # Order by site
            location_labels  = pd.unique(fin_sum_grid_year['Site']).tolist()  

            # Add strings with solar and battery capacity to the top
            fin_sum_grid_year['Solar capacity (MW)'] = fin_sum_grid_year['Solar capacity (MW)'].astype('int').astype('str')
            fin_sum_grid_year['Battery storage capacity (MW)'] = fin_sum_grid_year['Battery storage capacity (MW)'].astype('int').astype('str')
            fin_sum_grid_year['Battery storage duration (hr)'] = fin_sum_grid_year['Battery storage duration (hr)'].astype('int').astype('str')
            hybrid_solar_capacity = fin_sum_grid_year.loc[fin_sum_grid_year['Renewables case']=='Wind+PV+bat','Solar capacity (MW)'].values.tolist()
            hybrid_battery_capacity = fin_sum_grid_year.loc[fin_sum_grid_year['Renewables case']=='Wind+PV+bat','Battery storage capacity (MW)'].values.tolist()
            hybrid_battery_duration = fin_sum_grid_year.loc[fin_sum_grid_year['Renewables case']=='Wind+PV+bat','Battery storage duration (hr)'].values.tolist()
            hybrid_solar_capacity_labels = []
            hybrid_solar_battery_capacity_labels=[]
            for j in range(len(location_labels)):
                hybrid_solar_battery_capacity_labels.append('PV Cap: '+hybrid_solar_capacity[j]+' MW \nBat Cap: '+hybrid_battery_capacity[j] + ' MW \nBat dur: '+hybrid_battery_duration[j]+' hr')

            resolution = 150

            # Plot LCOH, Electrolyzer capacity factor, Average stack life, and storage duration
            if grid_case == 'off-grid':
                fin_df_plot_idx=['LCOH ($/kg)','Electrolyzer CF (-)','Average stack \n life (yrs)','Hydrogen storage \n duration (hr)']
                #fin_df_desc_idx=['Wind capacity (MW)','Solar capacity (MW)','Battery storage capacity (MW)','Battery storage duration (hr)']
                #plant_desc='PV: {}MW Battery: {}MW {}hr'.format(fin_df.loc['Solar capacity (MW)'].values[0],fin_df.loc['Battery storage capacity (MW)'].values[0],fin_df.loc['Battery storage duration (hr)'].values[0])
                font = 'Arial'
                title_font_size=20
                axis_font_size=14
                legend_font_size=12

                fig,ax=plt.subplots(len(fin_df_plot_idx),1,sharex=True,dpi= resolution)
                fig.tight_layout()
                fig.set_figwidth(12)
                fig.set_figheight(8)    
                barwidth=0.35
                xsep=0.4
                linecolor='white'

                fin_sum_grid_year_dict = {}
                color_dict = {}
                hatch_dict = {}

                for axi,var in enumerate(fin_df_plot_idx):
                    fin_sum_grid_year_dict['Wind'] = np.array(fin_sum_grid_year.loc[fin_sum_grid_year['Renewables case']=='Wind',var].values.tolist())
                    fin_sum_grid_year_dict['Wind+PV+bat'] = fin_sum_grid_year.loc[fin_sum_grid_year['Renewables case']=='Wind+PV+bat',var]
                    color_dict['Wind']='mediumblue'
                    color_dict['Wind+PV+bat']='darkred'
                    hatch_dict['Wind']=None
                    hatch_dict['Wind+PV+bat']='\/'
                    width=0.35
                    multiplier=0

                    x = np.arange(len(locations))

                    for attribute,measurement in fin_sum_grid_year_dict.items():
                        offset=width * multiplier
                        rects=ax[axi].bar(x+offset,measurement,width*0.9,color=color_dict[attribute],hatch=hatch_dict[attribute],ec=linecolor,label=attribute)
                        if var != 'Hydrogen storage \n duration (hr)':
                            ax[axi].bar_label(rects,padding=3,fmt='%.2f')
                        else:
                            ax[axi].bar_label(rects,padding=3,fmt='%.0f')
                        multiplier +=1
                        

                    #ax[0].text(1,0,hybrid_solar_capacity_labels)        
                    #ax[axi].set_xticks(x+width/2,locations)
                    y_min = min(fin_sum_grid_year[var].values.tolist())
                    y_max = max(fin_sum_grid_year[var].values.tolist())
                    ax[axi].set_ylabel(var,fontname=font,fontsize=axis_font_size)
                    ax[axi].axhline(y=0, color='k', linestyle='-',linewidth=1.5)
                    if y_min > 0:
                        ax[axi].set_ylim([0,1.25*y_max])
                    else:
                        ax[axi].set_ylim([-3,1.25*y_max])

                    if axi==0:
                        for j in range(len(x)):
                            ax[axi].text(x[j],1.3*y_max,hybrid_solar_battery_capacity_labels[j])

                ax[axi].set_xticks(x+width/2,location_labels,fontname = font,fontsize=axis_font_size)
                fig.align_ylabels()
                ax[0].legend(loc='upper right', ncols=2,prop = {'family':font,'size':legend_font_size})
                

                #ax[0].legend(bbox_to_anchor=(0,1.02,1,0.2),loc='lower left',mode='expand',ncol=5)#,title='Lowest LCOH Case')    
                fig.suptitle(title_desc,fontname=font,fontsize=title_font_size)
                fig.tight_layout()
                #plt.show()
                if save_plot:
                    fig.savefig(plot_directory + plot_subdirectory+'wind_vs_hybrid_h2prodstats_'  + filename +  '.png',bbox_inches='tight')
                #[]

            # Plot LCOH, LCOS, and LCOA
            fin_df_plot_idx=['LCOH ($/kg)','Steel price: \n Total ($/tonne)','Ammonia price: \n Total ($/kg)']
            font = 'Arial'
            title_font_size=20
            axis_font_size=14
            legend_font_size=12

            fig,ax=plt.subplots(len(fin_df_plot_idx),1,sharex=True,dpi= resolution)
            fig.tight_layout()
            fig.set_figwidth(12)
            fig.set_figheight(8)    
            barwidth=0.35
            xsep=0.4
            linecolor='white'

            fin_sum_grid_year_dict = {}
            color_dict = {}
            hatch_dict = {}

            for axi,var in enumerate(fin_df_plot_idx):
                fin_sum_grid_year_dict['Wind'] = np.array(fin_sum_grid_year.loc[fin_sum_grid_year['Renewables case']=='Wind',var].values.tolist())
                fin_sum_grid_year_dict['Wind+PV+bat'] = fin_sum_grid_year.loc[fin_sum_grid_year['Renewables case']=='Wind+PV+bat',var]
                color_dict['Wind']='mediumblue'
                color_dict['Wind+PV+bat']='darkred'
                hatch_dict['Wind']=None
                hatch_dict['Wind+PV+bat']='\/'
                width=0.35
                multiplier=0

                x = np.arange(len(locations))

                for attribute,measurement in fin_sum_grid_year_dict.items():
                    offset=width * multiplier
                    rects=ax[axi].bar(x+offset,measurement,width*0.9,color=color_dict[attribute],hatch=hatch_dict[attribute],ec=linecolor,label=attribute)
                    if var != 'Steel price: \n Total ($/tonne)':
                        ax[axi].bar_label(rects,padding=3,fmt='%.2f')
                    else:
                        ax[axi].bar_label(rects,padding=3,fmt='%.0f')
                    multiplier +=1
                #ax[axi].set_xticks(x+width/2,locations)
                y_min = min(fin_sum_grid_year[var].values.tolist())
                y_max = max(fin_sum_grid_year[var].values.tolist())
                ax[axi].set_ylabel(var,fontname=font,fontsize=axis_font_size)
                ax[axi].axhline(y=0, color='k', linestyle='-',linewidth=1.5)
                if y_min > 0:
                    ax[axi].set_ylim([0,1.25*y_max])
                else:
                    if var=='LCOH ($/kg)':
                        ax[axi].set_ylim([-2,1.25*y_max])
                    elif var=='Ammonia price: \n Total ($/kg)':
                        ax[axi].set_ylim([-1,1.25*y_max])
                    else:
                        ax[axi].est_ylim([2*y_min,1.25*y_max])
                if axi==0:
                        for j in range(len(x)):
                            ax[axi].text(x[j],1.3*y_max,hybrid_solar_battery_capacity_labels[j])

            ax[axi].set_xticks(x+width/2,location_labels,fontname = font,fontsize=axis_font_size)
            fig.align_ylabels()
            ax[0].legend(loc='upper right', ncols=2,prop = {'family':font,'size':legend_font_size})
            fig.tight_layout()
            #ax[0].legend(bbox_to_anchor=(0,1.02,1,0.2),loc='lower left',mode='expand',ncol=5)#,title='Lowest LCOH Case')    
            fig.suptitle(title_desc,fontname=font,fontsize=title_font_size)
            fig.tight_layout()
            #plt.show()
            if save_plot:
                fig.savefig(plot_directory + plot_subdirectory+'wind_vs_hybrid_h2steelammoniaprices_'  + filename +  '.png',bbox_inches='tight')
            []









#         for si,state in enumerate(locations):
#             for renbat_string in ren_cases:

#                 fin_sum_loc_case = fin_sum_grid_year.loc[(fin_sum_grid_year['Site']==state)&(fin_sum_grid_year['Renewables case']==renbat_string)]

#                 if renbat_string=='Wind+PV+bat':
#                     bar_color='red'
#                     hatch_style='\/'
#                     x_pos = si+xsep
#                     #state_info[state]={'Wind+PV+bat':fin_sum_loc_case.loc[fin_df_plot_idx]}
#                     plant_desc='PV: {}MW Battery: {}MW {}hr'.format(fin_sum_loc_case.loc['Solar capacity (MW)'].values[0],fin_sum_loc_case.loc['Battery storage capacity (MW)'].values[0],fin_sum_loc_case.loc['Battery storage duration (hr)'].values[0])
#                 else:
#                     bar_color='blue'
#                     hatch_style=None
#                     x_pos=si
#                     #state_info[state]={'Wind':fin_sum_loc_case.loc[fin_df_plot_idx]}
#                     plant_desc='Wind Farm {} MW'.format(round(fin_sum_loc_case.loc['Wind capacity (MW)'].values[0],2))
#                 for axi,var in enumerate(fin_df_plot_idx):
#                     ax[axi].bar(x_pos,fin_sum_loc_case.loc[var].values[0],color=bar_color,width=barwidth,hatch=hatch_style,ec=linecolor,label=plant_desc)
#                     ax[axi].set_ylabel(var,fontname = font, fontsize = axis_label_size)

#         ax[axi].set_xticks(np.arange(0,5,1)+(xsep/2),locations,fontsize=14)
#         fig.align_ylabels()
#         fig.tight_layout()
#         ax[0].legend(bbox_to_anchor=(0,1.02,1,0.2),loc='lower left',mode='expand',ncol=5)#,title='Lowest LCOH Case')    
#         fig.suptitle(title_desc,fontname=font,fontsize=title_size)
#         fig.tight_layout()
#         []

# main_dir=parent_path + '/examples/H2_Analysis/Phase1B/'
# pv_dir = parent_path + '/examples/H2_Analysis/results/PV_Parametric_Sweep/'
# energy_subdir = 'Energy_profiles/Energy'
# finan_subdir = 'Fin_summary/Fin_sum'
# profast_subdir = 'ProFAST_price/H2_PF_PB'
# control_methods=['basic','optimize']
# electrolysis_cases=['Distributed','Centralized']
# grid_connection_cases=['off-grid','grid-only','hybrid-grid']
# sim_tools=['pysam','floris']

# states=['IN','TX','IA','MS','WY']
# turb_ratings=[6,6,8,4,6]

# atb_year=2030
# electrolysis_case='Distributed'
# grid_connection_scenario='off-grid'
# control_method='basic'
# grid_connection_scenario='off-grid'
# policy_opt='no-policy'
# sim_tool='pysam'
# deg_string='no-deg' #'no-deg'


# ren_cases=['Wind','Wind+PV+bat']

# fin_df_desc_idx=['Wind capacity (MW)','Solar capacity (MW)','Battery storage capacity (MW)','Battery storage duration (hr)']
# fin_df_title_desc_idx=['Electrolyzer Capacity (MW)','ATB Year']
# fin_df_plot_idx=['Electricity CF (-)','LCOH ($/kg)','Electrolyzer CF (-)']
# title_desc='{} {} {} \n{} Electrolyzer Control ({})'.format(atb_year,electrolysis_case,grid_connection_cases,control_method,deg_string)
# state_info={}
# fig,ax=plt.subplots(len(fin_df_plot_idx),1,sharex=True)
# fig.tight_layout()
# fig.set_figwidth(12)
# fig.set_figheight(8)    
# barwidth=0.35
# xsep=0.4
# linecolor='white'
# save_plot=False
# save_plot_folder='/Users/egrant/Desktop/HOPP-GIT/HOPP/examples/H2_Analysis/Phase1B/plots/'

# for si,state in enumerate(states):
#     for renbat_string in ren_cases:
#         file_root='_{}_{}_{}MW_{}_EC-cost-mid_{}_{}_{}_{}_{}-pen_stack-op-{}_SM_1.0.csv'.format(state,atb_year,turb_ratings[si],electrolysis_case,policy_opt,grid_connection_scenario,renbat_string,sim_tool,deg_string,control_method)
#         fin_df=pd.read_csv(main_dir + finan_subdir + file_root,index_col='Unnamed: 0')
#         #e_df=pd.read_csv(main_dir + energy_subdir + file_root,index_col='Unnamed: 0')
#         #pf_df=pd.read_csv(main_dir + profast_subdir + file_root,index_col='Unnamed: 0')
        
#         if renbat_string=='Wind+PV+bat':
#             bar_color='red'
#             hatch_style='\/'
#             x_pos = si+xsep
#             state_info[state]={'Wind+PV+bat':fin_df.loc[fin_df_plot_idx]}
#             plant_desc='PV: {}MW Battery: {}MW {}hr'.format(fin_df.loc['Solar capacity (MW)'].values[0],fin_df.loc['Battery storage capacity (MW)'].values[0],fin_df.loc['Battery storage duration (hr)'].values[0])
#         else:
#             bar_color='blue'
#             hatch_style=None
#             x_pos=si
#             state_info[state]={'Wind':fin_df.loc[fin_df_plot_idx]}
#             plant_desc='Wind Farm {} MW'.format(round(fin_df.loc['Wind capacity (MW)'].values[0],2))
#         for axi,var in enumerate(fin_df_plot_idx):
#             ax[axi].bar(x_pos,fin_df.loc[var].values[0],color=bar_color,width=barwidth,hatch=hatch_style,ec=linecolor,label=plant_desc)
#             ax[axi].set_ylabel(var,fontsize=14)
#         []
# ax[axi].set_xticks(np.arange(0,5,1)+(xsep/2),states,fontsize=14)
# fig.align_ylabels()
# fig.tight_layout()
# ax[0].legend(bbox_to_anchor=(0,1.02,1,0.2),loc='lower left',mode='expand',ncol=5)#,title='Lowest LCOH Case')    
# fig.suptitle(title_desc)
# fig.tight_layout()
# if save_plot:
#     fig.savefig(save_plot_folder  + title_desc +  '.pdf',bbox_inches='tight')
    
    
# # Fin_sum_TX_2030_6MW_Distributed_EC-cost-mid_no-policy_off-grid_Wind_pysam_no-deg-pen_stack-op-optimize_SM_1.0.csv
# # Energy_TX_2030_6MW_Distributed_EC-cost-mid_no-policy_off-grid_Wind_pysam_no-deg-pen_stack-op-optimize_SM_1.0.csv
