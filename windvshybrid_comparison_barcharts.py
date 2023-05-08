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

smr_directory = 'examples/H2_Analysis/Phase1B/SMR_fin_summary'

retail_string = 'retail-flat'
plot_subdirectory = 'Wind_vs_hybrid_barcharts/'

# Read in the summary data from the database
conn = sqlite3.connect(main_directory+'/Default_summary.db')
financial_summary  = pd.read_sql_query("SELECT * From Summary",conn)

conn.commit()
conn.close()

# Get storage capacity in metric tonnes
financial_summary['Hydrogen storage capacity (tonnes)']=financial_summary['Hydrogen storage capacity (kg)']/1000

# Read in the summary data from the smr case database
conn = sqlite3.connect(smr_directory+'/Default_summary.db')
financial_summary_smr  = pd.read_sql_query("SELECT * From Summary",conn)

conn.commit()
conn.close()

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
            #'hybrid-grid-'+retail_string
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
            fin_sum_grid_year = fin_sum_grid_year.rename(columns = {'Hydrogen storage capacity (tonnes)':'Hydrogen storage \n capacity (tonnes)','Steel price: Total ($/tonne)':'Steel price: \n Total ($/tonne)','Ammonia price: Total ($/kg)':'Ammonia price: \n Total ($/kg)'})

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

            fin_sum_smr_year = financial_summary_smr.loc[(financial_summary_smr['Year']==year)&(financial_summary_smr['Policy Option']=='no policy')&(financial_summary_smr['CCS Case']=='woCCS')]
            smr_lcoh_avg = fin_sum_smr_year['LCOH ($/kg)'].mean()
            smr_lcos_avg = fin_sum_smr_year['Steel price: Total ($/tonne)'].mean()
            smr_lcoa_avg = fin_sum_smr_year['Ammonia price: Total ($/kg)'].mean()

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
                fin_df_plot_idx=['LCOH ($/kg)','Electrolyzer CF (-)','Average stack \n life (yrs)','Hydrogen storage \n capacity (tonnes)']
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
                        if var != 'Hydrogen storage \n capacity (tonnes)':
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
                    error = np.zeros(len(locations))
                    if var == 'LCOH ($/kg)':
                        ax[axi].axhline(y=smr_lcoh_avg, color='k', linestyle='--',linewidth=1.5)
                        ax[axi].text(-0.35,smr_lcoh_avg+0.05,'SMR')
                    elif var == 'Steel price: \n Total ($/tonne)':
                        ax[axi].axhline(y=smr_lcos_avg, color='k', linestyle='--',linewidth=1.5)
                        ax[axi].text(-0.35,smr_lcos_avg+20,'SMR')
                    elif var == 'Ammonia price: \n Total ($/kg)':
                        ax[axi].axhline(y=smr_lcoa_avg, color='k', linestyle='--',linewidth=1.5)
                        ax[axi].text(-0.35,smr_lcoa_avg+0.02,'SMR')
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
                        ax[axi].set_ylim([-3,1.25*y_max])
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
