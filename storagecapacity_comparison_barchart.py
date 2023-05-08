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
sensitivity_directory = 'examples/H2_Analysis/Phase1B/Fin_sum_sens'
plot_directory = 'examples/H2_Analysis/Phase1B/Plots/'
plot_subdirectory = 'Sensitivity_barcharts/'

# Read in the summary data from the database
conn = sqlite3.connect(main_directory+'/Default_summary.db')
financial_summary  = pd.read_sql_query("SELECT * From Summary",conn)

conn.commit()
conn.close()

# Get storage capacity in metric tonnes
financial_summary['Hydrogen storage capacity (tonnes)']=financial_summary['Hydrogen storage capacity (kg)']/1000

# Order matrix by location
financial_summary.loc[financial_summary['Site']=='IN','Order']= 0
financial_summary.loc[financial_summary['Site']=='TX','Order']= 1
financial_summary.loc[financial_summary['Site']=='IA','Order']= 2
financial_summary.loc[financial_summary['Site']=='MS','Order']= 3
financial_summary.loc[financial_summary['Site']=='WY','Order']= 4

# Read in the sensitivity
conn = sqlite3.connect(sensitivity_directory+'/Default_summary.db')
financial_summary_sens  = pd.read_sql_query("SELECT * From Summary",conn)

conn.commit()
conn.close()

financial_summary_sens['Hydrogen storage capacity (tonnes)']=financial_summary_sens['Hydrogen storage capacity (kg)']/1000

# Order matrix by location
financial_summary_sens.loc[financial_summary_sens['Site']=='IN','Order']= 0
financial_summary_sens.loc[financial_summary_sens['Site']=='TX','Order']= 1
financial_summary_sens.loc[financial_summary_sens['Site']=='IA','Order']= 2
financial_summary_sens.loc[financial_summary_sens['Site']=='MS','Order']= 3
financial_summary_sens.loc[financial_summary_sens['Site']=='WY','Order']= 4

# Hybri cases
ren_cases=[
    'Wind',
    #'Wind+PV+bat'
    ]

# Policy option
policy_options = [
                'no-policy',
                #'base',
                #'max'
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

grid_case = 'off-grid'

#storage_multiplier='1.0'

control_method='Basic'
deg_string='deg-pen' #'no-deg'

save_plot = True

for year in years:
    for ren_case in ren_cases:
        for policy_option in policy_options:

            # Downselect to case of interest
            fin_sum_grid_year = financial_summary.loc[(financial_summary['Year']==year) & (financial_summary['Renewables case']==ren_case) & (financial_summary['Electrolysis case']==electrolysis_case)\
                &(financial_summary['Grid case']==grid_case)&(financial_summary['Policy Option']==policy_option)]
                
            #fin_sum_grid_year['Average stack \n life (yrs)']=fin_sum_grid_year['Average stack life (hrs)']/8760
            fin_sum_grid_year = fin_sum_grid_year.rename(columns = {'Average stack life (hrs)':'Average stack \n life (hrs)','Hydrogen storage capacity (tonnes)':'Hydrogen storage \n capacity (tonnes)','Steel price: Total ($/tonne)':'Steel price: \n Total ($/tonne)','Ammonia price: Total ($/kg)':'Ammonia price: \n Total ($/kg)'})
                                
            fin_sum_grid_year_sens = financial_summary_sens.loc[(financial_summary_sens['Year']==year) & (financial_summary_sens['Renewables case']==ren_case) & (financial_summary_sens['Electrolysis case']==electrolysis_case)\
                &(financial_summary_sens['Policy Option']==policy_option)&(financial_summary_sens['Degradation modeled?']==deg_string)]

            # Doing this here because I forgot to do it in the main code (just for storage duration)
            fin_sum_grid_year_sens.loc[fin_sum_grid_year_sens['Storage multiplier']=='1.5','Hydrogen storage duration (hr)']=fin_sum_grid_year_sens['Hydrogen storage duration (hr)']*1.5

            fin_sum_grid_year_sens = fin_sum_grid_year_sens.rename(columns = {'Average stack life (hrs)':'Average stack \n life (hrs)','Hydrogen storage capacity (tonnes)':'Hydrogen storage \n capacity (tonnes)','Steel price: Total ($/tonne)':'Steel price: \n Total ($/tonne)','Ammonia price: Total ($/kg)':'Ammonia price: \n Total ($/kg)'})
            
            # Combine and sort cases
            fin_sum_grid_year_combined = pd.concat([fin_sum_grid_year,fin_sum_grid_year_sens],join='inner',ignore_index=True) 

            fin_sum_grid_year_combined = fin_sum_grid_year_combined.sort_values(by='Order',ignore_index=True)

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

            title_desc='{}, {}, {}, {}, {}, {} Electrolyzer Control'.format(year,electrolysis_case,grid_string,ren_case,policy_string,control_method)
            filename = '{}_{}_EC-cost-{}_{}_{}_{}_{}-control'.format(year,electrolysis_case,electrolysis_cost_case,grid_case,ren_case,policy_option,control_method)

            # Order by site
            location_labels  = pd.unique(fin_sum_grid_year_combined['Site']).tolist()  

            resolution = 150

            # Plot LCOH, LCOS, and LCOA
            fin_df_plot_idx=['LCOH ($/kg)','Steel price: \n Total ($/tonne)','Ammonia price: \n Total ($/kg)','Hydrogen storage \n capacity (tonnes)']
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
                fin_sum_grid_year_dict['No extra storage'] = np.array(fin_sum_grid_year_combined.loc[fin_sum_grid_year_combined['Storage multiplier']=='1.0',var].values.tolist())
                fin_sum_grid_year_dict['50% extra storage'] = fin_sum_grid_year_combined.loc[fin_sum_grid_year_combined['Storage multiplier']=='1.5',var]
                color_dict['No extra storage']='darkgreen'
                color_dict['50% extra storage']='deepskyblue'
                hatch_dict['No extra storage']='//'
                hatch_dict['50% extra storage']='..'
                width=0.35
                multiplier=0

                x = np.arange(len(locations))

                for attribute,measurement in fin_sum_grid_year_dict.items():
                    offset=width * multiplier
                    rects=ax[axi].bar(x+offset,measurement,width*0.9,color=color_dict[attribute],hatch=hatch_dict[attribute],ec=linecolor,label=attribute)
                    if var != 'Steel price: \n Total ($/tonne)' and var !='Hydrogen storage \n capacity (tonnes)':
                        ax[axi].bar_label(rects,padding=3,fmt='%.2f')
                    else:
                        ax[axi].bar_label(rects,padding=3,fmt='%.0f')
                    multiplier +=1
                #ax[axi].set_xticks(x+width/2,locations)
                y_min = min(fin_sum_grid_year_combined[var].values.tolist())
                y_max = max(fin_sum_grid_year_combined[var].values.tolist())
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
                # if axi==0:
                #         for j in range(len(x)):
                #             ax[axi].text(x[j],1.3*y_max,hybrid_solar_battery_capacity_labels[j])

            ax[axi].set_xticks(x+width/2,location_labels,fontname = font,fontsize=axis_font_size)
            fig.align_ylabels()
            ax[0].legend(loc='upper right', ncols=1,prop = {'family':font,'size':legend_font_size})
            fig.tight_layout()
            #ax[0].legend(bbox_to_anchor=(0,1.02,1,0.2),loc='lower left',mode='expand',ncol=5)#,title='Lowest LCOH Case')    
            fig.suptitle(title_desc,fontname=font,fontsize=title_font_size)
            fig.tight_layout()
            []
            if save_plot:
                fig.savefig(plot_directory + plot_subdirectory+'storagecapacity_comparison_h2steelammoniaprices_'  + filename +  '.png',bbox_inches='tight')