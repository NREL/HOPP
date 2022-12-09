import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

font = 'Arial'
title_size = 10
axis_label_size = 10
legend_size = 6
tick_size = 10
resolution = 150


path = 'examples/H2_Analysis/financial_summary_results/'
output_path = 'examples/H2_Analysis/Plots/'

# FIG 1
#==============================================================================

# use_case=[['SMR', 'SMR', 'On-Grid', 'On-Grid', '']
#     ,['SMR Only', 'SMR + CCS', 'Grid Only',\
#     'Grid + Co-located Renewables','Off-Grid']
#     ]
# best_case = [753, 233, 175,150,150]
# worst_case = [943,429,1633,1258,346]
# diff = list(np.array(worst_case) - np.array(best_case))

# fig = go.Figure(data=[
#     go.Bar(name='Best Case Scenario (Year 2035)', x=use_case, y= best_case,
#     hovertemplate='GHG Emissions' + '<br>%{y} kg CO<sub>2</sub>e per MT of Steel'),
#     go.Bar(name='Worst Case Scenario (Year 2020)', x=use_case, y=diff, marker_pattern_shape="x",
#     hovertemplate='Additional GHG Emissions' + '<br>%{y} kg CO<sub>2</sub>e per MT of Steel')

# ])
# # Change the bar mode
# fig.update_annotations()
# fig.update_layout(barmode='stack',
#         title='Use Case Direct Comparison: Green House Gas Emmision Intensity',
#         yaxis_title="GHG Emission Intensity (kg CO<sub>2</sub>e/ MT of Steel)",
#         yaxis = dict(),
#         title_font=dict(size=25,
#                         family="Lato, sans-serif"),
#         font=dict(
#         family="Lato, sans-serif",
#         size=23,),
#         legend=dict(
#             yanchor="middle",
#             y=0.79,
#             xanchor="right",
#             # x=0.01
#             ),
#             bargap = 0.5,
#     paper_bgcolor='rgba(0,0,0,0)',
#     plot_bgcolor='rgba(0,0,0,0)',

        
#     )
# fig.update_xaxes(showline=True, linewidth=2, linecolor='black', tickangle=0)
# fig.data[0].marker.color = ('darkslateblue','darkslateblue','darkgoldenrod','darkorange','mediumseagreen')
# fig.data[1].marker.color = ('slateblue','slateblue','gold','orange','darkseagreen')
# fig.add_hline(y=791,line_width=3, line_dash="dash")

# fig.add_annotation(x=1, y=791,
#             text="U.S. Average",
#             showarrow=True,
#             arrowhead=1,
#             yshift=10)

# fig.show()

# fig.write_html(output_path+"use_case_GHG_steel.html")


best_case = np.array([9.6,1.8,0.87,0.479,0.479,0.479])
#mid_case = np.array([9.7,2.1,10.2,0.479,0.479,0.479])
worst_case = np.array([9.7,2.29,19.9,14.3,0.569,0.569])
labels = ['SMR','SMR + CCS', 'Grid Only','Grid + Renewables','Off Grid, Centralized EC', 'Off Grid, Distributed EC']
scenario_title = 'Worst and Best GHG Emission Intensity'

width = 0.5
#fig, ax = plt.subplots()
fig, ax = plt.subplots(1,1,figsize=(4.8,3.6), dpi= resolution)
ax.bar(labels,best_case,width,label='Best GHG case: TX, 2035',edgecolor='sandybrown',color='darkorange')
#barbottom=best_case
#ax.bar(labels,mid_case,width,label='GHG case, TX, 2030',edgecolor ='c',hatch='//////',alpha=0.5)
#barbottom=barbottom+mid_case
ax.bar(labels,worst_case,width,label='Worst GHG case, 2020',edgecolor ='lavender',hatch='++++',alpha=0.2)
#barbottom=barbottom+worst_case
#ax.axhline(y=barbottom[0], color='k', linestyle='--',linewidth=1)

# Decorations
ax.set_title(scenario_title, fontsize=title_size)

ax.set_ylabel('GHG (kg CO2e/kg H2)', fontname = font, fontsize = axis_label_size)
#ax.set_xlabel('Scenario', fontname = font, fontsize = axis_label_size)
ax.legend(fontsize = legend_size, ncol = 2, prop = {'family':'Arial','size':7})
#max_y = np.max(barbottom)
ax.set_ylim([0,26])
ax.tick_params(axis = 'y',labelsize = 7,direction = 'in')
ax.tick_params(axis = 'x',labelsize = 7,direction = 'in',rotation=45)
#ax2 = ax.twinx()
#ax2.set_ylim([0,10])
#plt.xlim(x[0], x[-1])
plt.tight_layout()

# FIG 2: Steel
#==============================================================================

scope_1 = np.array([214,214,214,214,214,214])
scope_2 = np.array([22.9,22.9,22.9,22.9,22.9,22.9])
scope_3 = np.array([843.5,343.4,878.5,238.1,238.1,238.1])
labels = ['SMR','SMR + CCS', 'Grid Only','Grid + Renewables','Off Grid, Centralized EC', 'Off Grid, Distributed EC']
scenario_title = 'Steelmaking, TX, 2030'

width = 0.5
#fig, ax = plt.subplots()
fig, ax = plt.subplots(1,1,figsize=(4.8,3.6), dpi= resolution)
ax.bar(labels,scope_1,width,label='GHG Scope 1 Emissions',edgecolor='steelblue',color='deepskyblue')
barbottom=scope_1
ax.bar(labels,scope_2,width,bottom=barbottom,label = 'GHG Scope 2 Emissions',edgecolor='dimgray',color='dimgrey')
barbottom=barbottom+scope_2
ax.bar(labels,scope_3,width,bottom=barbottom,label='GHG Scope 3 Emissions',edgecolor='black',color='navy')
barbottom=barbottom+scope_3
ax.axhline(y=barbottom[0], color='k', linestyle='--',linewidth=1)

# Decorations
ax.set_title(scenario_title, fontsize=title_size)

ax.set_ylabel('GHG (kg CO2e/MT steel)', fontname = font, fontsize = axis_label_size)
#ax.set_xlabel('Scenario', fontname = font, fontsize = axis_label_size)
ax.legend(fontsize = legend_size, ncol = 2, prop = {'family':'Arial','size':7})
max_y = np.max(barbottom)
ax.set_ylim([0,1.3*max_y])
ax.tick_params(axis = 'y',labelsize = 7,direction = 'in')
ax.tick_params(axis = 'x',labelsize = 7,direction = 'in',rotation=45)
#ax2 = ax.twinx()
#ax2.set_ylim([0,10])
#plt.xlim(x[0], x[-1])
plt.tight_layout()

# FIG 3: Ammonia
#==============================================================================

scope_1 = np.array([0.5,0.5,0.5,0.5,0.5,0.5])
scope_2 = np.array([0.022,0.022,0.022,0.022,0.022,0.022])
scope_3 = np.array([2.026,0.51,2.133,0.191,0.191,0.191])
labels = ['SMR','SMR + CCS', 'Grid Only','Grid + Renewables','Off Grid, Centralized EC', 'Off Grid, Distributed EC']
scenario_title = 'Ammonia, TX, 2030'

width = 0.5
#fig, ax = plt.subplots()
fig, ax = plt.subplots(1,1,figsize=(4.8,3.6), dpi= resolution)
ax.bar(labels,scope_1,width,label='GHG Scope 1 Emissions',edgecolor='steelblue',color='deepskyblue')
barbottom=scope_1
ax.bar(labels,scope_2,width,bottom=barbottom,label = 'GHG Scope 2 Emissions',edgecolor='dimgray',color='dimgrey')
barbottom=barbottom+scope_2
ax.bar(labels,scope_3,width,bottom=barbottom,label='GHG Scope 3 Emissions',edgecolor='black',color='navy')
barbottom=barbottom+scope_3
ax.axhline(y=barbottom[0], color='k', linestyle='--',linewidth=1)

# Decorations
ax.set_title(scenario_title, fontsize=title_size)

ax.set_ylabel('GHG (kg CO2e/kg NH3)', fontname = font, fontsize = axis_label_size)
#ax.set_xlabel('Scenario', fontname = font, fontsize = axis_label_size)
ax.legend(fontsize = legend_size, ncol = 2, prop = {'family':'Arial','size':7})
max_y = np.max(barbottom)
ax.set_ylim([0,1.3*max_y])
ax.tick_params(axis = 'y',labelsize = 7,direction = 'in')
ax.tick_params(axis = 'x',labelsize = 7,direction = 'in',rotation=45)
#ax2 = ax.twinx()
#ax2.set_ylim([0,10])
#plt.xlim(x[0], x[-1])
plt.tight_layout()