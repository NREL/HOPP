import plotly.graph_objects as go
import numpy as np

path = 'examples/H2_Analysis/financial_summary_results/'
output_path = 'examples/H2_Analysis/Plots/'

use_case=[['SMR', 'SMR', 'On-Grid', 'On-Grid', '']
    ,['SMR Only', 'SMR + CCS', 'Grid Only',\
    'Grid + Co-located Renewables','Off-Grid']
    ]
best_case = [2.435, 0.859, 0.682,0.606,0.606]
worst_case = [2.618,1.05,4.704,3.572,0.807]
diff = list(np.array(worst_case) - np.array(best_case))

fig = go.Figure(data=[
    go.Bar(name='Best Case Scenario', x=use_case, y= best_case,
    hovertemplate='GHG Emissions' + '<br>%{y} kg CO<sub>2</sub>e per kg NH<sub>3</sub>'),
     go.Bar(name='Worst Case Scenario', x=use_case, y=diff, marker_pattern_shape="x",
    hovertemplate='Additional GHG Emissions' + '<br>%{y} kg CO<sub>2</sub>e per kg NH<sub>3</sub>')

])
# Change the bar mode
fig.update_annotations()
fig.update_layout(barmode='stack',
        title='Use Case Direct Comparison: Green House Gas Emmision Intensity',
        yaxis_title="GHG Emission Intensity (kg CO<sub>2</sub>e/ kg NH<sub>3</sub>)",
        yaxis = dict(),
        title_font=dict(size=25,
                        family="Lato, sans-serif"),
        font=dict(
        family="Lato, sans-serif",
        size=23,),
        legend=dict(
            yanchor="middle",
            y=0.79,
            xanchor="right",
            # x=0.01
            ),
            bargap = 0.5,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',

        
    )
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', tickangle=0)
fig.data[0].marker.color = ('darkslateblue','darkslateblue','darkgoldenrod','darkorange','mediumseagreen')
fig.data[1].marker.color = ('slateblue','slateblue','gold','orange','darkseagreen')
fig.add_hline(y=2.6,line_width=3, line_dash="dash")
fig.add_annotation(x=1, y=2.6,
            text="U.S. Average",
            showarrow=True,
            arrowhead=1,
            yshift=10)
fig.show()

fig.write_html(output_path+"use_case_GHG_NH3.html")