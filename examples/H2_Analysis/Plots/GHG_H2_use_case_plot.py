import plotly.graph_objects as go

path = 'examples/H2_Analysis/financial_summary_results/'
output_path = 'examples/H2_Analysis/Plots/'

use_case=[['SMR', 'SMR', 'On-Grid', 'On-Grid', '']
    ,['SMR Only', 'SMR + CCS', 'Grid Only',\
    'Grid + Co-located Renewables','Off-Grid']
    ]
with_policy = [34, 22, 25,32,4]

fig = go.Figure(data=[
    go.Bar(name='GHG', x=use_case, y= with_policy,
    hovertemplate='GHG Emissions' + '<br>%{y} kg CO<sub>2</sub>e per kg H<sub>2</sub>'),

])
# Change the bar mode
fig.update_annotations()
fig.update_layout(barmode='stack',
        title='Use Case Direct Comparison: Green House Gas Emmision Intensity',
        yaxis_title="GHG Emission Intensity (kg CO<sub>2</sub>e/ kg H<sub>2</sub>)",
        yaxis = dict(),
        title_font=dict(size=25,
                        family="Lato, sans-serif"),
        font=dict(
        family="Lato, sans-serif",
        size=18,),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
            ),
            bargap = 0.5,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',

        
    )
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', tickangle=0)
fig.data[0].marker.color = ('darkslateblue','darkslateblue','darkgoldenrod','darkorange','mediumseagreen')
# fig.data[1].marker.color = ('slateblue','slateblue','gold','orange','darkseagreen')
fig.add_hline(y=4,line_width=3, line_dash="dash")
fig.show()

fig.write_html(output_path+"use_case_GHG_H2.html")