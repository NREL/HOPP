import plotly.graph_objects as go
import numpy as np

path = 'examples/H2_Analysis/financial_summary_results/'

use_case=[['SMR', 'SMR', 'On-Grid', 'On-Grid', '']
    ,['SMR Only', 'SMR + CCS', 'Grid Only',\
    'Grid + Co-located Renewables','Off-Grid']
    ]
with_policy = [1.27, 1.11, 3.3668869,2.07787699,1.26683139]
without_policy = [1.27, 1.79, 3.5668869,2.27787699,2.26683139]
diff = list(np.array(without_policy) - np.array(with_policy))
total_without_policy = with_policy + without_policy
fig = go.Figure(data=[
    go.Bar(name='With Policy', x=use_case, y= with_policy,
    hovertemplate='With Policy' + '<br>$%{y} per kg of Hydrogen'),
    go.Bar(name='Without Policy', x=use_case, y= diff, marker_pattern_shape="x",
            hovertemplate='Without Policy' + '<br>Additional $%{y} per kg of Hydrogen'),

])
# Change the bar mode
fig.update_annotations()
fig.update_layout(barmode='stack',
        title='Use Case Direct Comparison: Levelized Cost of Hydrogen',
        yaxis_title="$/kg of Hydrogen ",
        yaxis = dict(range=[0,4]),
        title_font=dict(size=25,
                        family="Lato, sans-serif"),
        font=dict(
        family="Lato, sans-serif",
        size=23,),
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
fig.data[0].marker.color = ('darkslateblue','darkslateblue','darkgoldenrod','darkorange','darkgreen')
fig.data[1].marker.color = ('slateblue','slateblue','gold','orange','green')
fig.add_hline(y=2,line_width=3, line_dash="dash")
fig.add_annotation(x=.25, y=2,
            text="Current Cost of Hydrogen",
            showarrow=True,
            arrowhead=1,
            yshift=10)
fig.show()

fig.write_html(path+"use_case_direct_comparision_h2.html")