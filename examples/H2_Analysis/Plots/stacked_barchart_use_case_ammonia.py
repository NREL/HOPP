import plotly.graph_objects as go
import numpy as np

path = 'examples/H2_Analysis/financial_summary_results/'

use_case=[['SMR', 'SMR', 'On-Grid', 'On-Grid', '', '']
    ,['SMR Only', 'SMR + CCS', 'Grid Only',\
    'Grid+','Off-Grid','Best Case']
    ]
# Best case all scenarios
# with_policy = [0.332, 0.299, 0.688745445,0.450948427,0.267275365, 0.232965308]
# without_policy = [0.332, 0.436, 0.733822771,0.496025753,0.493827807, 0.45951775]

#Texas
with_policy = [0.334,
0.301,
0.806351462,
0.450948427,
0.267275365,
0.232965308]
without_policy = [0.334,
0.437,
0.851428788,
0.496025753,
0.493827807,
0.45951775]
diff = list(np.array(without_policy) - np.array(with_policy))
total_without_policy = with_policy + without_policy
fig = go.Figure(data=[
    go.Bar(name='With Policy', x=use_case, y= with_policy,
    hovertemplate='With Policy' + '<br>$%{y} per kg of Ammonia'),
    go.Bar(name='Without Policy', x=use_case, y= diff, marker_pattern_shape="x",
            hovertemplate='Without Policy' + '<br>Additional $%{y} per kg of Ammonia'),

])
# Change the bar mode
fig.update_annotations()
fig.update_layout(barmode='stack',
        title='Use Case Direct Comparison: Levelized Cost of Ammonia',
        yaxis_title="$/kg of Ammonia ",
        yaxis = dict(range=[0,1]),
        title_font=dict(size=35,
                        family="Lato, sans-serif"),
        font=dict(
        family="Lato, sans-serif",
        size=32,),
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
fig.data[0].marker.color = ('darkslateblue','darkslateblue','darkgoldenrod','darkorange','darkgreen','darkgreen')
fig.data[1].marker.color = ('slateblue','slateblue','gold','orange','green', 'green')
# fig.add_hline(y=0.4,line_width=3, line_dash="dash")
# fig.add_annotation(x=.25, y=0.4,
#             text="Current Cost of Ammonia",
#             showarrow=True,
#             arrowhead=1,
#             yshift=10)
fig.show()

fig.write_html(path+"use_case_direct_comparision_ammonia.html")