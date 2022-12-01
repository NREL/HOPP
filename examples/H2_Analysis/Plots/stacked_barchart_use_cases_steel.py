import plotly.graph_objects as go
import numpy as np

path = 'examples/H2_Analysis/financial_summary_results/'

use_case=[['SMR', 'SMR', 'On-Grid', 'On-Grid', '', '']
    ,['SMR Only', 'SMR + CCS', 'Grid Only',\
    'Grid+','Off-Grid','Best Case']
    ]
# Best of all sites 
# with_policy = [743, 733, 766.601182,746.5530174,858.0984174, 758.5800019]
# without_policy = [743, 779, 781.6723197,761.624155131997,933.8438835, 835]

#Texas Location
with_policy =[941,
930,
984.1218635,
933.1183804,
871.7091813,
758.5800019]
without_policy = [941,
975,
999.1930012,
948.1895181,
947.4546475,
835]
diff = list(np.array(without_policy) - np.array(with_policy))
total_without_policy = with_policy + without_policy
fig = go.Figure(data=[
    go.Bar(name='With Policy', x=use_case, y= with_policy,
    hovertemplate='With Policy' + '<br>$%{y} per MT of Steel'),
    go.Bar(name='Without Policy', x=use_case, y= diff, marker_pattern_shape="x",
            hovertemplate='Without Policy' + '<br>Additional $%{y} per MT of Steel'),

])
# Change the bar mode
fig.update_annotations()
fig.update_layout(barmode='stack',
        title='Use Case Direct Comparison: Levelized Cost of Steel',
        yaxis_title="$/MT of Steel ",
        yaxis = dict(range=[0,1200]),
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
# fig.add_hline(y=743,line_width=3, line_dash="dash")
# fig.add_hline(y=731,line_width=3, line_dash="dash")
# fig.add_annotation(x=4.5, y=731,
#             ax=4.5, 
#             ay=-165,
#             text="Current Cost of Steel",
#             showarrow=True,
#             arrowhead=1,
#             # yshift=20,
#             arrowsize=2,

#             )
fig.show()

fig.write_html(path+"use_case_direct_comparision_steel.html")