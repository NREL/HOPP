import plotly.graph_objects as go

path = 'examples/H2_Analysis/financial_summary_results/'

use_case=['SMR Only', 'SMR + CCS', 'Grid Only',\
    'Grid + Co-located Renewables','Off-grid']
with_policy = [0, 22, 25,32,34]
without_policy = [20, 22, 29,10,40]
total_without_policy = with_policy + without_policy
fig = go.Figure(data=[
    go.Bar(name='With Policy', x=use_case, y= with_policy,
    hovertemplate='With Policy' + '<br>$%{y} per MT of Steel'),
    go.Bar(name='Without Policy', x=use_case, y= without_policy, marker_pattern_shape="x",
            hovertemplate='Without Policy' + '<br>Additional $%{y} per MT of Steel'),

])
# Change the bar mode
fig.update_annotations()
fig.update_layout(barmode='stack',
        title='Use Case Direct Comparison: Levelized Cost of Steel',
        yaxis_title="$/MT of Steel ",
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
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)')
fig.data[0].marker.color = ('darkslateblue','darkslateblue','darkgoldenrod','darkorange','mediumseagreen')
fig.data[1].marker.color = ('slateblue','slateblue','gold','orange','darkseagreen')
fig.add_hline(y=20,line_width=3, line_dash="dash")
fig.show()

fig.write_html(path+"use_case_direct_comparision_steel.html")