import plotly.graph_objects as go

fig = go.Figure(go.Waterfall(
    name = "2018", 
    measure = ["relative", "relative", "relative", "relative", "relative",
                "relative", "relative",
                "total"],
    x = [   "LCOH", "Small vs. Large H<sub>2</sub>" ,"Pipe vs. Cable",
            "Shared Power Electronics", 
            "Transmission Cost", "Labor Savings","Policy",
            "Adjusted LCOH"],
    y=[ 2.13529799,  0.05936623, -0.01520596, -0.01733963, -0.02966132, -0.14013644,
 -1.40294874, None],
    connector = {"mode":"between", 
                "line":{"width":4, 
                        "color":"rgb(0, 0, 0)", 
                        "dash":"solid"}},
    textposition = "outside",
    text = [ "2.14","+0.059", "-0.015", "-0.017",  "-0.030","-0.14" , "-1.40", "0.59"]
))

fig.update_layout(title = "Off-grid, Tightly-Coupled, Co-Located Wind and H<sub>2</sub> Systems",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis_title="$/kg of Hydrogen ",
                yaxis = dict(range=[0,3.5]),
                title_font=dict(size=25,
                        family="Lato, sans-serif"),
                
        font=dict(
        family="Lato, sans-serif",
        size=20,),
)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='lightgray')
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='lightgray')

fig.show()