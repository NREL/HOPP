import plotly.graph_objects as go
import pandas as pd

path = 'examples/H2_Analysis/financial_summary_results/'
df1 = pd.read_csv(path+'Financial_Summary_PyFAST_TX_2030_6MW_Centralized_option 1.csv',header= 0,names = ['cost names','values'])
print(df1.head())
# steel_scenario1 = df1.copy()
# steel_scenario1 = steel_scenario1[steel_scenario1['cost names'].str.contains("Steel price", case=False)]
# steel_scenario1['cost names'] = steel_scenario1.loc[:,('cost names')].str.replace('Steel price: |[(]\$/tonne[)]|CAPEX', '',regex=True)
hydrogen_scenario1 = df1.copy()
hydrogen_scenario1 = hydrogen_scenario1.sort_values(by='values')
hydrogen_scenario1 = hydrogen_scenario1[hydrogen_scenario1['cost names'].str.contains("LCOH", case=False)]
hydrogen_scenario1['cost names'] = hydrogen_scenario1.loc[:,('cost names')].str.replace('LCOH:|[(]\$/kg[)]', '',regex=True)
hs1 = hydrogen_scenario1[hydrogen_scenario1['cost names'].str.contains('LCOH', case=False)]
hs1 = hs1.iloc[0]['values']
hydrogen_scenario1 = hydrogen_scenario1[~hydrogen_scenario1['cost names'].str.contains('LCOH', case=False)]
print(hydrogen_scenario1)

df2 = pd.read_csv(path+'Financial_Summary_PyFAST_IN_2030_6MW_Centralized_option 1.csv',header= 0,names = ['cost names','values'])
print(df2.head())
# steel_scenario2 = df2.copy()
# steel_scenario2 = steel_scenario2[steel_scenario2['cost names'].str.contains("Steel price", case=False)]
# steel_scenario2['cost names'] = steel_scenario2.loc[:,('cost names')].str.replace('Steel price: |[(]\$/tonne[)]|CAPEX', '',regex=True)
hydrogen_scenario2 = df2.copy()
hydrogen_scenario2 = hydrogen_scenario2.sort_values(by='values')
hydrogen_scenario2 = hydrogen_scenario2[hydrogen_scenario2['cost names'].str.contains("LCOH", case=False)]
hydrogen_scenario2['cost names'] = hydrogen_scenario2.loc[:,('cost names')].str.replace('LCOH:|[(]\$/kg[)]', '',regex=True)
hs2 = hydrogen_scenario2[hydrogen_scenario2['cost names'].str.contains('LCOH', case=False)]
hs2 = hs2.iloc[0]['values']
hydrogen_scenario2 = hydrogen_scenario2[~hydrogen_scenario2['cost names'].str.contains('LCOH', case=False)]
print(hydrogen_scenario2)


cost_names = hydrogen_scenario1['cost names']
sr1 = hydrogen_scenario1['values'].tolist()
sr2 = hydrogen_scenario2['values'].tolist()

#convert sr1
def Convert(lst):
    return [ -i for i in lst ]
sr3 = Convert(sr2)


fig = go.Figure()
fig.add_trace(go.Bar(y=cost_names, x=sr1,
                base=0,
                marker_color='rgb(158,202,225)',
                name=f'Texas 2030 ${hs1:.2f}/kg H<sub>2</sub>',
                marker_line_color='rgb(8,48,107)',
                orientation='h',
                marker_line_width=1.5,
                opacity= 0.7,
                text = sr1,
                textfont = {'size': 15},
                textposition='inside',
                insidetextanchor = 'end',
                texttemplate = "%{x:.2f}($/kg H<sub>2</sub>) ",
                hovertemplate='<b>Texas 2030<b>' + '<br>$%{text:.2f} per kg H<sub>2</sub>'
))
fig.add_trace(go.Bar(y=cost_names, x=sr2,
                base=sr3,
                marker_color='crimson',
                name=f'IN 2030 ${hs2:.2f}/H<sub>2</sub>',
                marker_line_color='red',
                orientation='h',
                marker_line_width=1.5,
                opacity= 0.7,
                text = sr2,
                textfont = {'size': 15},
                textposition='inside',
                insidetextanchor = 'start',
                texttemplate = "%{x:.2f}($/kg H<sub>2</sub>) ",
                hovertemplate='Indiana 2030' + '<br>$%{text:.2f} per kg H<sub>2</sub>'
))


fig.update_layout(
                height=800,
                margin=dict(t=50,l=10,b=10,r=10),
                title_text="Levelized Cost of Hydrogen Breakdown Comparison",
                title_font_family="Lato, sans-serif",
                title_font_size = 25,
                title_x=0.5, #to adjust the position along x-axis of the title
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
)


fig.update_layout(barmode='overlay', 
                  xaxis_tickangle=-45, 
                legend=dict(
                     x=0.80,
                     y=0.01,
                    bgcolor='rgba(255, 255, 255, 0)',
                    bordercolor='rgba(255, 255, 255, 0)'
                    ),
                yaxis=dict(
                    title='Cost category',
                    titlefont_size=18,
                    tickfont_size=16
                    ),
                xaxis=dict(
                    title='$/kg H2',
                    titlefont_size=18,
                    tickfont_size=16
                    ),
                bargap=0.30)



fig.update_layout(legend=dict(
                        font=dict(
                            size= 20)
))

fig.show()

fig.write_html(path+"lcoh_tornado_plot.html")