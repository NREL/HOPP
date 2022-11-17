
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
# Must run 'conda install -c plotly plotly=5.11.0'

path = 'examples/H2_Analysis/financial_summary_results/'
df = pd.read_csv(path+'Financial_Summary_PyFAST_IA_2020_8MW_Centralized_option 1.csv',header= 0,names = ['cost names','values'])
print(df.head())

# Copy in steel cost breakdown
steel_scenario = df.copy()
steel_scenario = steel_scenario[steel_scenario['cost names'].str.contains("Steel price", case=False)]
steel_scenario['cost names'] = steel_scenario.loc[:,('cost names')].str.replace('Steel price: |[(]\$/tonne[)]|CAPEX', '',regex=True)
hydrogen_lcos_per_tonne = steel_scenario[steel_scenario['cost names'].str.contains("Hydrogen", case=False).values]
hydrogen_lcos_per_tonne = hydrogen_lcos_per_tonne['values'].values
print(hydrogen_lcos_per_tonne)
# Add cost categories
labels = ['Capital Costs','Capital Costs','Capital Costs','Capital Costs','Capital Costs','Capital Costs','Capital Costs',
'Capital Costs','Capital Costs','Fixed O&M Costs','Fixed O&M Costs','Fixed O&M Costs','Capital Costs','Feedstock Costs','Feedstock Costs',
'Feedstock Costs','Feedstock Costs','Feedstock Costs','Feedstock Costs','Feedstock Costs','Feedstock Costs','Feedstock Costs','Taxes and Financial',
'Taxes and Financial','Total Cost']
steel_scenario['Labels'] = labels
# Remove total from df
total_cost = steel_scenario[steel_scenario['cost names'].str.contains("Total", case=False)]
steel_scenario = steel_scenario[~steel_scenario['cost names'].str.contains("Total", case=False)]
# Relmove hydrogen from df (more specific breakdown)
steel_scenario = steel_scenario[~steel_scenario['cost names'].str.contains("Hydrogen", case=False)]
print(steel_scenario)

# Copy in hydrogen cost breakdown
hydrogen_scenario = df.copy()
hydrogen_scenario = hydrogen_scenario[hydrogen_scenario['cost names'].str.contains("LCOH", case=False)]
hydrogen_scenario['cost names'] = hydrogen_scenario.loc[:,('cost names')].str.replace('LCOH:|[(]\$/kg[)]', '',regex=True)
hydrogen_scenario = hydrogen_scenario.rename(columns={'cost names': 'hydrogen','values': 'LCOH values'}, errors="raise")
hydrogen_scenario = hydrogen_scenario[~hydrogen_scenario['hydrogen'].str.contains('LCOH', case=False)]
total_lcoh = hydrogen_scenario['LCOH values'].sum()
print(total_lcoh)
hydrogen_scenario['percentage'] = hydrogen_scenario['LCOH values'].div(total_lcoh)
hydrogen_scenario['cost names'] = ['Hydrogen']*len(hydrogen_scenario)
hydrogen_scenario['Labels'] = ['Feedstock Costs']*len(hydrogen_scenario)
# Multiply percentage of LCOH breakdown with $/tonne feedstock cost in LCOS breakdown 
hydrogen_scenario['values'] = hydrogen_scenario['percentage']*hydrogen_lcos_per_tonne
print(hydrogen_scenario)

result = pd.concat([steel_scenario, hydrogen_scenario])
print(result)



levels = ['hydrogen', 'cost names', 'Labels']
value_column = 'values'

# Build a hierarchy of levels for Sunburst or Treemap charts.
# Levels are given starting from the bottom to the top of the hierarchy,
 #ie the last level corresponds to the root.

def build_hierarchical_dataframe(df, levels, value_column):

    df_all_trees = pd.DataFrame(columns=['id', 'parent', 'value', 'color'])
    for i, level in enumerate(levels):
        df_tree = pd.DataFrame(columns=['id', 'parent', 'value', 'color'])  
        dfg = df.groupby(levels[i:]).sum()
        dfg = dfg.reset_index()
        df_tree['id'] = dfg[level].copy()
        if i < len(levels) - 1:
            df_tree['parent'] = dfg[levels[i+1]].copy()
        else:
            value = '%.2f' %dfg[value_column].sum()
            df_tree['parent'] = 'Total LCOS \n' + value
        df_tree['value'] = dfg[value_column]
        df_all_trees = df_all_trees.append(df_tree, ignore_index=True)
    total = pd.Series(dict(id='Total LCOS \n' + value, parent='',
                              value=df[value_column].sum()
                              ))
    df_all_trees = df_all_trees.append(total, ignore_index=True)
    return df_all_trees

df_all_trees = build_hierarchical_dataframe(result, levels, value_column)


fig = make_subplots(1, 2, specs=[[{"type": "domain"}, {"type": "domain"}]],)

fig.add_trace(go.Sunburst(
    labels=df_all_trees['id'],
    parents=df_all_trees['parent'],
    values=df_all_trees['value'],
    branchvalues='total',
#     marker=dict(
#         colors=df_all_trees['color'],
#         colorscale='RdBu',
#         cmid=average_score),
     hovertemplate='<b>%{label} Contribution: </b> <br>  %{value:.2f} $/tonne of steel',
    name=''
    ), 1, 1)

fig.add_trace(go.Sunburst(
    labels=df_all_trees['id'],
    parents=df_all_trees['parent'],
    values=df_all_trees['value'],
    branchvalues='total',
#     marker=dict(
#         colors=df_all_trees['color'],
#         colorscale='RdBu',
#         cmid=average_score),
       hovertemplate='<b>%{label} </b> <br> Contribution: %{value:.2f} ($/tonne of steel)',
    maxdepth=2
    ), 1, 2)

fig.update_layout(margin=dict(t=10, b=10, r=10, l=10))

fig.show()
# fig.write_image(path + "sunburst_plot.png")

# Output only single plot
# figy = go.Figure(go.Sunburst(labels=df_all_trees['id'],
#     parents=df_all_trees['parent'],
#     values=df_all_trees['value'],
#     branchvalues='total',
#        hovertemplate='<b>%{label} </b> <br> Contribution: %{value:.2f} ($/tonne of steel)'
#     ))
# figy.show()


    