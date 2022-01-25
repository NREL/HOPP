from PySAM import Singleowner

from optimization_problem_alt import expand_financial_model
import ipywidgets as widgets
from ipywidgets import interact, Layout
import plotly.express as px
import pandas as pd
import numpy as np
import os

# Plotting imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import plotly.io as pio
PAPER_LAYOUT=dict(font=dict(family="Computer Modern", size=18),
                  margin=dict(t=40))
pio.templates["paper"] = go.layout.Template(layout=PAPER_LAYOUT)
pio.templates.default = "paper"


def calc_capacity_credit_perc(model_dict, N=100):
    TIMESTEPS_YEAR = 8760
    
    price = model_dict['Revenue']['dispatch_factors_ts'][:TIMESTEPS_YEAR]
    gen = model_dict['SystemOutput']['gen'][:TIMESTEPS_YEAR]
    sys_cap = model_dict['CapacityPayments']['cp_system_nameplate']
    
    data = pd.DataFrame({'price': price, 'gen':gen})
    selected = data.nlargest(N, 'price', keep='all')
    
    cap_perc = min(100, 100 * selected['gen'][:N].sum() / (sys_cap * N))
    
    return (cap_perc,)


def recalc_financial_objective(df, scenarios, obj_col):
    f_cols = [col for col in df.columns if col.endswith('_financial_model')]
    
    data = dict()
    for idx, row in df.iterrows():
        for i, scenario in enumerate(scenarios):
            
            for j, col in enumerate(f_cols):
                new_col = col.split('_')[0] + '_' + scenario['name']
                if new_col not in data:
                    data[new_col] = []
                
                d = expand_financial_model(row[col])
                model = Singleowner.new() #.default(defaults[tech_prefix[j]])
                model.assign(d)
                
                for key, val in scenario['values'].items():
                    if key == 'cp_capacity_credit_percent':
                        model.value(key, calc_capacity_credit_perc(d, val))
                    else:
                        model.value(key, val)
                        
                model.execute()   
                data[new_col].append(model.value(obj_col))
                        
    for key, val in data.items():
        df[key] = val

    return df


def interact_row_wrap(_InteractFactory__interact_f=None, **kwargs):
    def patch(obj):
        if hasattr(obj.widget, 'layout'):
            obj.widget.layout = Layout(flex_flow='row wrap')
        return obj
    if _InteractFactory__interact_f is None:
        def decorator(f):
            obj = interact(f, **kwargs)
            return patch(obj)
        return decorator
    else:
        obj = interact(_InteractFactory__interact_f, **kwargs)
        return patch(obj)
        

def scaler(s):
    return (s - s.min()) / (s.max() - s.min())
    

def plotDataframe(df: pd.DataFrame, points="all", barmode="group", **kwargs):
    """plot different chart types to compare performance and see performance variability across random seeds

    uses Plotly express; all available keyword arguments can be passed through to px.bar(), px.scatter(), etc.
    """

    options = [None] + [col for col in df.columns if not hasattr(df[col].iloc[0], '__len__')]
    if 'width' not in kwargs.keys():
        kwargs['width'] = 900

    # read custom axis data or use common defaults for summary DataFrame
    x_default = kwargs.pop("x", options[1])
    y_default = kwargs.pop("y", options[2])
    color_default = kwargs.pop("color", options[3])

    # set type to None to disable plotting until a type is selected
    type_default = kwargs.pop("type", "scatter")

    # pop values for remaining widgets to avoid double inputs
    symbol_default = kwargs.pop("symbol", None)
    log_x_default = kwargs.pop("log_x", False)
    log_y_default = kwargs.pop("log_y", False)
    scale_x_default = kwargs.pop("scale_x", False)
    scale_y_default = kwargs.pop("scale_y", False)

    switches = dict(
        x=widgets.Dropdown(options=options, value=x_default),
        y=widgets.Dropdown(options=options, value=y_default),
        color=widgets.Dropdown(options=options, value=color_default),
        type=widgets.Dropdown(
            options=["box", "bar", "scatter", "line"], value=type_default
        ),
        symbol=widgets.Dropdown(options=options, value=symbol_default),
        log_x=widgets.Checkbox(value=log_x_default),
        log_y=widgets.Checkbox(value=log_y_default),
        scale_x=widgets.Checkbox(value=scale_x_default),
        scale_y=widgets.Checkbox(value=scale_y_default),
    )
    
    def plot(x, y, color, type, symbol, log_x, log_y, scale_x, scale_y):
        keys = list()
        data_list = list()
        
        for i, key in enumerate([x, y, color, symbol]):
            if key is not None:
                keys.append(key)
                
                if i < 2:
                    data_list.append((scaler(df[key]) if scale_x else df[key]) if i==0 
                                     else (scaler(df[key]) if scale_y else df[key]))
                else:
                    data_list.append(df[key])
                    
        plot_data = pd.concat(data_list, axis=1, keys=keys)
        
        if type == "box":
            return px.box(plot_data, x=x, y=y, color=color, points=points, log_x=log_x, log_y=log_y, **kwargs)
        
        elif type == "bar":
            return px.bar(plot_data, x=x, y=y, color=color, barmode=barmode, log_x=log_x, log_y=log_y, **kwargs)
        
        elif type == "scatter":
            return px.scatter(plot_data, x=x, y=y, color=color, symbol=symbol, log_x=log_x, log_y=log_y, **kwargs)
        
        elif type == "line":    
            return px.line(plot_data, x=x, y=y, color=color, symbol=symbol, log_x=log_x, log_y=log_y, **kwargs)
    
    return interact_row_wrap(plot, **switches).widget         
            


