import matplotlib.pyplot as plt
import openmdao.api as om
import numpy as np


cr = om.CaseReader("cases.sql")
cases = cr.list_cases('driver')

results = {}
for case in cases:
    outputs = cr.get_case(case).outputs
    for key in outputs:

        if key not in results.keys():
            results[key] = []

        results[key].append(outputs[key])

for key in results:
    results[key] = np.array(results[key]) 

# Plot DV vs various objectives, intended for one design variable and 1+ objectives
des_var = list(cr.get_case(case).get_design_vars().keys())[0]
no_plot_outputs = ['pv_generation_profile', 
                   'wind_generation_profile', 
                   'hybrid_generation_profile', 
                   'pv_resource_gh', 
                   'wind_resource_speed', 
                   'wind_resource_temp', 
                   'wind_resource_press', 
                   'wind_resource_dir',
                   ]
for key in results:
    if key != des_var and key not in no_plot_outputs :
        plt.scatter(results[des_var],results[key])
        plt.xlabel('{}'.format(des_var))
        plt.ylabel('{}'.format(key))
        fname = '{}_vs_{}.png'.format(des_var,key)
        plt.savefig(fname)
        plt.clf()
    # plt.show()