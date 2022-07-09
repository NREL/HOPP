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

## determine max/min point found in parameter sweep
# hybrid_npv
print('max hybrid_npv = ', np.max(results['hybrid_npv']))
max_index = np.where(results['hybrid_npv'] == np.max(results['hybrid_npv']))
print('wind_fraction for max hybrid_npv = ', results[des_var][max_index][0])

#hybrid_lcoe_real
print('min hybrid_lceo_real = ', np.min(results['hybrid_lcoe_real']))
min_index = np.where(results['hybrid_lcoe_real'] == np.min(results['hybrid_lcoe_real']))
print('wind_fraction for min hybrid_lcoe_real = ', results[des_var][min_index][0])

#hybrid_annual_energy
print('max hybrid_annual_energy = ', np.max(results['hybrid_annual_energy']))
max_index = np.where(results['hybrid_annual_energy'] == np.max(results['hybrid_annual_energy']))
print('wind_fraction for max hybrid_annual_energy = ', results[des_var][max_index][0])

for key in results:
    if key != des_var and key not in no_plot_outputs :
        plt.scatter(results[des_var],results[key])
        plt.xlabel('{}'.format(des_var))
        plt.ylabel('{}'.format(key))
        fname = '{}_vs_{}.png'.format(des_var,key)
        plt.savefig(fname)
        plt.clf()
    # plt.show()

