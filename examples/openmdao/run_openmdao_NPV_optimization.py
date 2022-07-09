import openmdao.api as om
import json
from pathlib import Path
import time

from hybrid.openmdao_wrapper import HybridSystem
from hybrid.sites import SiteInfo, flatirons_site
from hybrid.hybrid_simulation import HybridSimulation
from hybrid.log import hybrid_logger as logger
from hybrid.keys import set_nrel_key_dot_env

start_time = time.time()

        
resource_files_dir = Path(__file__).parent.absolute().parent.absolute().parent.absolute() / 'resource_files'

# Set API key
set_nrel_key_dot_env()

if __name__ == "__main__":
    ### build the model
    prob = om.Problem()
    prob.model.add_subsystem('hybrid_system', HybridSystem(location=flatirons_site, battery=False, grid=True, sim_duration_years=25, hybrid_rated_power=250), promotes_inputs=['wind_fraction'], promotes_outputs=['hybrid_npv'])

    ### setup the optimization
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'trust-constr'
    prob.driver.options['tol'] = 1e-4
    prob.driver.options['disp'] = True
    # prob.driver = om.DOEDriver(om.FullFactorialGenerator(levels=76))
    prob.driver.options['debug_print'] = ["desvars", "objs"]
    prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))
    # prob.driver.recording_options['includes'] = ['*']

    ### setup design variables
    # Solar DVs
    # prob.model.add_design_var('solar_size_mw', lower=0.001, upper=15.)
    # prob.model.add_design_var('solar_fraction', lower=0.001, upper=0.999)

    # Wind DVs
    prob.model.add_design_var('wind_fraction', lower=0., upper=1.)
    # prob.model.add_design_var('wind_size_mw', lower=0.001, upper=15.)
    # prob.model.add_design_var('turbine_rating_kw', lower=10, upper=14000)

    # Battery DVs
    # prob.model.add_design_var('battery_capacity_mwh', lower=0., upper=5.)
    # prob.model.add_design_var('battery_power_mw', lower=0., upper=5.) 

    # Grid DVs
    # prob.model.add_design_var('interconnection_size_mw', lower=0., upper=5.)

    # COBYLA Constraints:
    prob.model.add_constraint('wind_fraction', lower=0., upper=1.)

    ## setup objective function
    prob.model.add_objective('hybrid_npv', ref=-1.e9)
    # prob.model.add_objective('hybrid_lcoe_real', ref=1.)
    # prob.model.add_objective('hybrid_annual_energy', ref=-1.e9)
     
    prob.model.approx_totals(method='fd', step=1e-3, form='forward')

    prob.setup()
    prob.run_driver()

    print('design variable =', prob['wind_fraction'])
    print('objective =', prob['hybrid_npv'])

    prob.model.list_outputs()

    end_time = time.time()
    print('run time = ', (end_time - start_time))