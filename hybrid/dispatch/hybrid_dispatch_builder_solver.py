from typing import Union
import sys, os
from pathlib import Path
import time

import pyomo.environ as pyomo
from pyomo.network import Port, Arc
from pyomo.opt import TerminationCondition
from pyomo.util.check_units import assert_units_consistent

from hybrid.sites import SiteInfo
from hybrid.dispatch import HybridDispatch, HybridDispatchOptions, DispatchProblemState
from hybrid.clustering import Clustering
from hybrid.log import hybrid_logger as logger


class HybridDispatchBuilderSolver:
    """Helper class for building hybrid system dispatch problem, solving dispatch problem, and simulating system
    with dispatch solution."""
    def __init__(self,
                 site: SiteInfo,
                 power_sources: dict,
                 dispatch_options: dict = None):
        """

        Parameters
        ----------
        dispatch_options :
            Contains attribute key, value pairs to change default dispatch options.
            For details see HybridDispatchOptions in hybrid_dispatch_options.py

        """
        self.opt = None
        self.site: SiteInfo = site
        self.power_sources = power_sources
        self.options = HybridDispatchOptions(dispatch_options)

        # deletes previous log file under same name
        if os.path.isfile(self.options.log_name):
            os.remove(self.options.log_name)

        self.needs_dispatch = any(item in ['battery', 'tower', 'trough'] for item in self.power_sources.keys())

        if self.needs_dispatch:
            self._pyomo_model = self._create_dispatch_optimization_model()
            if self.site.follow_desired_schedule:
                self.dispatch.create_min_operating_cost_objective()
            else:
                self.dispatch.create_max_gross_profit_objective()
            self.dispatch.create_arcs()
            assert_units_consistent(self.pyomo_model)
            self.problem_state = DispatchProblemState()
        
        # Clustering (optional)
        self.clustering = None
        if self.options.use_clustering:
            #TODO: Add resource data for wind
            self.clustering = Clustering(power_sources.keys(), self.site.solar_resource.filename, wind_resource_data = None, price_data = self.site.elec_prices.data)
            self.clustering.n_cluster = self.options.n_clusters
            if len(self.options.clustering_weights.keys()) == 0:
                self.clustering.use_default_weights = True
            elif self.options.clustering_divisions.keys() != self.options.clustering_weights.keys():
                print ('Warning: Keys in user-specified dictionaries for clustering weights and divisions do not match. Reverting to default weights/divisions')
                self.clustering.use_default_weights = True
            else:
                self.clustering.weights = self.options.clustering_weights
                self.clustering.divisions = self.options.clustering_divisions
                self.clustering.use_default_weights = False
            self.clustering.run_clustering()  # Create clusters and find exemplar days for simulation

    def _create_dispatch_optimization_model(self):
        """
        Creates monolith dispatch model
        """
        model = pyomo.ConcreteModel(name='hybrid_dispatch')
        #################################
        # Sets                          #
        #################################
        model.forecast_horizon = pyomo.Set(doc="Set of time periods in time horizon",
                                           initialize=range(self.options.n_look_ahead_periods))
        #################################
        # Blocks (technologies)         #
        #################################
        module = getattr(__import__("hybrid"), "dispatch")
        for source, tech in self.power_sources.items():
            if source == 'battery':
                tech._dispatch = self.options.battery_dispatch_class(
                    model,
                    model.forecast_horizon,
                    tech._system_model,
                    tech._financial_model,
                    block_set_name=source,
                    dispatch_options=self.options)
            else:
                try:
                    dispatch_class_name = getattr(module, source.capitalize() + "Dispatch")
                except AttributeError:
                    raise ValueError("Could not find {} in hybrid.dispatch module. Is {} supported in the hybrid "
                                     "dispatch model?".format(source.capitalize() + "Dispatch", source))
                tech._dispatch = dispatch_class_name(
                    model,
                    model.forecast_horizon,
                    tech._system_model,
                    tech._financial_model)

        self._dispatch = HybridDispatch(
            model,
            model.forecast_horizon,
            self.power_sources,
            self.options)
        return model

    def solve_dispatch_model(self, start_time: int, n_days: int):
        # Solve dispatch model
        if self.options.solver == 'glpk':
            solver_results = self.glpk_solve()
        elif self.options.solver == 'cbc':
            solver_results = self.cbc_solve()
        elif self.options.solver == 'xpress':
            solver_results = self.xpress_solve()
        elif self.options.solver == 'xpress_persistent':
            solver_results = self.xpress_persistent_solve()
        elif self.options.solver == 'gurobi_ampl':
            solver_results = self.gurobi_ampl_solve()
        elif self.options.solver == 'gurobi':
            solver_results = self.gurobi_solve()
        else:
            raise ValueError("{} is not a supported solver".format(self.options.solver))

        self.problem_state.store_problem_metrics(solver_results, start_time, n_days,
                                                 self.dispatch.objective_value)

    @staticmethod
    def glpk_solve_call(pyomo_model: pyomo.ConcreteModel,
                        log_name: str = "",
                        user_solver_options: dict = None):

        # log_name = "annual_solve_GLPK.log"  # For debugging MILP solver
        # Ref. on solver options: https://en.wikibooks.org/wiki/GLPK/Using_GLPSOL
        glpk_solver_options = {'cuts': None,
                               'presol': None,
                               # 'mostf': None,
                               # 'mipgap': 0.001,
                               'tmlim': 30
                               }
        solver_options = SolverOptions(glpk_solver_options, log_name, user_solver_options,'log')
        with pyomo.SolverFactory('glpk') as solver:
            results = solver.solve(pyomo_model, options=solver_options.constructed)
        HybridDispatchBuilderSolver.log_and_solution_check(log_name, solver_options.instance_log, results.solver.termination_condition, pyomo_model)
        return results
        
    def glpk_solve(self):
        return HybridDispatchBuilderSolver.glpk_solve_call(self.pyomo_model,
                                                           self.options.log_name,
                                                           self.options.solver_options)
        
    @staticmethod
    def gurobi_ampl_solve_call(pyomo_model: pyomo.ConcreteModel,
                               log_name: str = "",
                               user_solver_options: dict = None):

        # Ref. on solver options: https://www.gurobi.com/documentation/9.1/ampl-gurobi/parameters.html
        gurobi_solver_options = {'timelim': 60,
                                 'threads': 1}
        solver_options = SolverOptions(gurobi_solver_options, log_name, user_solver_options,'logfile')

        with pyomo.SolverFactory('gurobi', executable='/opt/solvers/gurobi', solver_io='nl') as solver:
            results = solver.solve(pyomo_model, options=solver_options.constructed)
        HybridDispatchBuilderSolver.log_and_solution_check(log_name, solver_options.instance_log, results.solver.termination_condition, pyomo_model)
        return results

    def gurobi_ampl_solve(self):
        return HybridDispatchBuilderSolver.gurobi_ampl_solve_call(self.pyomo_model,
                                                                  self.options.log_name,
                                                                  self.options.solver_options)
                                                                  
    @staticmethod
    def gurobi_solve_call(opt: pyomo.SolverFactory,
                          pyomo_model: pyomo.ConcreteModel,
                          log_name: str = "",
                          user_solver_options: dict = None):

        # Ref. on solver options: https://www.gurobi.com/documentation/9.1/ampl-gurobi/parameters.html
        gurobi_solver_options = {'timelim': 60,
                                 'threads': 1}
        solver_options = SolverOptions(gurobi_solver_options, log_name, user_solver_options,'logfile')
        
        opt.options.update(solver_options.constructed)
        opt.set_instance(pyomo_model)
        results = opt.solve(save_results=False)
        HybridDispatchBuilderSolver.log_and_solution_check(log_name, solver_options.instance_log, results.solver.termination_condition, pyomo_model)
        return results

    def gurobi_solve(self):
        if self.opt is None:
            self.opt = pyomo.SolverFactory('gurobi', solver_io='persistent') 
    
        return HybridDispatchBuilderSolver.gurobi_solve_call(self.opt,
                                                             self.pyomo_model,
                                                             self.options.log_name,
                                                             self.options.solver_options)

    @staticmethod
    def cbc_solve_call(pyomo_model: pyomo.ConcreteModel,
                       log_name: str = "",
                       user_solver_options: dict = None):
        # log_name = "annual_solve_CBC.log"
        # Solver options can be found by launching executable 'start cbc.exe', verbose 15, ?
        # https://coin-or.github.io/Cbc/faq.html (a bit outdated)
        cbc_solver_options = {  # 'ratioGap': 0.001,
                              'seconds': 60}
        solver_options = SolverOptions(cbc_solver_options, log_name, user_solver_options,'log')

        if sys.platform == 'win32' or sys.platform == 'cygwin':
            cbc_path = Path(__file__).parent / "cbc_solver" / "cbc-win64" / "cbc"
            if log_name != "":
                logger.warning("Warning: CBC solver logging is active... This will significantly increase simulation time.")
                solver_options.constructed['log'] = 2
                solver = pyomo.SolverFactory('asl:cbc', executable=cbc_path)
                results = solver.solve(pyomo_model, logfile=solver_options.instance_log, options=solver_options.constructed)
            else:
                solver = pyomo.SolverFactory('cbc', executable=cbc_path, solver_io='nl')
                results = solver.solve(pyomo_model, options=solver_options.constructed)
        elif sys.platform == 'darwin' or sys.platform == 'linux':
            solver = pyomo.SolverFactory('cbc')
            results = solver.solve(pyomo_model, options=solver_options.constructed)
        else:
            raise SystemError('Platform not supported ', sys.platform)
        
        HybridDispatchBuilderSolver.log_and_solution_check(log_name, solver_options.instance_log, results.solver.termination_condition, pyomo_model)
        return results

    def cbc_solve(self):
        return HybridDispatchBuilderSolver.cbc_solve_call(self.pyomo_model,
                                                          self.options.log_name,
                                                          self.options.solver_options)

    @staticmethod
    def xpress_solve_call(pyomo_model: pyomo.ConcreteModel,
                          log_name: str = "",
                          user_solver_options: dict = None):

        # FIXME: Logging does not work
        # log_name = "annual_solve_Xpress.log"  # For debugging MILP solver
        # Ref. on solver options: https://ampl.com/products/solvers/solvers-we-sell/xpress/options/
        xpress_solver_options = {'mipgap': 0.001,
                                 'maxtime': 30}
        solver_options = SolverOptions(xpress_solver_options, log_name, user_solver_options,'LOGFILE')

        with pyomo.SolverFactory('xpress_direct') as solver:
            results = solver.solve(pyomo_model, options=solver_options.constructed)
        HybridDispatchBuilderSolver.log_and_solution_check(log_name, solver_options.instance_log, results.solver.termination_condition, pyomo_model)
        return results

    def xpress_solve(self):
        return HybridDispatchBuilderSolver.xpress_solve_call(self.pyomo_model,
                                                             self.options.log_name,
                                                             self.options.solver_options)

    @staticmethod
    def xpress_persistent_solve_call(opt: pyomo.SolverFactory,
                                     pyomo_model: pyomo.ConcreteModel,
                                     log_name: str = "",
                                     user_solver_options: dict = None):

        # log_name = "annual_solve_Xpress.log"  # For debugging MILP solver
        # Ref. on solver options: https://ampl.com/products/solvers/solvers-we-sell/xpress/options/
        xpress_solver_options = {'mipgap': 0.001,
                                 'MAXTIME': 30}
        solver_options = SolverOptions(xpress_solver_options, log_name, user_solver_options,'LOGFILE')

        opt.options.update(solver_options.constructed)
        opt.set_instance(pyomo_model)
        results = opt.solve(save_results=False)
        HybridDispatchBuilderSolver.log_and_solution_check(log_name, solver_options.instance_log, results.solver.termination_condition, pyomo_model)
        return results

    def xpress_persistent_solve(self):
        if self.opt is None:
            self.opt = pyomo.SolverFactory('xpress', solver_io='persistent')

        return HybridDispatchBuilderSolver.xpress_persistent_solve_call(self.opt,
                                                                        self.pyomo_model,
                                                                        self.options.log_name,
                                                                        self.options.solver_options)
    @staticmethod
    def mindtpy_solve_call(pyomo_model: pyomo.ConcreteModel,
                           log_name: str = ""):
        raise NotImplementedError
        solver = pyomo.SolverFactory('mindtpy')
        results = solver.solve(pyomo_model,
                               mip_solver='glpk',
                               nlp_solver='ipopt',
                               tee=True)

        HybridDispatchBuilderSolver.log_and_solution_check("", "", results.solver.termination_condition, pyomo_model)
        return results

    @staticmethod
    def log_and_solution_check(log_name:str, solve_log: str, solver_termination_condition, pyomo_model):
        if log_name != "":
            HybridDispatchBuilderSolver.append_solve_to_log(log_name, solve_log)
        HybridDispatchBuilderSolver.check_solve_condition(solver_termination_condition, pyomo_model)

    @staticmethod
    def check_solve_condition(solver_termination_condition, pyomo_model):
        if solver_termination_condition == TerminationCondition.infeasible:
            HybridDispatchBuilderSolver.print_infeasible_problem(pyomo_model)
        elif not solver_termination_condition == TerminationCondition.optimal:
            logger.warning("Warning: Dispatch problem termination condition was '"
                  + str(solver_termination_condition) + "'")

    @staticmethod
    def append_solve_to_log(log_name: str, solve_log: str):
        # Appends single problem instance log to annual log file
        fin = open(solve_log, 'r')
        data = fin.read()
        fin.close()

        ann_log = open(log_name, 'a+')
        ann_log.write("=" * 50 + "\n")
        ann_log.write(data)
        ann_log.close()

    @staticmethod
    def print_infeasible_problem(model: pyomo.ConcreteModel):
        original_stdout = sys.stdout
        with open('infeasible_instance.txt', 'w') as f:
            sys.stdout = f
            print('\n' + '#' * 20 + ' Model Parameter Values ' + '#' * 20 + '\n')
            HybridDispatchBuilderSolver.print_all_parameters(model)
            print('\n' + '#' * 20 + ' Model Blocks Display ' + '#' * 20 + '\n')
            HybridDispatchBuilderSolver.display_all_blocks(model)
            sys.stdout = original_stdout
        raise ValueError("Dispatch optimization model is infeasible.\n"
                         "See 'infeasible_instance.txt' for parameter values.")

    @staticmethod
    def print_all_parameters(model: pyomo.ConcreteModel):
        param_list = list()
        block_list = list()
        for param_object in model.component_objects(pyomo.Param, active=True):
            name_to_print = param_object.getname()
            parent_block = param_object.parent_block().parent_component()
            block_name = parent_block.getname()
            if (name_to_print not in param_list) or (block_name not in block_list):
                block_list.append(block_name)
                param_list.append(name_to_print)
                print("\nParent Block Name: ", block_name)
                print("Parameter: ", name_to_print)
                for index in parent_block.index_set():
                    val_to_print = pyomo.value(getattr(parent_block[index], param_object.getname()))
                    print("\t", index, "\t", val_to_print)

    @staticmethod
    def display_all_blocks(model: pyomo.ConcreteModel):
        for block_object in model.component_objects(pyomo.Block, active=True):
            for index in block_object.index_set():
                block_object[index].display()

    def simulate_power(self):
        if self.needs_dispatch:
            # Dispatch Optimization Simulation with Rolling Horizon
            logger.info("Simulating system with dispatch optimization...")
        else:
            logger.info("Dispatch optimization not required...")
            return
        ti = list(range(0, self.site.n_timesteps, self.options.n_roll_periods))
        self.dispatch.initialize_parameters()

        if self.clustering is None:
            # Solving the year in series
            for i, t in enumerate(ti):
                if self.options.is_test_start_year or self.options.is_test_end_year:
                    if (self.options.is_test_start_year and i < 5) or (self.options.is_test_end_year and i > 359):
                        start_time = time.time()
                        self.simulate_with_dispatch(t)
                        sim_w_dispath_time = time.time()
                        logger.info('Day {} dispatch optimized.'.format(i))
                        logger.info("      %6.2f seconds required to simulate with dispatch" % (sim_w_dispath_time - start_time))
                    else:
                        continue
                        # TODO: can we make the csp and battery model run with heuristic dispatch here?
                        #  Maybe calling a simulate_with_heuristic() method
                else:
                    if (i % 73) == 0:
                        logger.info("\t {:.0f} % complete".format(i*20/73))
                    self.simulate_with_dispatch(t)
        else:

            initial_states = {tech:{'day':[], 'soc':[], 'load':[]} for tech in ['trough', 'tower', 'battery'] if tech in self.power_sources.keys()}  # List of known charge states at 12 am from completed simulations
            npercluster = self.clustering.clusters['count']
            inds = sorted(range(len(npercluster)), key=npercluster.__getitem__)  # Indicies to sort clusters by low-to-high number of days represented
            for i in range(self.clustering.clusters['n_cluster']):
                j = inds[i]  # cluster index
                time_start, time_stop = self.clustering.get_sim_start_end_times(j)
                battery_soc = self.clustering.battery_soc_heuristic(j, initial_states['battery']) if 'battery' in self.power_sources.keys() else None

                # Set CSP initial states (need to do this prior to update_time_series_parameters() or update_initial_conditions(), both pull from the stored plant state)
                for tech in ['trough', 'tower']:
                    if tech in self.power_sources.keys():
                        self.power_sources[tech].plant_state = self.power_sources[tech].set_initial_plant_state()  # Reset to default initial state
                        csp_soc, is_cycle_on, initial_cycle_load = self.clustering.csp_initial_state_heuristic(j, self.power_sources[tech].solar_multiple, initial_states[tech])
                        self.power_sources[tech].set_tes_soc(csp_soc)  
                        self.power_sources[tech].set_cycle_state(is_cycle_on)  
                        self.power_sources[tech].set_cycle_load(initial_cycle_load)

                self.simulate_with_dispatch(time_start, self.clustering.ndays+1, battery_soc, n_initial_sims = 1)  

                # Update lists of known states at 12am
                for tech in ['trough', 'tower', 'battery']: 
                    if tech in self.power_sources.keys():
                        for d in range(self.clustering.ndays):
                            day  = self.clustering.sim_start_days[j]+d
                            initial_states[tech]['day'].append(day)
                            if tech in ['trough', 'tower']:
                                initial_states[tech]['soc'].append(self.power_sources[tech].get_tes_soc(day*24))
                                initial_states[tech]['load'].append(self.power_sources[tech].get_cycle_load(day*24))
                            elif tech in ['battery']:
                                step = day*24 * int(self.site.n_timesteps/8760)
                                initial_states[tech]['soc'].append(self.power_sources[tech].Outputs.SOC[step])

            # After exemplar simulations, update to full annual generation array for dispatchable technologies
            for tech in self.power_sources.keys():
                if tech in ['battery']:
                    for key in ['gen', 'P', 'SOC']:
                        val = getattr(self.power_sources[tech].Outputs, key)
                        setattr(self.power_sources[tech].Outputs, key, list(self.clustering.compute_annual_array_from_cluster_exemplar_data(val)))
                elif tech in ['trough', 'tower']:
                    for key in ['gen', 'P_out_net', 'P_cycle', 'q_dot_pc_startup', 'q_pc_startup', 'e_ch_tes', 'eta', 'q_pb']:  # Data quantities used in capacity value calculations
                        self.power_sources[tech].outputs.ssc_time_series[key] = list(self.clustering.compute_annual_array_from_cluster_exemplar_data(self.power_sources[tech].outputs.ssc_time_series[key])) 

    def simulate_with_dispatch(self,
                               start_time: int,
                               n_days: int = 1,
                               initial_soc: float = None,
                               n_initial_sims: int = 0):
        # this is needed for clustering effort
        update_dispatch_times = list(range(start_time,
                                           start_time + n_days * self.site.n_periods_per_day,
                                           self.options.n_roll_periods))

        for i, sim_start_time in enumerate(update_dispatch_times):
            # Update battery initial state of charge
            if 'battery' in self.power_sources.keys():
                self.power_sources['battery'].dispatch.update_dispatch_initial_soc(initial_soc=initial_soc)
                initial_soc = None

            for model in self.power_sources.values():
                if model.system_capacity_kw == 0:
                    continue
                model.dispatch.update_time_series_parameters(sim_start_time)

            if self.site.follow_desired_schedule:
                n_horizon = len(self.power_sources['grid'].dispatch.blocks.index_set())
                if start_time + n_horizon > len(self.site.desired_schedule):
                    system_limit = list(self.site.desired_schedule[start_time:])
                    system_limit.extend(list(self.site.desired_schedule[0:n_horizon - len(system_limit)]))
                else:
                    system_limit = self.site.desired_schedule[start_time:start_time + n_horizon]

                transmission_limit = self.power_sources['grid'].value('grid_interconnection_limit_kwac') / 1e3
                for count, value in enumerate(system_limit):
                    if value > transmission_limit:
                        logger.warning('Warning: Desired schedule is greater than transmission limit. '
                              'Overwriting schedule to transmission limit')
                        system_limit[count] = transmission_limit

                self.power_sources['grid'].dispatch.generation_transmission_limit = system_limit

            if 'heuristic' in self.options.battery_dispatch:
                # TODO: this is not a good way to do this... This won't work with CSP addition...
                self.battery_heuristic()
                # TODO: we could just run the csp model without dispatch here
            else:
                self.solve_dispatch_model(start_time, n_days)

            store_outputs = True
            battery_sim_start_time = sim_start_time
            if i < n_initial_sims:
                store_outputs = False
                battery_sim_start_time = None

            # simulate using dispatch solution
            if 'battery' in self.power_sources.keys():
                self.power_sources['battery'].simulate_with_dispatch(self.options.n_roll_periods,
                                                                     sim_start_time=battery_sim_start_time)

            if 'trough' in self.power_sources.keys():
                self.power_sources['trough'].simulate_with_dispatch(self.options.n_roll_periods,
                                                                    sim_start_time=sim_start_time,
                                                                    store_outputs=store_outputs)
            if 'tower' in self.power_sources.keys():
                self.power_sources['tower'].simulate_with_dispatch(self.options.n_roll_periods,
                                                                   sim_start_time=sim_start_time,
                                                                   store_outputs=store_outputs)

    def battery_heuristic(self):
        tot_gen = [0.0]*self.options.n_look_ahead_periods
        if 'pv' in self.power_sources.keys():
            pv_gen = self.power_sources['pv'].dispatch.available_generation
            tot_gen = [pv + gen for pv, gen in zip(pv_gen, tot_gen)]
        if 'wind' in self.power_sources.keys():
            wind_gen = self.power_sources['wind'].dispatch.available_generation
            tot_gen = [wind + gen for wind, gen in zip(wind_gen, tot_gen)]

        grid_limit = self.power_sources['grid'].dispatch.generation_transmission_limit

        if 'one_cycle' in self.options.battery_dispatch:
            # Get prices for one cycle heuristic
            prices = self.power_sources['grid'].dispatch.electricity_sell_price
            self.power_sources['battery'].dispatch.prices = prices

        self.power_sources['battery'].dispatch.set_fixed_dispatch(tot_gen, grid_limit)

    @property
    def pyomo_model(self) -> pyomo.ConcreteModel:
        return self._pyomo_model

    @property
    def dispatch(self) -> HybridDispatch:
        return self._dispatch

class SolverOptions:
    """Class for housing solver options"""
    def __init__(self, solver_spec_options: dict, log_name: str="", user_solver_options: dict = None, solver_spec_log_key: str="logfile"):
        self.instance_log = "dispatch_solver.log"
        self.solver_spec_options = solver_spec_options
        self.user_solver_options = user_solver_options
        
        self.constructed = solver_spec_options
        if log_name != "":
            self.constructed[solver_spec_log_key] = self.instance_log
        if user_solver_options is not None:
            self.constructed.update(user_solver_options)
        
            