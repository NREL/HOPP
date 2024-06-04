"""
This file is based on the WISDEM file of the same name: https://github.com/WISDEM/WISDEM
"""

import os
import warnings

import numpy as np
from typing import Optional, Union

import openmdao.api as om

from greenheart.simulation.greenheart_simulation import GreenHeartSimulationConfig
from greenheart.tools.optimization.openmdao import TurbineDistanceComponent, BoundaryDistanceComponent
from hopp.simulation import HoppInterface

class PoseOptimization(object):
    """This class contains a collection of methods for setting up an openmdao optimization problem for a greenheart simulation.

    Args:
        config (GreenHeartSimulationConfig): instance of a greenheart config containing all desired simulation set up
    """
    def __init__(self, config: GreenHeartSimulationConfig):
        """This method primarily establishes lists of optimization methods available through different optimization drivers
        """

        self.config = config

        self.nlopt_methods = [
            "GN_DIRECT",
            "GN_DIRECT_L",
            "GN_DIRECT_L_NOSCAL",
            "GN_ORIG_DIRECT",
            "GN_ORIG_DIRECT_L",
            "GN_AGS",
            "GN_ISRES",
            "LN_COBYLA",
            "LD_MMA",
            "LD_CCSAQ",
            "LD_SLSQP",
        ]

        self.scipy_methods = [
            "SLSQP",
            "Nelder-Mead",
            "COBYLA",
        ]

        self.pyoptsparse_methods = [
            "SNOPT",
            "CONMIN",
            "NSGA2",
        ]

    def get_number_design_variables(self):
        """This method counts the number of design variables required given the provided set up and returns the result

        Returns:
            int: number of design variables
        """
        # Determine the number of design variables
        n_DV = 0
        
        if self.config.greenheart_config["opt_options"]["design_variables"]["electrolyzer_rating_kw"]["flag"]:
            n_DV += 1
        if self.config.greenheart_config["opt_options"]["design_variables"]["pv_capacity_kw"]["flag"]:
            n_DV += 1
        if self.config.greenheart_config["opt_options"]["design_variables"]["wave_capacity_kw"]["flag"]:
            n_DV += 1
        if self.config.greenheart_config["opt_options"]["design_variables"]["battery_capacity_kw"]["flag"]:
            n_DV += 1
        if self.config.greenheart_config["opt_options"]["design_variables"]["battery_capacity_kwh"]["flag"]:
            n_DV += 1
        if self.config.greenheart_config["opt_options"]["design_variables"]["turbine_x"]["flag"]:
            n_DV += self.config.hopp_config["technologies"]["wind"]["num_turbines"]
        if self.config.greenheart_config["opt_options"]["design_variables"]["turbine_y"]["flag"]:
            n_DV += self.config.hopp_config["technologies"]["wind"]["num_turbines"]
        
        # Wrap-up at end with multiplier for finite differencing
        if "form" in self.config.greenheart_config["opt_options"]["driver"]["optimization"].keys():
            if self.config.greenheart_config["opt_options"]["driver"]["optimization"]["form"] == "central": # TODO this should probably be handled at the MPI point to avoid confusion with n_DV being double what would be expected
                n_DV *= 2

        return n_DV

    def _get_step_size(self):
        """If a step size for the driver-level finite differencing is provided, use that step size. Otherwise use a default value.

        Returns:
            step size (float): step size for optimization
        """

        if not "step_size" in self.config.greenheart_config["opt_options"]["driver"]["optimization"]:
            step_size = 1.0E-6
            warnings.warn(f"Step size was not specified, setting step size to {step_size}. Step size may be set in the greenheart \
                          config file under opt_options/driver/optimization/step_size and should be of type float", UserWarning)
        else:
            step_size = self.config.greenheart_config["opt_options"]["driver"]["optimization"]["step_size"]

        return  step_size

    def _set_optimizer_properties(self, opt_prob, options_keys=[], opt_settings_keys=[], mapped_keys={}):
        """Set the optimizer properties, both the `driver.options` and
        `driver.opt_settings`. See OpenMDAO documentation on drivers
        to determine which settings are set by either options or
        opt_settings.

        Args:
            opt_prob (OpenMDAO problem object):  The hybrid plant OpenMDAO problem object.
            options_keys (list, optional): List of keys for driver opt_settings to be set. Defaults to [].
            opt_settings_keys (list, optional): List of keys for driver options to be set. Defaults to [].
            mapped_keys (dict, optional): Key pairs where the yaml name differs from what's expected
                                          by the driver. Specifically, the key is what's given in the yaml
                                          and the value is what's expected by the driver. Defaults to {}.

        Returns:
            opt_prob (OpenMDAO problem object): The updated openmdao problem object with driver settings applied.
        """

        opt_options = self.config.greenheart_config["opt_options"]["driver"]["optimization"]

        # Loop through all of the options provided and set them in the OM driver object
        for key in options_keys:
            if key in opt_options:
                if key in mapped_keys:
                    opt_prob.driver.options[mapped_keys[key]] = opt_options[key]
                else:
                    opt_prob.driver.options[key] = opt_options[key]

        # Loop through all of the opt_settings provided and set them in the OM driver object
        for key in opt_settings_keys:
            if key in opt_options:
                if key in mapped_keys:
                    opt_prob.driver.opt_settings[mapped_keys[key]] = opt_options[key]
                else:
                    opt_prob.driver.opt_settings[key] = opt_options[key]

        return opt_prob

    def set_driver(self, opt_prob):
        """set which optimization driver to use and set options

        Args:
            opt_prob (openmdao problem instance): openmdao problem class instance for current optimization problem

        Raises:
            ImportError: An optimization algorithm from pyoptsparse was selected, but pyoptsparse is not installed
            ImportError: An optimization algorithm from pyoptsparse was selected, but the algorithm code is not currently installed within pyoptsparse
            ImportError: An optimization algorithm was requested from NLopt, but NLopt is not currently installed.
            ValueError: The selected optimizer is not yet supported.
            Exception: The specified generator type for the OpenMDAO design of experiments is unsupported.

        Returns:
            opt_prob (openmdao problem instance): openmdao problem class instance, edited from input with desired driver and driver options
        """

        folder_output = self.config.greenheart_config["opt_options"]["general"]["folder_output"]

        if self.config.greenheart_config["opt_options"]["driver"]["optimization"]["flag"]:
            opt_options = self.config.greenheart_config["opt_options"]["driver"]["optimization"]
            step_size = self._get_step_size()

            if "step_calc" in opt_options.keys():
                if opt_options["step_calc"] == "None":
                    step_calc = None
                else:
                    step_calc = opt_options["step_calc"]
            else:
                step_calc = None

            if "form" in opt_options.keys():
                if opt_options["form"] == "None":
                    form = None
                else:
                    form = opt_options["form"]
            else:
                form = None

            opt_prob.model.approx_totals(method="fd", step=step_size, form=form, step_calc=step_calc)

            # Set optimization solver and options. First, Scipy's SLSQP and COBYLA
            if opt_options["solver"] in self.scipy_methods:
                opt_prob.driver = om.ScipyOptimizeDriver()
                opt_prob.driver.options["optimizer"] = opt_options["solver"]

                options_keys = ["tol", "max_iter", "disp"]
                opt_settings_keys = ["rhobeg", "catol", "adaptive"]
                mapped_keys = {"max_iter": "maxiter"}
                opt_prob = self._set_optimizer_properties(opt_prob, options_keys, opt_settings_keys, mapped_keys)

            # The next two optimization methods require pyOptSparse.
            elif opt_options["solver"] in self.pyoptsparse_methods:
                try:
                    from openmdao.api import pyOptSparseDriver
                except:
                    raise ImportError(
                        f"You requested the optimization solver {opt_options['solver']}, but you have not installed pyOptSparse. Please do so and rerun."
                    )
                opt_prob.driver = pyOptSparseDriver(gradient_method=opt_options["gradient_method"])
                
                try:
                    opt_prob.driver.options["optimizer"] = opt_options["solver"]
                except:
                    raise ImportError(
                        f"You requested the optimization solver {opt_options['solver']}, but you have not installed it within pyOptSparse. Please build {opt_options['solver']} and rerun."
                    )

                # Most of the pyOptSparse options have special syntax when setting them,
                # so here we set them by hand instead of using `_set_optimizer_properties` for SNOPT and CONMIN.
                if opt_options["solver"] == "CONMIN":
                    opt_prob.driver.opt_settings["ITMAX"] = opt_options["max_iter"]

                if opt_options["solver"] == "NSGA2":
                    opt_settings_keys = [
                        "PopSize",
                        "maxGen",
                        "pCross_real",
                        "pMut_real",
                        "eta_c",
                        "eta_m",
                        "pCross_bin",
                        "pMut_bin",
                        "PrintOut",
                        "seed",
                        "xinit",
                    ]
                    opt_prob = self._set_optimizer_properties(opt_prob, opt_settings_keys=opt_settings_keys)

                elif opt_options["solver"] == "SNOPT":
                    opt_prob.driver.opt_settings["Major optimality tolerance"] = float(opt_options["tol"])
                    opt_prob.driver.opt_settings["Major iterations limit"] = int(opt_options["max_major_iter"])
                    opt_prob.driver.opt_settings["Iterations limit"] = int(opt_options["max_minor_iter"])
                    opt_prob.driver.opt_settings["Major feasibility tolerance"] = float(opt_options["tol"])
                    if "time_limit" in opt_options:
                        opt_prob.driver.opt_settings["Time limit"] = int(opt_options["time_limit"])
                    opt_prob.driver.opt_settings["Summary file"] = os.path.join(folder_output, "SNOPT_Summary_file.txt")
                    opt_prob.driver.opt_settings["Print file"] = os.path.join(folder_output, "SNOPT_Print_file.txt")
                    if "hist_file_name" in opt_options:
                        opt_prob.driver.hist_file = opt_options["hist_file_name"]
                    if "verify_level" in opt_options:
                        opt_prob.driver.opt_settings["Verify level"] = opt_options["verify_level"]
                    else:
                        opt_prob.driver.opt_settings["Verify level"] = -1
                if "hotstart_file" in opt_options:
                    opt_prob.driver.hotstart_file = opt_options["hotstart_file"]

            elif opt_options["solver"] == "GA":
                opt_prob.driver = om.SimpleGADriver()
                options_keys = [
                    "Pc",
                    "Pm",
                    "bits",
                    "compute_pareto",
                    "cross_bits",
                    "elitism",
                    "gray",
                    "max_gen",
                    "multi_obj_exponent",
                    "multi_obj_weights",
                    "penalty_exponent",
                    "penalty_parameter",
                    "pop_size",
                    "procs_per_model",
                    "run_parallel",
                ]
                opt_prob = self._set_optimizer_properties(opt_prob, options_keys)

            elif opt_options["solver"] in self.nlopt_methods:
                try:
                    from wisdem.optimization_drivers.nlopt_driver import NLoptDriver
                except:
                    raise ImportError(
                        "You requested an optimization method from NLopt, but need to first install NLopt to use this method."
                    )

                opt_prob.driver = NLoptDriver()
                opt_prob.driver.options["optimizer"] = opt_options["solver"]
                options_keys = ["tol", "xtol", "max_iter", "max_time", "numgen"]
                mapped_keys = {"max_iter": "maxiter", "max_time": "maxtime"}
                opt_prob = self._set_optimizer_properties(opt_prob, options_keys, mapped_keys=mapped_keys)

            else:
                raise ValueError(f"The {self.config.greenheart_config['opt_options']['driver']['optimization']['solver']} optimizer is not yet supported!")

            if opt_options["debug_print"]:
                opt_prob.driver.options["debug_print"] = ["desvars", "ln_cons", "nl_cons", "objs", "totals"]

        elif self.config.greenheart_config["opt_options"]["driver"]["design_of_experiments"]["flag"]:
            doe_options = self.config.greenheart_config["opt_options"]["driver"]["design_of_experiments"]
            if doe_options["generator"].lower() == "uniform":
                generator = om.UniformGenerator(
                    num_samples=int(doe_options["num_samples"]),
                    seed=doe_options["seed"],
                )
            elif doe_options["generator"].lower() == "fullfact":
                generator = om.FullFactorialGenerator(levels=int(doe_options["levels"]))
            elif doe_options["generator"].lower() == "plackettburman":
                generator = om.PlackettBurmanGenerator()
            elif doe_options["generator"].lower() == "boxbehnken":
                generator = om.BoxBehnkenGenerator()
            elif doe_options["generator"].lower() == "latinhypercube":
                generator = om.LatinHypercubeGenerator(
                    samples=int(doe_options["num_samples"]),
                    criterion=doe_options["criterion"],
                    seed=doe_options["seed"],
                )
            else:
                raise Exception("The generator type {} is unsupported.".format(doe_options["generator"]))

            # Initialize driver
            opt_prob.driver = om.DOEDriver(generator)

            if doe_options["debug_print"]:
                opt_prob.driver.options["debug_print"] = ["desvars", "ln_cons", "nl_cons", "objs"]

            # options
            opt_prob.driver.options["run_parallel"] = doe_options["run_parallel"]

        elif self.config.greenheart_config["opt_options"]["driver"]["step_size_study"]["flag"]:
            pass

        else:
            print("WARNING: Design variables are set to be optimized or studied, but no driver is selected.")
            print("         If you want to run an optimization, please enable a driver.")

        return opt_prob

    def set_objective(self, opt_prob):
        """Set merit figure. Each objective has its own scaling.  Check first for user override

        Args:
            opt_prob (openmdao problem instance): openmdao problem instance for current optimization problem

        Returns:
            opt_prob (openmdao problem instance): openmdao problem instance for current optimization problem with objective set
        """
        # 
        if self.config.greenheart_config["opt_options"]["merit_figure_user"]["name"] != "":
            coeff = -1.0 if self.config.greenheart_config["opt_options"]["merit_figure_user"]["max_flag"] else 1.0
            opt_prob.model.add_objective(self.config.greenheart_config["opt_options"]["merit_figure_user"]["name"],
                                       ref=coeff*np.abs(self.config.greenheart_config["opt_options"]["merit_figure_user"]["ref"]))

        return opt_prob

    def set_design_variables(self, opt_prob, config, hi):
        """Set optimization design variables.

        Args:
            opt_prob (openmdao problem instance): openmdao problem instance for current optimization problem
            config (GreenHeartSimulationConfig): data class containing modeling, simulation, and optimization settings
            hi (HoppInterface): Main HOPP class

        Returns:
            opt_prob (openmdao problem instance): openmdao problem instance for current optimization problem with design variables set
        """
        
        design_variables_dict = {}
        for key in self.config.greenheart_config["opt_options"]["design_variables"].keys():
            if self.config.greenheart_config["opt_options"]["design_variables"][key]["flag"]:
                design_variables_dict[key] = config.greenheart_config["opt_options"]["design_variables"][key]
            
        print("ADDING DESIGN VARIABLES:")
        for dv, d in design_variables_dict.items():
            print(f"   {dv}")
            opt_prob.model.add_design_var(dv, lower=d["lower"], upper=d["upper"], units=d["units"])

        return opt_prob
    
    def set_constraints(self, opt_prob, hi: Optional[Union[None, HoppInterface]] = None):
        """sets up optimization constraints for the greenheart optimization problem

        Args:
            opt_prob (openmdao problem instance): openmdao problem instance for current optimization problem
            hi (Optional[Union[None, HoppInterface]], optional): Accepts an instance of the HoppInterface class (required when x and y are design variables). Defaults to None.

        Raises:
            Exception: all design variables must have at least one of an upper and lower bound specified

        Returns:
            opt_prob (openmdao problem instance): openmdao problem instance for current optimization problem edited to include constraint setup
        """

        if (hi is not None) and (self.config.hopp_config["technologies"]["wind"]["model_name"] == "floris"):
            turbine_x_init = hi.system.wind.config.floris_config["farm"]["layout_x"]
            turbine_y_init = hi.system.wind.config.floris_config["farm"]["layout_y"]
        else:
            # randomly generate initial turbine locations if not provided
            turbine_x_init = 1E3*np.random.rand(self.config.hopp_config["technologies"]["wind"]["num_turbines"])
            turbine_y_init = 1E3*np.random.rand(self.config.hopp_config["technologies"]["wind"]["num_turbines"])

        # turbine spacing constraint
        if self.config.greenheart_config["opt_options"]["constraints"]["turbine_spacing"]["flag"]:
            lower = self.config.greenheart_config["opt_options"]["constraints"]["turbine_spacing"]["lower"]
            
            opt_prob.model.add_subsystem("con_spacing", subsys=TurbineDistanceComponent(turbine_x_init=turbine_x_init, turbine_y_init=turbine_y_init), promotes=["*"])
            opt_prob.model.add_constraint("spacing_vec", lower=lower)
        
        # bondary distance constraint
        if self.config.greenheart_config["opt_options"]["constraints"]["boundary_distance"]["flag"]:
            lower = self.config.greenheart_config["opt_options"]["constraints"]["boundary_distance"]["lower"]
            opt_prob.model.add_subsystem("con_boundary", subsys=BoundaryDistanceComponent(hopp_interface=self.config.greenheart_config, turbine_x_init=turbine_x_init, turbine_y_init=turbine_y_init), promotes=["*"])
            opt_prob.model.add_constraint("boundary_distance_vec", lower=lower)

        # solar/platform size
        if self.config.greenheart_config["opt_options"]["constraints"]["pv_to_platform_area_ratio"]["flag"]:
            upper = self.config.greenheart_config["opt_options"]["constraints"]["pv_to_platform_area_ratio"]["upper"]
            opt_prob.model.add_subsystem("con_pv_platform_area", subsys=om.ExecComp(['pv_platform_ratio=pv_area/platform_area']), promotes=["*"])
            opt_prob.model.add_constraint("pv_platform_ratio", upper=upper)

        # User constraints
        user_constr = self.config.greenheart_config["opt_options"]["constraints"]["user"]
        for k in range(len(user_constr)):
            var_k = user_constr[k]["name"]
            
            if "lower_bound" in user_constr[k]:
                lower_k = user_constr[k]["lower_bound"]
            elif "lower" in user_constr[k]:
                lower_k = user_constr[k]["lower"]
            else:
                lower_k = None
                
            if "upper_bound" in user_constr[k]:
                upper_k = user_constr[k]["upper_bound"]
            elif "upper" in user_constr[k]:
                lower_k = user_constr[k]["upper"]
            else:
                upper_k = None
                
            if "indices" in user_constr[k]:
                idx_k = user_constr[k]["indices"]
            else:
                idx_k = None

            if lower_k is None and upper_k is None:
                raise Exception(f"Must include a lower_bound and/or an upper bound for {var_k}")
            
            opt_prob.model.add_constraint(var_k, lower=lower_k, upper=upper_k, indices=idx_k)
        
        return opt_prob

    def set_recorders(self, opt_prob):
        """sets up a recorder for the openmdao problem as desired in the input yaml

        Args:
            opt_prob (openmdao problem instance): openmdao problem instance for current optimization problem

        Returns:
            opt_prob (openmdao problem instance): openmdao problem instance for current optimization problem edited to include a set up recorder
        """
        folder_output = self.config.greenheart_config["opt_options"]["general"]["folder_output"]

        # Set recorder on the OpenMDAO driver level using the `optimization_log`
        # filename supplied in the optimization yaml
        if self.config.greenheart_config["opt_options"]["recorder"]["flag"]:
            recorder = om.SqliteRecorder(os.path.join(folder_output, self.config.greenheart_config["opt_options"]["recorder"]["file_name"]))
            opt_prob.driver.add_recorder(recorder)
            opt_prob.add_recorder(recorder)

            opt_prob.driver.recording_options["excludes"] = ["*_df"]
            opt_prob.driver.recording_options["record_constraints"] = True
            opt_prob.driver.recording_options["record_desvars"] = True
            opt_prob.driver.recording_options["record_objectives"] = True

            if self.config.greenheart_config["opt_options"]["recorder"]["includes"]:
                opt_prob.driver.recording_options["includes"] = self.config.greenheart_config["opt_options"]["recorder"]["includes"]

        return opt_prob

    def set_initial(self, opt_prob, config):
        
        return opt_prob

    def set_restart(self, opt_prob):
        """prepares to restart from last recorded iteration if the original problem was set up for warm start

        Args:
            opt_prob (openmdao problem instance): openmdao problem instance for current optimization problem

        Returns:
            opt_prob (openmdao problem instance): openmdao problem instance for current optimization problem set up for warm start
        """

        if "warmstart_file" in self.config.greenheart_config["opt_options"]["driver"]["optimization"]:
            # Directly read the pyoptsparse sqlite db file
            from pyoptsparse import SqliteDict

            db = SqliteDict(self.config.greenheart_config["opt_options"]["driver"]["optimization"]["warmstart_file"])

            # Grab the last iteration's design variables
            last_key = db["last"]
            desvars = db[last_key]["xuser"]

            # Obtain the already-setup OM problem's design variables
            if opt_prob.model._static_mode:
                design_vars = opt_prob.model._static_design_vars
            else:
                design_vars = opt_prob.model._design_vars

            # Get the absolute names from the promoted names within the OM model.
            # We need this because the pyoptsparse db has the absolute names for
            # variables but the OM model uses the promoted names.
            prom2abs = opt_prob.model._var_allprocs_prom2abs_list["output"]
            abs2prom = {}
            for key in design_vars:
                abs2prom[prom2abs[key][0]] = key

            # Loop through each design variable
            for key in desvars:
                prom_key = abs2prom[key]

                # Scale each DV based on the OM scaling from the problem.
                # This assumes we're running the same problem with the same scaling
                scaler = design_vars[prom_key]["scaler"]
                adder = design_vars[prom_key]["adder"]

                if scaler is None:
                    scaler = 1.0
                if adder is None:
                    adder = 0.0

                scaled_dv = desvars[key] / scaler - adder

        return opt_prob