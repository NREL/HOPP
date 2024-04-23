"""
This file is based on the WISDEM file of the same name
"""

import os

import numpy as np
from typing import Optional, Union

import openmdao.api as om


from greenheart.tools.optimization.openmdao import TurbineDistanceComponent, BoundaryDistanceComponent
from hopp.simulation import HoppInterface

class PoseOptimization(object):
    def __init__(self, config):
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
        # Determine the number of design variables
        n_DV = 0

        pv_opt = self.config.greenheart_config["opt_options"]["design_variables"]["pv"]
        
        if pv_opt["flag"]:
            n_DV += 1
        
        # Wrap-up at end with multiplier for finite differencing
        if self.config.greenheart_config["opt_options"]["driver"]["optimization"]["form"] == "central": # TODO this should probably be handled at the MPI point to avoid confusion with n_DV being double what would be expected
            n_DV *= 2

        return n_DV

    def _get_step_size(self):
        # If a step size for the driver-level finite differencing is provided, use that step size. Otherwise use a default value.
        return (
            1.0e-6
            if not "step_size" in self.config.greenheart_config["opt_options"]["driver"]["optimization"]
            else self.config.greenheart_config["opt_options"]["driver"]["optimization"]["step_size"]
        )

    def _set_optimizer_properties(self, wt_opt, options_keys=[], opt_settings_keys=[], mapped_keys={}):
        """
        Set the optimizer properties, both the `driver.options` and
        `driver.opt_settings`. See OpenMDAO documentation on drivers
        to determine which settings are set by either options or
        opt_settings.

        Parameters
        ----------
        wt_opt : OpenMDAO problem object
            The wind turbine problem object.
        options_keys : list
            List of keys for driver options to be set.
        opt_settings_keys: list
            List of keys for driver opt_settings to be set.
        mapped_keys: dict
            Key pairs where the yaml name differs from what's expected
            by the driver. Specifically, the key is what's given in the yaml
            and the value is what's expected by the driver.

        Returns
        -------
        wt_opt : OpenMDAO problem object
            The updated wind turbine problem object with driver settings applied.
        """

        opt_options = self.config.greenheart_config["opt_options"]["driver"]["optimization"]

        # Loop through all of the options provided and set them in the OM driver object
        for key in options_keys:
            if key in opt_options:
                if key in mapped_keys:
                    wt_opt.driver.options[mapped_keys[key]] = opt_options[key]
                else:
                    wt_opt.driver.options[key] = opt_options[key]

        # Loop through all of the opt_settings provided and set them in the OM driver object
        for key in opt_settings_keys:
            if key in opt_options:
                if key in mapped_keys:
                    wt_opt.driver.opt_settings[mapped_keys[key]] = opt_options[key]
                else:
                    wt_opt.driver.opt_settings[key] = opt_options[key]

        return wt_opt

    def set_driver(self, wt_opt):
        folder_output = self.config.greenheart_config["opt_options"]["general"]["folder_output"]

        if self.config.greenheart_config["opt_options"]["driver"]["optimization"]["flag"]:
            opt_options = self.config.greenheart_config["opt_options"]["driver"]["optimization"]
            step_size = self._get_step_size()

            if opt_options["step_calc"] == "None":
                step_calc = None
            else:
                step_calc = opt_options["step_calc"]
            wt_opt.model.approx_totals(method="fd", step=step_size, form=opt_options["form"], step_calc=step_calc)

            # Set optimization solver and options. First, Scipy's SLSQP and COBYLA
            if opt_options["solver"] in self.scipy_methods:
                wt_opt.driver = om.ScipyOptimizeDriver()
                wt_opt.driver.options["optimizer"] = opt_options["solver"]

                options_keys = ["tol", "max_iter", "disp"]
                opt_settings_keys = ["rhobeg", "catol", "adaptive"]
                mapped_keys = {"max_iter": "maxiter"}
                wt_opt = self._set_optimizer_properties(wt_opt, options_keys, opt_settings_keys, mapped_keys)

            # The next two optimization methods require pyOptSparse.
            elif opt_options["solver"] in self.pyoptsparse_methods:
                try:
                    from openmdao.api import pyOptSparseDriver
                except:
                    raise ImportError(
                        f"You requested the optimization solver {opt_options['solver']}, but you have not installed pyOptSparse. Please do so and rerun."
                    )
                wt_opt.driver = pyOptSparseDriver(gradient_method=opt_options["gradient_method"])
                
                try:
                    wt_opt.driver.options["optimizer"] = opt_options["solver"]
                except:
                    raise ImportError(
                        f"You requested the optimization solver {opt_options['solver']}, but you have not installed it within pyOptSparse. Please build {opt_options['solver']} and rerun."
                    )

                # Most of the pyOptSparse options have special syntax when setting them,
                # so here we set them by hand instead of using `_set_optimizer_properties` for SNOPT and CONMIN.
                if opt_options["solver"] == "CONMIN":
                    wt_opt.driver.opt_settings["ITMAX"] = opt_options["max_iter"]

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
                    wt_opt = self._set_optimizer_properties(wt_opt, opt_settings_keys=opt_settings_keys)

                elif opt_options["solver"] == "SNOPT":
                    wt_opt.driver.opt_settings["Major optimality tolerance"] = float(opt_options["tol"])
                    wt_opt.driver.opt_settings["Major iterations limit"] = int(opt_options["max_major_iter"])
                    wt_opt.driver.opt_settings["Iterations limit"] = int(opt_options["max_minor_iter"])
                    wt_opt.driver.opt_settings["Major feasibility tolerance"] = float(opt_options["tol"])
                    if "time_limit" in opt_options:
                        wt_opt.driver.opt_settings["Time limit"] = int(opt_options["time_limit"])
                    wt_opt.driver.opt_settings["Summary file"] = os.path.join(folder_output, "SNOPT_Summary_file.txt")
                    wt_opt.driver.opt_settings["Print file"] = os.path.join(folder_output, "SNOPT_Print_file.txt")
                    if "hist_file_name" in opt_options:
                        wt_opt.driver.hist_file = opt_options["hist_file_name"]
                    if "verify_level" in opt_options:
                        wt_opt.driver.opt_settings["Verify level"] = opt_options["verify_level"]
                    else:
                        wt_opt.driver.opt_settings["Verify level"] = -1
                if "hotstart_file" in opt_options:
                    wt_opt.driver.hotstart_file = opt_options["hotstart_file"]

            elif opt_options["solver"] == "GA":
                wt_opt.driver = om.SimpleGADriver()
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
                wt_opt = self._set_optimizer_properties(wt_opt, options_keys)

            elif opt_options["solver"] in self.nlopt_methods:
                try:
                    from wisdem.optimization_drivers.nlopt_driver import NLoptDriver
                except:
                    raise ImportError(
                        "You requested an optimization method from NLopt, but need to first install NLopt to use this method."
                    )

                wt_opt.driver = NLoptDriver()
                wt_opt.driver.options["optimizer"] = opt_options["solver"]
                options_keys = ["tol", "xtol", "max_iter", "max_time", "numgen"]
                mapped_keys = {"max_iter": "maxiter", "max_time": "maxtime"}
                wt_opt = self._set_optimizer_properties(wt_opt, options_keys, mapped_keys=mapped_keys)

            else:
                raise ValueError(f"The {self.config.greenheart_config['opt_options']['driver']['optimization']['solver']} optimizer is not yet supported!")

            if opt_options["debug_print"]:
                wt_opt.driver.options["debug_print"] = ["desvars", "ln_cons", "nl_cons", "objs", "totals"]

        elif self.config.greenheart_config["opt_options"]["driver"]["design_of_experiments"]["flag"]:
            doe_options = self.config.greenheart_config["opt_options"]["driver"]["design_of_experiments"]
            if doe_options["generator"].lower() == "uniform":
                generator = om.UniformGenerator(
                    num_samples=int(doe_options["num_samples"]),
                    seed=doe_options["seed"],
                )
            elif doe_options["generator"].lower() == "fullfact":
                generator = om.FullFactorialGenerator(levels=int(doe_options["num_samples"]))
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
            wt_opt.driver = om.DOEDriver(generator)

            if doe_options["debug_print"]:
                wt_opt.driver.options["debug_print"] = ["desvars", "ln_cons", "nl_cons", "objs"]

            # options
            wt_opt.driver.options["run_parallel"] = doe_options["run_parallel"]

        elif self.config.greenheart_config["opt_options"]["driver"]["step_size_study"]["flag"]:
            pass

        else:
            print("WARNING: Design variables are set to be optimized or studied, but no driver is selected.")
            print("         If you want to run an optimization, please enable a driver.")

        return wt_opt

    def set_objective(self, wt_opt):
        # Set merit figure. Each objective has its own scaling.  Check first for user override
        if self.config.greenheart_config["opt_options"]["merit_figure_user"]["name"] != "":
            coeff = -1.0 if self.config.greenheart_config["opt_options"]["merit_figure_user"]["max_flag"] else 1.0
            wt_opt.model.add_objective(self.config.greenheart_config["opt_options"]["merit_figure_user"]["name"],
                                       ref=coeff*np.abs(self.config.greenheart_config["opt_options"]["merit_figure_user"]["ref"]))

        return wt_opt

    def set_design_variables(self, wt_opt, config, hi):
        # Set optimization design variables.
        design_variables_dict = {}
        for key in self.config.greenheart_config["opt_options"]["design_variables"].keys():
            if self.config.greenheart_config["opt_options"]["design_variables"][key]["flag"]:
                design_variables_dict[key] = config.greenheart_config["opt_options"]["design_variables"][key]
            
        print("ADDING DESIGN VARIABLES:")
        for dv, d in design_variables_dict.items():
            print(f"   {dv}")
            wt_opt.model.add_design_var(dv, lower=d["lower"], upper=d["upper"], units=d["units"])

        return wt_opt
    
    def set_constraints(self, wt_opt, hi: Optional[Union[None, HoppInterface]] = None):

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
            
            wt_opt.model.add_subsystem("con_spacing", subsys=TurbineDistanceComponent(turbine_x_init=turbine_x_init, turbine_y_init=turbine_y_init), promotes=["*"])
            wt_opt.model.add_constraint("spacing_vec", lower=lower)
        
        # bondary distance constraint
        if self.config.greenheart_config["opt_options"]["constraints"]["boundary_distance"]["flag"]:
            lower = self.config.greenheart_config["opt_options"]["constraints"]["boundary_distance"]["lower"]
            wt_opt.model.add_subsystem("con_boundary", subsys=BoundaryDistanceComponent(hopp_interface=self.config.greenheart_config, turbine_x_init=turbine_x_init, turbine_y_init=turbine_y_init), promotes=["*"])
            wt_opt.model.add_constraint("boundary_distance_vec", lower=0)

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
            
            wt_opt.model.add_constraint(var_k, lower=lower_k, upper=upper_k, indices=idx_k)
        
        return wt_opt

    def set_recorders(self, wt_opt):
        folder_output = self.config.greenheart_config["opt_options"]["general"]["folder_output"]

        # Set recorder on the OpenMDAO driver level using the `optimization_log`
        # filename supplied in the optimization yaml
        if self.config.greenheart_config["opt_options"]["recorder"]["flag"]:
            recorder = om.SqliteRecorder(os.path.join(folder_output, self.config.greenheart_config["opt_options"]["recorder"]["file_name"]))
            wt_opt.driver.add_recorder(recorder)
            wt_opt.add_recorder(recorder)

            wt_opt.driver.recording_options["excludes"] = ["*_df"]
            wt_opt.driver.recording_options["record_constraints"] = True
            wt_opt.driver.recording_options["record_desvars"] = True
            wt_opt.driver.recording_options["record_objectives"] = True

            if self.config.greenheart_config["opt_options"]["recorder"]["includes"]:
                wt_opt.driver.recording_options["includes"] = self.config.greenheart_config["opt_options"]["recorder"]["includes"]

        return wt_opt

    def set_initial(self, wt_opt, config):
        
        return wt_opt

    def set_restart(self, wt_opt):
        if "warmstart_file" in self.config.greenheart_config["opt_options"]["driver"]["optimization"]:
            # Directly read the pyoptsparse sqlite db file
            from pyoptsparse import SqliteDict

            db = SqliteDict(self.config.greenheart_config["opt_options"]["driver"]["optimization"]["warmstart_file"])

            # Grab the last iteration's design variables
            last_key = db["last"]
            desvars = db[last_key]["xuser"]

            # Obtain the already-setup OM problem's design variables
            if wt_opt.model._static_mode:
                design_vars = wt_opt.model._static_design_vars
            else:
                design_vars = wt_opt.model._design_vars

            # Get the absolute names from the promoted names within the OM model.
            # We need this because the pyoptsparse db has the absolute names for
            # variables but the OM model uses the promoted names.
            prom2abs = wt_opt.model._var_allprocs_prom2abs_list["output"]
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

                # Special handling for blade twist as we only have the
                # last few control points as design variables
                if "twist_opt" in key:
                    wt_opt[key][2:] = scaled_dv
                else:
                    wt_opt[key][:] = scaled_dv

        return wt_opt