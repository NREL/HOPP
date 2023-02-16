.. _dispatch_solar_study:


Dispatchable Solar Case Study
=============================

This is a tutorial to evaluate dispatchable solar projects using HOPP. This includes 
any combinations of the following technologies:

    #. Concentrating solar power (CSP) molten salt power tower
    #. CSP parabolic trough
    #. Photovoltaic
    #. Electro-chemical batteries

Contents:

.. contents::
   :local:
   :depth: 2

Folder structure
----------------

Below is the folder structure of the provided dispatchable solar example case study. 

| CSP_PV_Battery_Analysis
| ├── 01_help_docs
| │   ├── financial_parameter_names_definitions.xlsx
| │   └── tower_receiver_cost_scaling.xlsx
| ├── 02_weather_data
| │   ├── Blythe_CA
| │   ├── Daggett_CA
| │   ├── Imperial_CA
| │   ├── Phoenix_AZ
| │   └── Tucson_AZ
| ├── 03_cost_load_price_data
| │   ├── constant_norm_prices.csv
| │   ├── desired_schedule_normalized.csv
| │   ├── financial_parameters_SAM.json
| │   ├── PG&E2016_norm_prices.csv
| │   └── system_costs_SAM.json
| ├── simulation_init.py
| ├── print_output.py
| ├── 1_single_plant.py
| ├── 2_parametric_study.py
| ├── 3_optimize.py
| └── 4_parallel_sample_optimize.py

File overview
^^^^^^^^^^^^^

In the folder structure above:

* ``01_help_docs`` contains helpful documentation and files for understanding parameter names within HOPP.
    
    * ``help_docs/financial_parameter_names_definitions.xlsx`` contains a list of financial parameters for the single owner SAM model containing name, descriptions, units, SAM default values, and notes. These are organized according to the SAM pages within the GUI.
    * ``help_docs/tower_receiver_cost_scaling.xlsx`` is an interactive tool to visualize the cost curves for the CSP tower and receiver base on your cost assumptions parameters.

* ``02_weather_data`` contains weather data of the 5 example locations (CA: Blythe, Daggett, and Imperial, AZ: Phoenix and Tucson)
* ``03_cost_load_price_data`` contains all HOPP input data except for weather data. This includes financial assumptions around grid prices, desired load profile, financial parameters, and system costs.
    
    * ``input_data/constant_norm_prices.csv`` contains hourly normalized prices that are constant across the year (i.e., 1 for all hours of the year)
    * ``input_data/desired_schedule_normalized.csv`` is an example of a desired load profile (8760 hours for the year) normalized between zero and one.
    * ``input_data/financial_parameters_SAM.json`` contains all of the financial parameters (except for system costs) used in the single owner financial model. The values within this file correspond to the default values found in the SAM GUI. Please use ``help_docs/financial_parameter_names_definitions.xlsx`` and `SAM's help <https://samrepo.nrelcloud.org/help/index.html>`_ for more information.
    * ``input_data/PG&E2016_norm_prices.csv`` contains hourly normalized prices for the PG&E 2016 SAM example.
    * ``input_data/system_costs_SAM.json`` contains the system cost assumptions for each of the modeled technologies. This includes battery replacement schedule.

* ``simulation_init.py`` contains the ``init_hybrid_plant`` `function` and ``DesignProblem`` class. The former is used to initialize the hybrid simulation based on the technologies to simulate and case study specific inputs. The latter is used to set up sampling and optimization.
* ``print_output.py`` contains some help printing functions for print simulation results to the console.
* ``1_single_plant.py`` contains a simple script for running various system configurations with a fixed sizing.
* ``2_parametric_study.py`` contains two methods for generating design samples and executes samples in parallel.
* ``3_optimize.py`` contains a basic script for setting up and executing an optimization on high-level design sizing variables.
* ``4_parallel_sample_optimize.py`` contains a script that can sample the design space in parallel and then optimize based on the provided sample results.

Setting Up a Specific Case Study
--------------------------------

In this section, we will go through setting up a case study and explain the various work flows available to the users. 
While the intent of this work is to try to provide the user a method for evaluating various dispatchable solar systems, it is not by any means comprehensive. 
The HOPP framework, by design, is somewhat flexible in the way users can script various analyses for their specific use case. That being said, users have the freedom to modify this case study as they see fit.

Input data
^^^^^^^^^^

Before running sampling and/or optimization on design sizing variables, it is imperative to understand and validate the case study inputs. This includes but not limited to the following (in no particular order):

#. Weather data: provides the performance models information about the available solar resource for a specific location. Weather data can come in many different types (i.e., typical meteorological year (TMY), historic year, P50, P90, etc.). It is important to understand the implications of using these various weather file types.
#. Time-of-delivery prices of energy: provides the dispatch model an economic incentive to generation power during the highest-valued periods.
#. Desired load profile: provides the dispatch model a system desired load profile. The dispatch model will minimize system operating cost including the cost of missing load.
#. Financial parameters: provides the underlining project financial assumptions that will determine the economic value of the system's performance. These parameters will be dictated by project financing and potential revenue stream. See `SAM's help <https://samrepo.nrelcloud.org/help/index.html>`_ for additional information.  
#. System costs assumptions: provides HOPP parameters for scaling total install costs. These parameters will have a significant impact on system sizing and the trade-off between technologies.  

.. note::
    Modeling golden rule: **`Garbage in, Garbage out`** meaning that no model can overcome nonsense or bad input data.

Setting up a hybrid simulation initialize function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This function provides a method for the iteration methods (design sampling or optimization) to initialize a hybrid simulation object using specific project inputs.
An example of this function can be found in ``simulation_init.py`` called ``init_hybrid_plant``. Users can change this function to model their specific case study assumptions. 
If the users wanted to have multiple case studies with different costs or financial assumptions, the user could create multiple simulation initialize 
functions and `point` the ``DesignProblem`` to the specific function they would like to analyze (Note, this is an advance technique and introduces the potential for user error).

Next we are going to discuss the specific ``init_hybrid_plant`` `function` provided within the case study.

1. Set up the :class:`hybrid.sites.site_info.SiteInfo` class

    In our example, we provide the following inputs to ``SiteInfo``: ``site_data``, ``solar_resource_file``, ``grid_resource_file``, ``desired_schedule``.

    * ``site_data``: `dictionary`, containing site location information.

        .. note::
            ``site_data`` must be provided to ``SiteInfo``; however, the keys ``lat``, ``lon``, ``elev``, ``tz`` do nothing when the ``solar_resource_file`` is provided directly. 
            When ``solar_resource_file`` is not provided, HOPP can uses these location parameters to download resource information from `NSRDB <https://maps.nrel.gov/nsrdb-viewer/>`_, if available. 

    * ``solar_resource_file``: `csv file`, contain site annual weather resource.

        .. note::
            Currently, HOPP does not have a robust weather file format checking as a result the CSP models can be sensitive to the formatting of the weather data.
            If the weather data is not appropriately formatted, the CSP model can `soft` error where the model does not fail to execute but the annual generation is negative.
            If this occurs after changing the weather file, please look at the formatting of the example weather files and correct the new file appropriately.

    * ``grid_resource_file``: `csv file`, contains hourly normalized grid prices that will be scaled by ``ppa_price`` of the :class:`hybrid.hybrid_simulation.HybridSimulation` `class`.
    * ``desired_schedule``: `csv file`, contains hourly normalized desired system generation profile and is scaled by ``schedule_scale`` within ``init_hybrid_plant`` `function`.

    In our example, the ``SiteInfo`` class is initialized by the following:

    .. code-block::

        site = SiteInfo(site_data, 
                        solar_resource_file=solar_file, 
                        grid_resource_file=prices_file,
                        desired_schedule=desired_schedule
                        )


2. Set up the :class:`hybrid.hybrid_simulation.HybridSimulation` class

    The ``HybridSimulation`` class is HOPP's main class that handles the simulation of a specific hybrid design.
    In our example, we provide the following inputs to ``HybridSimulation``: ``sim_techs``, ``site``, ``dispatch_options``, ``cost_info``.

    * ``sim_techs``: `nested dictionary`, contains the technologies names to simulate (first level of keys) and their specific configuration dictionaries (second level of keys). 
    
        .. note::
            For each technology type, only specific configuration keys are required and used. The technology configuration dictionary cannot apply general SAM parameter values.
            This must be done after the ``HybridSimulation`` is initialized. See the specific technology classes for more details on specific configuration keys required and used.

        In our example, we have a two methods for setting this dictionary.
        
            1. Using the values that are hard coded within the ``init_hybrid_plant`` `function`. This is done by passing a list of technologies through the ``init_hybrid_plant`` `function` input ``techs_in_sim``. Then, the following will select only the technologies provided in ``techs_in_sim``.

            .. code-block::

                sim_techs = {key: technologies[key] for key in techs_in_sim}

            2. Providing a user-defined ``technologies`` nested dictionary. This is an optional input ``ud_techs`` for the ``init_hybrid_plant`` `function` and will overwrite the default dictionary.

            .. note::
                When a user-defined ``technologies`` dictionary is provided, the ``init_hybrid_plant`` `function` will still filter based on the list ``techs_in_sim``.

    * ``site``: :class:`hybrid.sites.site_info.SiteInfo` class, provided by step 1.
    * ``dispatch_options``: `dictionary`, Options for modifying dispatch. 
    
        For details see :class:`hybrid.dispatch.hybrid_dispatch_options.HybridDispatchOptions`
    
    * ``cost_info``: `dictionary`, Cost information for PV, Wind, and battery storage only. 
        
        For details see :class:`tools.analysis.bos.cost_calculator.CostCalculator`
        In our example, the ``cost_info`` `dictionary` is provided by ``input_data/system_costs_SAM.json``.

        .. note::
            It is important to understand how the total installed cost is scaled based on the system design. Currently, HOPP scales PV costs using installed DC capacity only.
            The implications of this is HOPP cannot make an economic assessment on the DC-to-AC ratio of the PV system.

    In our example, the ``HybridSimulation`` class is initialized by the following:

    .. code-block::

        hybrid_plant = HybridSimulation(sim_techs,
                                        site,
                                        dispatch_options={
                                            'is_test_start_year': is_test,
                                            'is_test_end_year': is_test,
                                            'solver': 'cbc',
                                            'grid_charging': False,
                                            'pv_charging_only': True
                                            },
                                        cost_info=cost_info['cost_info']
                                        )

3. Change any :class:`hybrid.hybrid_simulation.HybridSimulation` default values after initialization

    This step can be very open-ended because the amount the user changes the defaults of the ``HybridSimulation`` class is dependent on the use case.
    In our example, we have provided a way to modify the following inputs:

    a. Tower and trough CSP system cost assumptions

        .. code-block::

            if hybrid_plant.tower:
                hybrid_plant.tower.ssc.set(cost_info['tower_costs'])
            if hybrid_plant.trough:
                hybrid_plant.trough.ssc.set(cost_info['trough_costs'])

        where ``cost_info`` is a `dictionary` is provided by ``input_data/system_costs_SAM.json``.

    b. CSP dispatch optimization objective cost coefficients

        .. code-block::

            csp_dispatch_obj_costs = {'cost_per_field_generation': 0.5,
                                    'cost_per_field_start_rel': 0.0,
                                    'cost_per_cycle_generation': 2.0,
                                    'cost_per_cycle_start_rel': 0.0,
                                    'cost_per_change_thermal_input': 0.5}

            if hybrid_plant.tower:
                hybrid_plant.tower.dispatch.objective_cost_terms = csp_dispatch_obj_costs
            if hybrid_plant.trough:
                hybrid_plant.trough.dispatch.objective_cost_terms = csp_dispatch_obj_costs     
    
    c. O&M costs assumptions for each technology

        .. code-block::

            for tech in ['tower', 'trough', 'pv', 'battery']:
                if not tech in techs_in_sim:
                    cost_info["SystemCosts"].pop(tech)

            hybrid_plant.assign(cost_info["SystemCosts"])

        where ``cost_info`` is a `dictionary` is provided by ``input_data/system_costs_SAM.json``.

    d. Financial parameters of the SAM singleowner model

        .. code-block::

            with open('input_data/financial_parameters_SAM.json') as f:
                fin_info = json.load(f)

            hybrid_plant.assign(fin_info["FinancialParameters"])
            hybrid_plant.assign(fin_info["TaxCreditIncentives"])
            hybrid_plant.assign(fin_info["Revenue"])
            hybrid_plant.assign(fin_info["Depreciation"])
            hybrid_plant.assign(fin_info["PaymentIncentives"])

    f. Technology specific parameters 

        .. code-block::

            if hybrid_plant.pv:
                hybrid_plant.pv.dc_degradation = [0.5] * 25 # assuming 0.5% degradation each year
                hybrid_plant.pv.value('array_type', 2)      # 1-axis tracking
                hybrid_plant.pv.value('tilt', 0)            # Tilt for 1-axis

        .. note::
            In this example, we use an ``if`` statement before trying to access a specific technology attribute within the ``HybridSimulation`` class, i.e., ``hybrid_plant``.
            The reason for this is to skip those lines of code when a particular technology does not exist within the simulation, otherwise the code would raise 
            an ``Exception`` when accessing a non-existent technology

4. Set ``ppa_price`` and return the ``HybridSimulation`` class

    If the ``grid_resource_file`` of ``SiteInfo`` contains normalized prices, then the user must set the ``ppa_price`` via:

    .. code-block::

        hybrid_plant.ppa_price = (0.10,)  # $/kWh
    
    A alternative options is to provide a ``grid_resource_file`` of ``SiteInfo`` that contains absolute prices in $/MWh. 
    HOPP assumes a default ``ppa_price`` of 0.001 which converts $/MWh to $/kWh.

The last requirement for the hybrid simulation initialize function is to ``return`` the ``HybridSimulation`` class. 
This is the expected output used by other upstream processes for design sampling and optimization.

In our example, this can be done simple by ending the function with:

.. code-block::

    return hybrid_plant

Simulating a single plant design
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the most basic computation a user can do within this example case study framework. 
``1_single_plant.py`` provides an example of how to simulate various cases containing different combinations of technologies which will be described in detail here.
Within this script, we have a `dictionary` that contains different case names as `keys` and a `list` of technologies as `item` for each case.

.. code-block::

    # Cases to run with technologies to include
    cases = {
            'pv_batt': ['pv', 'battery'],
            'tower': ['tower'],
            'tower_pv': ['tower', 'pv'],
            # 'tower_pv_batt': ['tower', 'pv', 'battery'],
            # 'trough': ['trough'],
            # 'trough_pv_batt': ['trough', 'pv', 'battery']
            }

The users can comment, delete, and/or add cases as they see fit for their use case.

Next, this script loops through the different ``cases``, initializes the hybrid simulation, simulates the system, saves the output to a file, 
and prints output to the console. The initialization of the hybrid simulation is done using the function developed in the previous section, i.e., ``init_hybrid_plant``,
using the default technology configuration values, e.g. PV system DC capacity set to 120 MW.

If the user wants to change these defaults, they can do so one of two ways.

    1. Change them in the ``init_hybrid_plant`` `function` directly. This can be done by modify the numbers in lines 124 to 143 of ``simulation_init.py``
    2. Passing in a user-defined technologies `nested dictionary` through ``init_hybrid_plant`` parameter ``ud_techs``. The user-defined technologies `nested dictionary` must following the same format as the one presented in ``init_hybrid_plant``

        .. code-block::

            technologies = {'tower': {
                                'cycle_capacity_kw': 200 * 1000,
                                'solar_multiple': 2.5,
                                'tes_hours': 10.0,
                                'optimize_field_before_sim': not is_test,
                                'scale_input_params': True,
                                },
                            'trough': {
                                'cycle_capacity_kw': 200 * 1000,
                                'solar_multiple': 2.5,
                                'tes_hours': 10.0
                            },
                            'pv': {
                                'system_capacity_kw': 120 * 1000
                                },
                            'battery': {
                                'system_capacity_kwh': 200 * 1000,
                                'system_capacity_kw': 100 * 1000
                                },
                            'grid': grid_interconnect_mw * 1000}

If the user is testing the ``init_hybrid_plant`` `function` for syntax errors, they can set the optional parameter ``is_test`` to ``True`` to enable the dispatch to 
only run the first and last 5 days of the simulation.

.. note::
    The results provided by ``HybridSimulation`` when ``is_test`` is set to ``True`` are invalid and **should not be used** for any analysis!

Once the ``HybridSimulation`` class is initialized using the ``init_hybrid_plant`` `function`, the user can simulate the design by calling the ``simulate`` method.
This will conduct the performance and financial simulation of the specific technologies within the ``case``. 

.. note::
    The user can further modify the ``HybridSimulation`` class by adding script between the ``init_hybrid_plant`` `function` call and the ``simulate`` call.
    
After the simulation is complete, this example uses ``hybrid_simulation_outputs`` to save the outputs to a `csv` file with the case name appended to the beginning of the filename.
Additionally, this example uses the printing helper functions ``print_hybrid_output`` and ``print_BCR_table`` to print specific results to the console. 
In this example script, this can be toggled on and off via the ``print_summary_results`` `boolean`.

Setting up a helper class for parametric and optimization analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to enable a `clean` way to run design parametric (sampling) and/or optimization, we can construct a helper class that 
stores the design variables, calculates the problem size, and initializes the :class:`alt_dev.optimization_problem_alt.HybridSizingProblem` class. 
In our example, this is done using the ``DesignProblem`` class that lives within ``simulation_init.py``. 
This is just an example of a helper class and could be modified to assist specific desired analysis.
This section will describe this helper class in greater detail.

The ``DesignProblem`` class follows a similar declaration as the ``init_hybrid_plant`` `function`. The class is initialized using the following parameters:

    * ``techs_in_sim``: `list`, List of technologies to include in the simulation.
    * ``design_variables``: (optional) `nested dictionary`, Containing technologies, variable names, and bounds.

        This optional parameter provides the user a method for overwriting the default variables and bounds specified within the ``__init__`` `function` of the ``DesignProblem``.
        This parameter requires a `nested dictionary` structure similar to the one used in ``ud_techs`` of ``init_hybrid_plant``.
        The user should use the default ``variables`` as a template.

        .. code-block::
            
            csp_vars = {'cycle_capacity_kw': {'bounds': (50 * 1e3, 200 * 1e3)},
                        'solar_multiple': {'bounds': (1.0, 4.0)},
                        'tes_hours': {'bounds': (4, 18)}}

            variables = {'tower': csp_vars,
                         'trough': csp_vars,
                         'pv': {'system_capacity_kw':  {'bounds': (50*1e3,  400*1e3)}},
                         'battery': {'system_capacity_kwh': {'bounds': (50*1e3, 15*200*1e3)},
                                     'system_capacity_kw':  {'bounds': (50*1e3,  200*1e3)}}}

        Where ``csp_vars`` is a helper `dictionary` to set both 'tower' and 'trough' to the same design variables and bounds.
        In this `nested dictionary`, the first level `key` is technology keyword consistent with strings expected within ``techs_in_sim`` `list`.
        The second level `key` is variable name within HOPP. This can any of the special configuration variables required within the ``technologies`` `dictionary` 
        used by the ``HybridSimulation`` class or it can be any *continuous* variable used within the underlining SAM models. 
        Please refer to the SAM GUI *input browser* and/or `SAM's help <https://samrepo.nrelcloud.org/help/index.html>`_ for more information.
        The last level `key` is ``'bounds'`` which contains a `tuple` with the lower and upper bounds as values, respectively.

        Similar to the ``technologies`` `dictionary` within the ``init_hybrid_plant`` `function`, the ``variables`` `dictionary` is filtered based on which technologies within the simulation.

        .. code-block::

            # Set design variables based on technologies in simulation
            self.design_vars = {key: variables[key] for key in self.techs_in_sim}

    * ``is_test``: (optional) `boolean`, if ``True``, runs dispatch for the first and last 5 days of the year and turns off tower and receiver optimization.

    Lastly, the user can customize what information is stored by the ``ProblemDriver`` class by changing the ``out_options`` within the class ``__init__`` `function`. 

    .. code-block::

        self.out_options = {"dispatch_factors": True,       # add dispatch factors to objective output
                            "generation_profile": True,     # add technology generation profile to output
                            "financial_model": False,       # add financial model dictionary to output
                            "shrink_output": False}         # keep only the first year of output
    
    .. note::
        The ``DesignProblem`` could be modified to accept the ``out_options`` to overwrite default behavior.

    .. note::
        The user should take care in setting the ``out_options`` to provide enough information for their analysis while not provide excess data that could cause memory issues when executing a large number of samples.
        In practice, the user can set-up the output options and test the size of the results of a small case to ensure their computing system has adequate memory resource for the expected larger study.   

The ``DesignProblem`` class has two methods that are used by the sampling and optimization example scripts, ``create_problem`` and ``get_problem_dimension``.

    * ``create_problem`` creates a sizing problem based on hybrid simulation callable, design variables, and output options. This function uses ``init_hybrid_plant`` to provide the :class:`alt_dev.optimization_problem_alt.HybridSizingProblem` class a callable that initializes the ``HybridSimulation`` class.
    * ``get_problem_dimension`` calculates the problem dimensionality, i.e., the number of design variables in the problem, and is used to configure sampling and optimization.

.. _sampling:

Setting up a parametric design study
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A parametric design study can be used to investigate and explore the general trends of the design variable space by using a sampling technique. 

The general workflow of this analysis is as follows:

    1. Generate values for the design variables using a sampling technique.
    2. Simulate the performance of the system using the values generated in step 1. 
    3. Analyze the results to uncover general trends, trade-offs, and regions of interest.

The advantages of this approach over only executing an optimization algorithm are as follows:

    1. Enables parallel computing of sample evaluation.
    2. Provides more information about the design space then an local optimization algorithms.
    3. Provides an initial evaluations of the objective function for the optimization algorithms (only some algorithms use this information).

However, the major disadvantage of this approach is that a large number of sampling points can become computationally intractable. 
Additionally, depending on the sampling technique specific samples can provide little information resulting in wasted computational resources.

In our example, ``2_parametric_study.py`` provides an example of how to set-up and execute a parametric design study. 
This script has structures that mimic structures found in the ``1_single_plant.py`` script. We will repeat any descriptions here for completeness.

At the top of the script, we have parameters that can easily be modified by the user. 

    * ``is_test``: `boolean`, If ``True``, the simulation only runs the first and last 5 days of the simulation.

        .. note::
            Results when ``is_test`` is set to ``True`` should **not be used!** This should only be used to test script surrounding the ``HybridSimulation`` `class`.

    * ``run_name``: `string`, Name of sampling run. Used when saving results.
    * ``write_to_csv``: `boolean`, If ``True`` results are saved as both a `pandas` ``DataFrame`` and a `csv` file, else only saved as a ``DataFrame``.
    * ``cases``: `dictionary`, cases to run with specific technologies to include.

        The ``cases`` `dictionary` contains different case names as `keys` and a `list` of technologies as `item` for each case.

        .. code-block::

            # Cases to run with technologies to include
            cases = {
                    'pv_batt': ['pv', 'battery'],
                    'tower': ['tower'],
                    'tower_pv': ['tower', 'pv'],
                    # 'tower_pv_batt': ['tower', 'pv', 'battery'],
                    # 'trough': ['trough'],
                    # 'trough_pv_batt': ['trough', 'pv', 'battery']
                    }

        The users can comment, delete, and/or add cases as they see fit for their use case.

    * ``save_samples``, `boolean`, If ``True`` sample values are saved to a file before execution begins. This can be used to restart the simulation of samples.

        .. note::
            Code to restart sampling simulation does not exist within the examples provided but is possible.

    * ``sampling_method``, `string`, ``'fullfact'``= Full factorial, ``'lhs'`` = Latin hypercube sampling.

        .. note::
            A user could add additional sampling methods to this script.
    
    * ``N_levels``, `int`, Sets the number of levels for all dimensions in full factorial sampling (only required for the full factorial case).
    * ``N_samples``, `int`, Sets the total number of samples in the latin hypercube sampling (only required for Latin hypercube sampling).
    * ``N_smb``, `int`, Number of small batches to break up the sampling runs.
    
        .. note::
            Cases containing the ``'tower'`` model with field and receiver optimization should be limited to less than 100 samples (depending on RAM of system).
            There is a known memory leak within this code that will result in RAM not being freed after each evaluation.
            Left unchecked, this memory leak will result in a hard fault. To mitigate the issue, the sampling code is set up to run small batches of samples 
            where the :class:`alt_dev.optimization_problem_alt.OptimizationDriver` `class` is destroyed and reinitialized. 
            This frees the memory consumed by the ``'tower'`` model and enables continuous evaluation of large sampling studies.

    * ``N_processors``, `int`, Number of processors available for parallelization.

After the parameters definitions, the sampling execution script begins (line 35). The sampling script can `loop` through multiple cases containing various technologies.

For each case (set of technologies), the script does the following:

    1. Creates a ``DesignProblem`` using the helper `class`` created in the previous step.

        .. note::
            This is where you modify the design variables and their bounds by either passing in the optional ``design_variables`` `nested dictionary` or updating ``DesignProblem``.
            The former is preferred over the latter as it can isolate the changes to a specific sampling script. The latter could impact the other analyses using the ``DesignProblem`` helper `class`. 

    2. Creates a :class:`alt_dev.optimization_problem_alt.OptimizationDriver` `class` based on the driver configuration which contains number of processors, the cache directory, and the writing csv option.
    3. Generates normalized sample values based on the ``sample_method`` parameter.

        * ``fullfact``, will result in a full factorial sampling based on the number of levels parameter, ``N_levels``.

            This will result in a uniform mesh of sampling points for all combinations of ``N_levels`` and the number of design variables (number of factors). 
            For example, if there are 3 design variables with 5 levels this results in 125 samples, i.e., 5^3. Where the levels occur at 0, 0.25, 0.5, 0.75, and 1.0.  

            .. note::
                Full factorial sampling provides uniform sampling; however, the number of samples to evaluated becomes intractable as the number of levels and/or factors increase.

        * ``lhs``, will result in a Latin hypercube sampling based on the number of samples parameter, ``N_samples``.

            Currently, the example is set-up to use the sampling criterion that maximize the minimum distance between points with a centered value within the intervals.
        
        Both methods use ``pyDOE`` package to generate samples. See  `pyDOE <https://pythonhosted.org/pyDOE/index.html>`_ for more information. 

    4. Saves sample values to file if ``save_samples`` is ``True``.
    5. Breaks sampling up into batches based on the ``N_smb`` parameter.
    6. Loops through the small batches and executes them in parallel using ``parallel_sample`` of the ``OptimizationDriver`` `class`.
    7. Kills and reconnects ``OptimizationDriver`` once a small batch of samples are complete.

.. _optimization:

Setting up a design optimization study
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A optimization study uses derivative-free 'black-box' optimization algorithms which iterate on variables to minimize or maximize an objective function.
Derivative-free optimization provides no guarantee of global optimal solution and/or convergence. However, this technique has provided 'good' engineering solutions in practice.
Derivative-free optimization algorithms use various techniques for addressing the problem which can have varying weighting of importance on exploration and exploitation of the variable space.
This weighting of importance will impact the quality of solutions and the speed of convergence. 

The general workflow of this analysis is as follows:

    1. Set-up problem and optimization driver.
    2. Create an objective function.
    3. Execute optimization through driver `class`.
    4. Analyze the results.

There are two major disadvantages of this approach:

    1. A majority of derivative-free optimization algorithms execute candidate points serially, thereby limiting throughput when multiple CPU cores are available.
    2. Evaluation points eventually tend to be closely grouped to a local optima and the user does not get enough information to understand general trends and trade-offs.

The first disadvantage can be addressed by using an evolutionary algorithm where a population could take advantage of parallel computing.
The second disadvantage can be addressed by using an initial sampling of the design space which will be presented in the next section.

In our example, ``3_optimize.py`` provides an example of how to set-up and execute a design optimization study. 
This script has structures that mimic structures found in the ``1_single_plant.py`` script. We will repeat any descriptions here for completeness.

At the top of the script, we have parameters that can easily be modified by the user. 

    * ``is_test``: `boolean`, If ``True``, the simulation only runs the first and last 5 days of the simulation.

        .. note::
            Results when ``is_test`` is set to ``True`` should **not be used!** This should only be used to test script surrounding the ``HybridSimulation`` `class`.

    * ``run_name``: `string`, Name of sampling run. Used when saving results.
    * ``write_to_csv``: `boolean`, If ``True`` results are saved as both a `pandas` ``DataFrame`` and a `csv` file, else only saved as a ``DataFrame``.
    * ``cases``: `dictionary`, cases to run with specific technologies to include.

        The ``cases`` `dictionary` contains different case names as `keys` and a `list` of technologies as `item` for each case.

        .. code-block::

            # Cases to run with technologies to include
            cases = {
                    'pv_batt': ['pv', 'battery'],
                    'tower': ['tower'],
                    'tower_pv': ['tower', 'pv'],
                    # 'tower_pv_batt': ['tower', 'pv', 'battery'],
                    # 'trough': ['trough'],
                    # 'trough_pv_batt': ['trough', 'pv', 'battery']
                    }

        The users can comment, delete, and/or add cases as they see fit for their use case.

    * ``N_calls``, `int`, Sets the number of optimization calls.
    * ``N_init_points``, `int`, Sets the number of evaluations of the objective function with initialization points.
    * ``N_processors``, `int`, Number of processors available for parallelization.

After the parameters definitions, the optimization execution script begins (line 39). The optimization script can `loop` through multiple cases containing various technologies.

For each case (set of technologies), the script does the following:

    1. Creates a ``DesignProblem`` using the helper `class` created in the previous step.

        .. note::
            This is where you modify the design variables and their bounds by either passing in the optional ``design_variables`` `nested dictionary` or updating ``DesignProblem``.
            The former is preferred over the latter as it can isolate the changes to a specific script. The latter could impact the other analyses using the ``DesignProblem`` helper `class`. 

    2. Creates a :class:`alt_dev.optimization_problem_alt.OptimizationDriver` `class` based on the driver configuration which contains number of processors, the cache directory, and the writing csv option.
    3. Defines optimization configuration `dictionary` which provides the optimizer with the problem dimensions, number of calls, verbose option, and number of initial points.
        
        In this example, we use the optimizers from `skopt` For more information about the optimizer configuration see `skopt <https://scikit-optimize.github.io/stable/>`_ for more information. 
    
        .. note::
            In this example, the initial points are executed in series and **does not exploit parallelization!** 

    4. Executes optimization algorithm using the driver `class` ``optimize`` `function` specifying the optimizer, optimizer configuration, and the objective function.

        .. note::
            This example uses the ``optimize`` driver `function` which can only handle one optimizer and/or objective function at a time. 
            To run multiple optimizers, use ``parallel_optimize`` which is shown in the next section.

Creating an objective function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To execute a optimization, the user must provide the optimization driver with an objective function. 
To do this, the user must create a function that takes in the result `dictionary` and returns the objective function value.
In our example script, there exists three examples of objective functions near the top of the script, i.e., ``maxBCR``, ``minimize_real_lcoe``, and ``minimize_nom_lcoe``.

Below, we present an example of an objective function that maximizes benefit cost ratio. 

.. code-block::

    def maxBCR(result):
        return -result['Hybrid Benefit cost Ratio (-)']

Note the negative sign in front of the return value. Most optimization algorithms within Python are configured to minimize an objective function.
To convert a maximization problem to an equivalent minimization problem, the user can multiple by the objective function value by -1.
The result `dictionary` `keys` can be found by the using the ``hybrid_simulation_outputs`` `function` of the :class:`hybrid.hybrid_simulation.HybridSimulation` class.

Setting up a design optimization study with parallel initial sampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section will describe the ``4_parallel_sample_optimize.py`` script which combines the two techniques described in the previous two sections, i.e., parametric design sampling and optimization.
Using this script, a user can set up a run that does initial sampling using parallelization to evaluate samples then provide those initial sampled points to an optimizer to further iterate on the design variables.

The general workflow of this analysis is as follows:

    1. Set-up problem and optimization driver.
    2. Generate values for the design variables using a sampling technique.
    3. Simulate the performance of the system using the values generated in step 2. 
    4. Pass initial sampled points to the optimizer.
    5. Execute optimization through driver `class`.
    6. Analyze the results.

This script mimics portions of the ``2_parametric_study.py`` and ``3_optimize.py`` scripts. 
We will only highlight the differences here and point the user to the previous sections for more information.

In terms of script parameters, this script has 3 new parameters that are `booleans` used to enable and disable specific functionality.

    * ``sample_design``, `boolean`, If ``True`` sampling is done, else skip sampling.
    * ``optimize_design``, `boolean`, If ``True`` optimization will execute, else skip optimization.
    * ``output_cache``, `boolean`, If ``True`` driver will reconnect to previous cache and write out the cache as a compressed `pickle` file, else skip reconnect

    .. note::
        All combinations of the above booleans have not tested exhaustively. Use with caution.

After the parameters definitions, the execution script begins (line 53). The script loops through multiple cases containing various technologies.

Lines 54-94 contain the sampling techniques that are covered within the '`Setting up a parametric design study`' `sampling`_ section.

Lines 95-134 contain the optimization technique that is covered with in '`Setting up a design optimization study`' `optimization`_ section

.. note::
    Depending on the number of initial samples an the degrees of freedom of the problem, the optimizers can converge to `'good'` solutions in 15-30 iterations or less. 

The major difference in this script compared to the ``3_optimize.py`` script is initializing the optimizer with initial evaluated points and using multiple optimizers on the same problem.
In this example, the initial points from sampling are passed to the optimizer through the optimizer configuration `dictionary` via the ``x0`` and ``y0`` parameters.

To use multiple optimizers, we use ``parallel_optimize`` from the :class:`alt_dev.optimization_problem_alt.OptimizationDriver` `class`. Where we pass a list of optimizers, optimizer configurations, and objectives.

.. note::
    This script could be modified to execute different objectives in parallel. 
    To do this, the user would need to take care when setting the ``y0`` parameter after sampling so that the optimizer configuration `dictionaries` (``opt_configs``) and the ``objectives`` list is consistent ordering.

This script enables the evaluation of samples in parallel and enables samples to be used to initialize multiple optimizers with various objectives.

.. note::
    Caches containing sampled points can be reloaded and used for optimization various objective functions as long as the underlining ``HybridSimulation`` parameters have not change.

The script under ``output_cache`` condition can be used to '`recover`' a cache that may not been written either because the user killed the script before completion or an unhandled exception raised.
To do this, comment all the cases except for the one being recovered, set ``sample_design`` and ``optimize_design`` to ``False``, and set ``output_cache`` to ``True``.
The resulting recovered cache will most likely be incomplete and will require modification to remove incomplete data. Additionally, the user will have to modify script to rerun missing values.
If recovery occurs during sampling step, the user can load in the samples if saved via ``save_samples`` and reconnect to the existing cache.
The driver will automatically skip samples that have already been evaluated within the cache which is seen as a '`cache hit`'. 

The '`cache hit`' will also work when running multiple optimizers if two optimizers try to evaluate the same point. In theory, this reduces the number of points to evaluate; however, in practice this event rarely occurs.