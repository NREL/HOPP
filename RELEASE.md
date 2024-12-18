# Release Notes

## Version 3.1.1, Dec. 18, 2024

* Enhanced PV plant functionality: added tilting solar panel support, improved system design handling, and refined tilt angle calculations.
* Integrated ProFAST into the custom financial model for LCOE and other financial calculations; updated CI and documentation accordingly.
* Adjusted battery financials to use discharged energy for levelized cost calculations.
* Improved testing and fixed various bugs, including logging errors, financial model instantiation, and configuration handling.
* Removes unnecessary packages from the dependency stack.
* Documentation moved to a jupyter-book style and build process.

## Version 3.1, Oct. 28, 2024
* Added [Cambium](https://www.nrel.gov/analysis/cambium.html) and [GREET](https://www.energy.gov/eere/greet) integration for LCA analysis
* Updated examples throughout
* Removed package dependencies to streamline installation on Windows

## Version 3.0, Oct. 18th, 2024
* Updated to use pyproject for package management
* Adopted FLORIS v4
* Removed hydrogen modeling and GreenHEART into a separate repository
* Removed out-dated examples to be update in a future release
* General clean-up of repository for planned new development

## Version 2.2.0, Apr. 23, 2024
* Added load following heuristic dispatch method for battery
* Fixed a bug with defining a user's email for the API calls from resource databases
* HOPP solver refactored to be more modular

## Version 2.1.0, Nov. 27, 2023
* Solar plant updated with new input parameters
* GitHub templates added for PRs and Issues
* Updated PySAM dependency to v4.2.0
* Re-enabled testing for hydrogen pressure vessel

## Version 2.0.0, Nov. 27, 2023
* Restructuring of core code to facilitate a new, more user-friendly interface and developer experience
* Developed HoppInterface as the single entry-point for users
* Expanded input files to yaml format for all technologies
* Restructured code by technology to make relevant code more easy to find
* Refactored technology classes using attrs library to better define classes
* Improved documentation for the technologies and what fields are available for each
* Cleaned up examples and resource files
* Added Wind + PV + Battery + PEM Electrolyzer analysis to examples in `H2_Analysis`
* Added default scenario inputs for hydrogen as end product with result files
* Added files for simulating, optimizing and plotting Wind + PV + Battery + PEM Electrolyzer hybrids
* Added a PEM electrolyzer model to hybrid (not integrated into HybridSimulation)
* Separate power and financial simulations for PowerSources and HybridSimulation
* Fixed multiprocessing issues
* Updates to integrate FLORIS v3

## Version 0.1.0.dev3, Mar. 11, 2022
* Include CBC package data for Windows

## Version 0.1.0.dev2, Mar. 11, 2022
* Add CBC open-source dispatch solver as default (faster than GLPK)
* Add Xpress and Gurobi AMPL commercial dispatch solvers
* Add clustering of simulation days with some tuning (use with caution)
* Remove taxes from grid dispatch model
* Add weighting of hybrid financials by production or cost ratio
* Update to csp dispatch cost parameters for absolute grid prices
* Add csv file output option to driver
* Add PV charging only and grid charging options for dispatch
* Add key parameter scaling when scaling CSP tower capacities
* Add capacity credit payment calculations based on dispatch
* Reformulate grid dispatch model
* Add desired schedule option to dispatch model to follow schedule at least cost
* Improve class documentation
* Fix general bugs and clean-up code

## Version 0.1.0.dev1, Oct. 11, 2021
* Updated requirements
* Added presolve to GLPK LP solver

## Version 0.1.0.dev, Oct. 10, 2021
* Battery dispatch and simulation with example in examples/simulate_hybrid_wbattery_dispatch.py 
* Separate layout-related functions from optimization code into Layout classes
* Refactor Optimizer classes to use HybridSimulation with examples in examples/optimization/hybrid_npv.py and examples/optimization/hybrid_sizing_problem.py
* Add FLORIS as custom module for Wind simulations with examples in examples/add_custom_module
* Move plotting functions into dedicated files
* Rename "Solar" classes to "PV"
* Add ElectricityPrices class with some example input files in resource_files/grid
* Add storage costs to CostCalculator
* Add concentrating solar power (CSP) tower and trough configurations through pySSC wrapper
* Add dispatch optimization model for CSP models
* Add design evaluation methods to iterate on design variables through space sampling, single objective derivative-free optimization, and multi-objective optimization

## Version 0.0.5, Apr 30, 2021
* Update PySAM requirements
* Fix flicker check for weight_option

## Version 0.0.3, Jan 21, 2021
* Allow flicker heatmap grid cell width and height to be changed
* Normalize time-weighted heatmaps by area as well

## Version 0.0.2, Dec 28, 2020
* Allow using swept area of blades inplace of individual blades for flicker model
* Allow wind direction and solar data inputs to flicker model
* Add flicker heatmap weighted by hours shaded as loss option
* Minor bug fixes and restructuring of flicker functions

## Version 0.0.1, Dec 14, 2020
* Fixed and updated data in Balance-of-station cost model

## Version 0.0.0, Dec 2, 2020
* Official beta release
* Wind & PV simulations
* Hybrid simulation combining mixes of wind & pv
* Flicker modeling
* Derivative-free optimization framework
* Balance-of-station cost model 
