# Release Notes

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
* Add Floris as custom module for Wind simulations with examples in examples/add_custom_module
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