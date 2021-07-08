# Hybrid Optimization and Performance Platform

As part of NREL's [Hybrid Energy Systems Research](https://www.nrel.gov/wind/hybrid-energy-systems-research.html), this
software assesses optimal designs for the deployment of utility-scale hybrid energy plants, particularly considering wind,
solar and storage.

## Software requirements
- Python version 3.5+ 64-bit

## Install
HOPP is available as a PyPi package:

`pip install HOPP`

or as a conda package:

`conda install hopp -c nrel -c conda-forge -c sunpower`

NOTE: If you install from conda you will need to install `global-land-mask`
from PyPi:

`pip install global-land-mask`

## Setting up environment ".env" file
The functions which download resource data require an NREL API key.
These keys can be obtained at https://developer.nrel.gov/signup/
There is an included ".env-example" file which contains a blank "NREL_API_KEY="
Copy .env-example to a new file, ".env" and edit the .env file using your preferred text editor to add your NREL_API_KEY

## Examples

The examples can be run by installing HOPP, then cloning the repo and calling each example file.

##### Basic Simulation
`python examples/simulate_hybrid.py`

##### Flicker Map
`python examples/flicker.py`

##### Single Location Analysis
`python examples/analysis/single_location.py`

##### Wind Layout Optimization
`python examples/optimization/wind_opt/run.py`

##### Hybrid Layout Optimization
`python examples/optimization/hybrid_opt/run.py`


