# Hybrid Optimization and Performance Platform

As part of NREL's [Hybrid Energy Systems Research](https://www.nrel.gov/wind/hybrid-energy-systems-research.html), this
software assesses optimal designs for the deployment of utility-scale hybrid energy plants, particularly considering wind,
solar and storage.

## Software requirements
- Python version 3.5+ 64-bit

## Setup
1. Using Git, navigate to a local target directory and clone repository:
    ```
    git clone https://github.com/NREL/HOPP.git
    ```

2. Open a terminal and navigate to /HOPP

3. Create a new virtual environment and change to it. Using Conda and naming it 'hopp':
    ```
    conda create --name hopp python=3.8 -y
    conda activate hopp
    ```

4. Install requirements:
    ```
    conda install -c conda-forge glpk -y
    conda install -c conda-forge shapely==1.7.1 -y
    pip install -r requirements.txt
    ```

5. Run install script:
    ```
    python setup.py install
    ```

6. The functions which download resource data require an NREL API key. Obtain a key from:
    
    [https://developer.nrel.gov/signup/](https://developer.nrel.gov/signup/)
    
7. Create the file `.env` in /HOPP and add one line to it, where \<key\> is the NREL API key you obtained:
    ```
    NREL_API_KEY=<key>
    ```

8. Verify setup by running an example:
    ```
    python examples/simulate_hybrid.py
    ```

## Using as a Standalone Package
HOPP is available as a PyPi package:

`pip install HOPP`

or as a conda package:

`conda install hopp -c nrel -c conda-forge -c sunpower`

NOTE: If you install from conda you will need to install `global-land-mask`
from PyPi:

`pip install global-land-mask`

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


