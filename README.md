# Hybrid Optimization and Performance Platform

![CI Tests](https://github.com/NREL/HOPP/actions/workflows/ci.yml/badge.svg)

As part of NREL's [Hybrid Energy Systems Research](https://www.nrel.gov/wind/hybrid-energy-systems-research.html), this
software assesses optimal designs for the deployment of utility-scale hybrid energy plants, particularly considering wind,
solar and storage.

## Software requirements
- Python version 3.5+ 64-bit

## Installing from Source
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
    conda install -c conda-forge coin-or-cbc -y
    conda install -c conda-forge shapely==1.7.1 -y
    pip install -r requirements.txt
    ```
    
    Note if you are on Windows, you will have to manually install Cbc: https://github.com/coin-or/Cbc

5. Run install script:
    ```
    python setup.py develop
    ```

6. The functions which download resource data require an NREL API key. Obtain a key from:
    
    [https://developer.nrel.gov/signup/](https://developer.nrel.gov/signup/)
    

7. To set up the `NREL_API_KEY` required for resource downloads, you can create an Environment Variable called 
   `NREL_API_KEY`. Otherwise, you can keep the key in a new file called ".env" in the root directory of this project. 

    Create a file ".env" that contains the single line:
     ```
    NREL_API_KEY=key
    ```

8. Verify setup by running an example:
    ```
    python examples/simulate_hybrid.py
    ```

## Installing from Package Repositories
1. HOPP is available as a PyPi package:

    ```
    pip install HOPP
    ```

    or as a conda package:

    ```
    conda install hopp -c nrel -c conda-forge -c sunpower
    ```

    NOTE: If you install from conda you will need to install `global-land-mask` from PyPi:

    ```
    pip install global-land-mask
    ```

2. To set up `NREL_API_KEY` for resource downloads, first refer to section 7 and 8 above. But for the `.env` file method,
   the file should go in the working directory of your Python project, e.g. directory from where you run `python`.

## Examples

The examples can be run by installing HOPP, then cloning the repo and calling each example file.

##### Basic Simulation
`python examples/simulate_hybrid.py`

##### Flicker Map
`python examples/flicker.py`

##### Single Location Analysis
`python examples/analysis/single_location.py`

##### Wind Layout Optimization
`python examples/optimization/layout_opt/wind_run.py`

##### Hybrid Layout Optimization
`python examples/optimization/layout_opt/hybrid_run.py`

## HOPP-demos

The https://github.com/dguittet/HOPP-demos repo contains a more full featured example with detailed technical and financial inputs, a few scenarios and the optimal PV, Wind, and Battery design results.

