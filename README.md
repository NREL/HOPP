# HOPP: Hybrid Optimization and Performance Platform

[![PyPI version](https://badge.fury.io/py/hopp.svg)](https://badge.fury.io/py/hopp)
[![CI Tests](https://github.com/NREL/HOPP/actions/workflows/ci.yml/badge.svg)](https://github.com/NREL/HOPP/actions/workflows/ci.yml)
[![image](https://img.shields.io/pypi/pyversions/hopp.svg)](https://pypi.python.org/pypi/hopp)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

As part of NREL's [Hybrid Energy Systems Research](https://www.nrel.gov/wind/hybrid-energy-systems-research.html), this
software assesses optimal designs for the deployment of distributed, commercial, and utility-scale hybrid energy plants, particularly considering wind,
solar and storage.


## Part of the WETO Stack

HOPP is primarily developed with the support of the U.S. Department of Energy and is part of the [WETO Software Stack](https://nrel.github.io/WETOStack). For more information and other integrated modeling software, see:
- [Portfolio Overview](https://nrel.github.io/WETOStack/portfolio_analysis/overview.html)
- [Entry Guide](https://nrel.github.io/WETOStack/_static/entry_guide/index.html)
- [Techno-Economic Modeling Workshop](https://nrel.github.io/WETOStack/workshops/user_workshops_2024.html#tea-and-cost-modeling)
- [Systems Engineering Workshop](https://nrel.github.io/WETOStack/workshops/user_workshops_2024.html#systems-engineering)


## Software requirements

- Python version 3.11 or higher

## Installing from Package Repositories

1. HOPP is available as a PyPi package:

    ```bash
    pip install HOPP
    ```

## Installing from Source

1. Using Git, navigate to a local target directory and clone repository:

    ```bash
    git clone https://github.com/NREL/HOPP.git
    ```

2. Navigate to `HOPP`

    ```bash
    cd HOPP
    ```

3. Create a new virtual environment and change to it. Using Conda and naming it 'hopp':

    ```bash
    conda create --name hopp python=3.11 -y
    conda activate hopp
    ```

4. Install HOPP and its dependencies:

    ```bash
    conda install -y -c conda-forge coin-or-cbc glpk
    ```

    Note if you are on Windows, you will have to manually install Cbc: https://github.com/coin-or/Cbc.

    - If you want to just use HOPP:

       ```bash
       pip install .  
       ```

    - If you want to work with the examples:

       ```bash
       pip install ".[examples]"
       ```

    - If you also want development dependencies for running tests and building docs. Note the `-e` flag which installs HOPP in-place so you can edit the HOPP package files:  

       ```bash
       pip install -e ".[develop]"
       ```

5. The functions which download resource data require an NREL API key. Obtain a key from:

    [https://developer.nrel.gov/signup/](https://developer.nrel.gov/signup/)

6. To set up the `NREL_API_KEY` and `NREL_API_EMAIL` required for resource downloads, you can create Environment Variables called `NREL_API_KEY` and `NREL_API_EMAIL`. Otherwise, you can keep the key in a new file called ".env" in the root directory of this project. 

    Create a file ".env" that contains the single line:

    ```bash
    NREL_API_KEY=key
    NREL_API_EMAIL=your.name@email.com
    ```

7. Verify setup by running tests:

    ```bash
    pytest tests/hopp
    ```

8. To set up `NREL_API_KEY` for resource downloads, first refer to section 6 and 7 above. But for the `.env` file method,
   the file should go in the working directory of your Python project, e.g. directory from where you run `python`.

## Getting Started

The [Examples](./examples/) contain Jupyter notebooks and sample YAML files for common usage scenarios in HOPP. These are actively maintained and updated to demonstrate HOPP's capabilities. For full details on simulation options and other features, see the [documentation](https://hopp.readthedocs.io/en/latest/).

## Contributing

Interested in improving HOPP? Please see the [Contributor's Guide](docs/CONTRIBUTING.md)
for more information.
