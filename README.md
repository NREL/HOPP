# Packages

## GreenHEART: Green Hydrogen Energy and Renewable Technologies

[![PyPI version](https://badge.fury.io/py/greeheart.svg)](https://badge.fury.io/py/hopp)
![CI Tests](https://github.com/NREL/GreenHEART/actions/workflows/ci.yml/badge.svg)
[![image](https://img.shields.io/pypi/pyversions/greeheart.svg)](https://pypi.python.org/pypi/greeheart)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Hybrid project power-to-x component-level system performance and financial modeling for control and
design optimization. Currently includes renewable energy, hydrogen, ammonia, and steel. Other
elements such as desalination systems, pipelines, compressors, and storage systems can also be
included as needed.

`greenheart` will install alongside `hopp` by following the instructions for installing HOPP from
source.

## Software requirements

- Python version 3.8, 3.9, 3.10 64-bit
- Other versions may still work, but have not been extensively tested at this time

## Installing from Package Repositories

1. GreenHEART is available as a PyPi package:

    ```bash
    pip install greenheart
    ```

## Installing from Source

1. Using Git, navigate to a local target directory and clone repository:

    ```bash
    git clone https://github.com/NREL/GreenHEART.git
    ```

2. Navigate to `GreenHEART`

    ```bash
    cd GreenHEART
    ```

3. Create a new virtual environment and change to it. Using Conda and naming it 'greenheart':

    ```bash
    conda create --name greenheart python=3.9 -y
    conda activate greenheart
    ```

4. Install GreenHEART and its dependencies:

    ```bash
    conda install -y -c conda-forge coin-or-cbc=2.10.8 glpk
    ```

    Note if you are on Windows, you will have to manually install Cbc: https://github.com/coin-or/Cbc.

    - If you want to just use GreenHEART:

       ```bash
       pip install .  
       ```

    - If you want to work with the examples:

       ```bash
       pip install ".[examples]"
       ```

    - If you also want development dependencies for running tests and building docs:  

       ```bash
       pip install -e ".[develop]"
       ```

    - In one step, all dependencies can be installed as:

      ```bash
      pip install -e ".[all]
      ```

5. The functions which download resource data require an NREL API key. Obtain a key from:

    [https://developer.nrel.gov/signup/](https://developer.nrel.gov/signup/)

6. To set up the `NREL_API_KEY` and `NREL_API_EMAIL` required for resource downloads, you can create
   Environment Variables called `NREL_API_KEY` and `NREL_API_EMAIL`. Otherwise, you can keep the key
   in a new file called ".env" in the root directory of this project.

    Create a file ".env" that contains the single line:

    ```bash
    NREL_API_KEY=key
    NREL_API_EMAIL=your.name@email.com
    ```

7. Verify setup by running tests:

    ```bash
    pytest
    ```


2. To set up `NREL_API_KEY` for resource downloads, first refer to section 7 and 8 above. But for
   the `.env` file method, the file should go in the working directory of your Python project, e.g.
   directory from where you run `python`.

## Parallel Processing for GreenHEART finite differences and design of experiments

GreenHEART is set up to run in parallel using MPI and PETSc for finite differencing and for design of
experiments runs through OpenMDAO. To use this capability you will need to follow the addtional installation
instruction below:

```bash
conda install -c conda-forge mpi4py petsc4py
```

For more details on implementation and installation, reference the documentation for OpenMDAO.

To to check that your installation is working, do the following:

```bash
cd tests/greenheart/
mpirun -n 2 pytest test_openmdao_mpi.py
```

## Getting Started

The [Examples](./examples/) contain Jupyter notebooks and sample YAML files for common usage
scenarios in GreenHEART. These are actively maintained and updated to demonstrate GreenHEART's
capabilities. For full details on simulation options and other features, documentation is
forthcoming.

## Contributing

Interested in improving GreenHEART? Please see the [Contributing](./CONTRIBUTING.md) section for more information.
