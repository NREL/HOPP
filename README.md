# Hybrid Optimization and Performance Platform

![CI Tests](https://github.com/NREL/HOPP/actions/workflows/ci.yml/badge.svg)

As part of NREL's [Hybrid Energy Systems Research](https://www.nrel.gov/wind/hybrid-energy-systems-research.html), this
software assesses optimal designs for the deployment of utility-scale hybrid energy plants, particularly considering wind,
solar and storage.

## Software requirements
- Python version 3.8, 3.9, 3.10 64-bit
- Other versions may still work, but have not been extensively tested at this time

## Installing from Package Repositories
1. HOPP is available as a PyPi package:

    ```
    pip install HOPP
    ```

## Installing from Source
1. Using Git, navigate to a local target directory and clone repository:
    ```
    git clone https://github.com/NREL/HOPP.git
    ```

2. Navigate to `HOPP`
    ```
    cd HOPP
    ```

3. Create a new virtual environment and change to it. Using Conda and naming it 'hopp':
    ```
    conda create --name hopp python=3.8 -y
    conda activate hopp
    ```

4. Install dependencies:
    ```
    conda install -c conda-forge coin-or-cbc=2.10.8 -y
    conda install -c conda-forge glpk -y
    pip install -r requirements.txt
    ```
    
    Note if you are on Windows, you will have to manually install Cbc: https://github.com/coin-or/Cbc

    If you also want development dependencies for running tests and building docs:

    ```
    pip install -r requirements-dev.txt
    ```

5. Install HOPP:
    ```
    pip install -e .
    ```

6. The functions which download resource data require an NREL API key. Obtain a key from:
    
    [https://developer.nrel.gov/signup/](https://developer.nrel.gov/signup/)
    

7. To set up the `NREL_API_KEY` and `NREL_API_EMAIL` required for resource downloads, you can create Environment Variables called `NREL_API_KEY` and `NREL_API_EMAIL`. Otherwise, you can keep the key in a new file called ".env" in the root directory of this project. 

    Create a file ".env" that contains the single line:
    ```
    NREL_API_KEY=key
    NREL_API_EMAIL=your.name@email.com
    ```

8. Verify setup by running tests:
    ```
    pytest
    ```


2. To set up `NREL_API_KEY` for resource downloads, first refer to section 7 and 8 above. But for the `.env` file method,
   the file should go in the working directory of your Python project, e.g. directory from where you run `python`.

## Parallel Processing for GreenHEART finite differences and design of experiments
GreenHEART is set up to run in parallel using MPI and PETSc for finite differencing and for design of experiments runs through OpenMDAO. To use this capability you will need to follow the addtional installation instruction below:
```
    conda install -c conda-forge mpi4py petsc4py
```
For more details on implementation and installation, reference the documentation for OpenMDAO.

To to check that your installation is working, do the following:
```
    cd tests/greenheart/
    mpirun -n 2 pytest test_openmdao_mpi.py
```

## Getting Started

The [Examples](./examples/) contain Jupyter notebooks and sample YAML files for common usage scenarios in HOPP. These are actively maintained and updated to demonstrate HOPP's capabilities. For full details on simulation options and other features, see the [documentation](https://hopp.readthedocs.io/en/latest/).

## Contributing

Interested in improving HOPP? Please see the [Contributing](./CONTRIBUTING.md) section for more information.
