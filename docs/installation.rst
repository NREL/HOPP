Installation
------------

HOPP currently supports the following versions of Python:

* CPython: 3.7

Using CONDA for Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The HOPP development team recommends using *conda* to create a virtual environment for HOPP analysis. *conda* is
included with the Anaconda distribution of Python. A conda environment can be set up using the *create* and *active* commands.

.. code-block::

    conda create --name <your_env_name> python=3.7 -y
    conda activate <your_env_name>

Where ``your_env_name`` corresponds to the name you wish to call your *conda* environment.

HOPP requires the *conda-forge* specific packages *shapely* and *glpk* or *coincbc*.

.. code-block::

    conda install -c conda-forge shapely==1.7.1 -y
    conda install -c conda-forge glpk -y
    conda install -c conda-forge coincbc -y

.. note::

    The *conda-forge* package *coincbc* is only supported on Linux and MacOS systems. HOPP is distributed with a *cbc*
    executable for Windows OS.

PIP Installing HOPP
~~~~~~~~~~~~~~~~~~~

In your *conda* environment, you can install HOPP by executing

.. code-block::

    pip install HOPP

or

.. code-block::

    pip install HOPP==<version_number>

where ``version_number`` corresponds to the specific version number you wish to install. Current development and release
versions can be found `here <https://pypi.org/project/HOPP/#history>`_.

