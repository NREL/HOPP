============
Installation
============

HOPP currently supports the following versions of Python:

* CPython: 3.7

Contents:

.. contents::
   :local:
   :depth: 2

Using CONDA for Virtual Environment
-----------------------------------
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

Installing HOPP package via PIP
-------------------------------

In your *conda* environment, you can install HOPP by executing

.. code-block::

    pip install HOPP

or

.. code-block::

    pip install HOPP==<version_number>

where ``version_number`` corresponds to the specific version number you wish to install. Current development and release
versions can be found `here <https://pypi.org/project/HOPP/#history>`_.

Installing from Source
----------------------

To develop within HOPP, please clone the `HOPP <https://github.com/NREL/HOPP/tree/master>`_ repository using Git in a local target directory.

.. code-block::

    git clone https://github.com/NREL/HOPP.git

.. _vscode:

Using Visual Studio Code for HOPP
---------------------------------

For source development, most NREL developers use `Visual Studio Code <https://code.visualstudio.com/>`_ as the IDE (integrated development environment).

1. Install `Visual Studio Code <https://code.visualstudio.com/>`_.
2. Open VS Code and click 'Open Folder...'
3. Navigate to where either where you want to create scripts using HOPP (analysis) or the HOPP repository (source development) and 'Select Folder'
4. Install `Python` extension for linting and debugging
5. Set-up your Python interpreter by:

    a. Pressing Ctrl + Shift + P or Going to View -> Command Palette… (the Command Palette should appear at the top of the screen)
    b. Type 'Interpreter', select 'Python: Select Interpreter'
    c. Choose the Conda environment created for HOPP

.. note::
    VS Code has many tutorials online for setting up Python projects. 
    Here is an video about `setting up Visual Studio Code for Python Beginners <https://www.youtube.com/watch?v=7FltByLPnrg&ab_channel=VisualStudioCode>`_.

Common installation issues
--------------------------

dlib fails to install
^^^^^^^^^^^^^^^^^^^^^

Error messages like

.. code-block::

      note: This error originates from a subprocess, and is likely not a problem with pip.
    error: legacy-install-failure

    x Encountered error while trying to install package.
    -> dlib

    note: This is an issue with the package mentioned above, not pip.
    hint: See above for output from the failure.

with something about C++ above in the error message

.. code-block::

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    You must use Visual Studio to build a python extension on windows.  If you
    are getting this error it means you have not installed Visual C++.  Note
    that there are many flavors of Visual Studio, like Visual Studio for C#
    development.  You need to install Visual Studio for C++.


    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


**Solution:**

    1. Upgrade ``pip``, ``wheel``, and ``setuptools``

        .. code-block::
            
            conda upgrade pip
            conda upgrade wheel
            conda upgrade setuptools

    2. Download C++ build tool through `Visual Studio <https://visualstudio.microsoft.com/vs/features/cplusplus/>`_

        When going through the installation, be sure to select 'Desktop development with C++' under 'Workloads'. 
        Once complete this may require a system restart.

Module not found error 
^^^^^^^^^^^^^^^^^^^^^^

Error message like 

.. code-block::

    ModuleNotFoundError: No module named 'PACKAGE' 

Where ``'PACKAGE'`` can be any number of Python packages, e.g., ``pandas``

**Solution:**

    1. Check your Python interpreter in VS Code see :ref:`Step 5 <vscode>` under `'Using Visual Studio Code for HOPP'`
    2. Check you have installed the `Python` extension for linting and debugging

