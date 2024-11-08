# Installation

HOPP currently supports the following versions of Python:

[![image](https://img.shields.io/pypi/pyversions/hopp.svg)](https://pypi.python.org/pypi/hopp)

HOPP is available on PyPI
[![PyPI version](https://badge.fury.io/py/hopp.svg)](https://badge.fury.io/py/hopp), and can be
downloaded using:

```bash
pip install HOPP
```

```{contents}
:depth: 2
```

## Setting Up a Conda Virtual Environment

The HOPP development team recommends using [Miniconda](https://docs.anaconda.com/miniconda/) to
create a virtual Python environment for HOPP analysis. Please follow the
[instructions on the Miniconda page](https://docs.anaconda.com/miniconda/) to learn more about
installing and working with Conda.

To create a new HOPP environment, simply run the following command in your terminal, Anaconda
Prompt, PowerShell, or etc. You may replace "hopp" with your preferred environment name.

```bash
conda create --name hopp python=3.11 -y
```

To activate this environment:

```bash
conda activate hopp
```

To deactivate this environment:

```bash
conda deactivate
```

HOPP requires the `conda-forge` specific packages `glpk` and `coin-or-cbc`.

```bash
conda install -y -c conda-forge glpk coin-or-cbc
```

```{note}
The `conda-forge` package `coin-or-cbc` is only supported on Linux and MacOS systems. HOPP is
distributed with a `cbc` executable for Windows OS.
```

## Installing HOPP package via PIP

In your Conda environment, you can install HOPP by executing:

```bash
pip install HOPP
```

or a specific version using:

```bash
pip install HOPP==<version_number>
```

where `<version_number>` corresponds to the specific version number you wish to install. Current
development and release versions can be found [here](https://pypi.org/project/HOPP/#history).

## Installing from Source

To develop within HOPP, please clone the `HOPP <https://github.com/NREL/HOPP>`_ repository using
Git in a local target directory.

```bash
git clone https://github.com/NREL/HOPP.git
```

```bash
cd HOPP/
pip install .
```

Additionally, the `-e` flag can be passed if you would like to modify the code and not have to worry
about rerunning the installation step each time a change is made.

Additional flags, which may be passed as `pip install ".[flag-name]"`:

- `develop`: Installs the developer tools for documentation and testing
- `examples`: Installs Jupyter Lab so you may run the examples
- `all`: All of the above

## Developer installations

Then install dependencies and an editable version of the HOPP package in your virtual environment.
Here, `"[all]"` is indicating that the examples requirements and the developer tools will be
installed. Any changes made to the repository for contributions to HOPP should have passing tests,
successful documentation builds, and fully working examples, and the developer tools ensure these
can all be checked.

```bash
cd HOPP/
pip install -e ".[all]"
```

(using-vscode)=
## Using Visual Studio Code for HOPP

For source development, most NREL developers use
[Visual Studio Code](https://code.visualstudio.com/) as their IDE (integrated development
environment).

1. Install `Visual Studio Code <https://code.visualstudio.com/>`_.
2. Open VS Code and click 'Open Folder...'
3. Navigate to where either where you want to create scripts using HOPP (analysis) or the HOPP repository (source development) and 'Select Folder'
4. Install `Python` extension for linting and debugging
5. Set up your Python interpreter by:

    a. Pressing Ctrl + Shift + P or Going to View -> Command Palette… (the Command Palette should appear at the top of the screen)
    b. Type 'Interpreter', select 'Python: Select Interpreter'
    c. Choose the Conda environment created for HOPP

```{note}
VS Code has many tutorials online for setting up Python projects. 
Here is a video about
[setting up Visual Studio Code for Python Beginners](https://www.youtube.com/watch?v=7FltByLPnrg&ab_channel=VisualStudioCode).
```

## Common installation issues

### dlib fails to install

Error messages like

```bash
note: This error originates from a subprocess, and is likely not a problem with pip.
error: legacy-install-failure

x Encountered error while trying to install package.
-> dlib

note: This is an issue with the package mentioned above, not pip.
hint: See above for output from the failure.
```

with something about C++ above in the error message

```bash
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


You must use Visual Studio to build a python extension on windows.  If you
are getting this error it means you have not installed Visual C++.  Note
that there are many flavors of Visual Studio, like Visual Studio for C#
development.  You need to install Visual Studio for C++.


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

**Solution:**

1. Upgrade `pip`, `wheel`, and `setuptools`

 ```bash
 conda upgrade pip
 conda upgrade wheel
 conda upgrade setuptools
 ```

1. Download C++ build tool through [Visual Studio](https://visualstudio.microsoft.com/vs/features/cplusplus/)
  When going through the installation, be sure to select 'Desktop development with C++' under
  'Workloads'. Once complete this may require a system restart.

### Module not found error

Error message like the following:

```python
ModuleNotFoundError: No module named 'PACKAGE' 
```

Where ``'PACKAGE'`` can be any number of Python packages, e.g., ``pandas``

**Solution:**

1. Check your Python interpreter in VS Code see [Step 5](using-vscode))
2. Check you have installed the `Python` extension for linting and debugging
