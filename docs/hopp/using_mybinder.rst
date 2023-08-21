.. _Using_MyBinder:


Using Binder to Access Jupyter Notebooks
=========================================
Binder (https://mybinder.org/) works by building a Docker image of a code repository that contains Jupyter notebooks. This allows multiple users to quickly access code and software tools like HOPP independent of computing environment of software development skill-level.

1. Go to https://mybinder.org/

2. Specify GitHub repository or URL: (eg., https://github.com/NREL/HOPP)

3. Set a Git ref (branch, tag, or commit): (eg., feature/osw_h2)

4. Provide a notebook file (optional): (eg., osw_h2_analysis.ipynb)

5. Press ``Launch`` and refer to the ``Build log`` to view progress.

These steps result in the repository content being hosted by a JupyterLab server. The user should be able to step through the notebook as if the entire repository and computing environment were installed on their own machine.

.. note::

   When attempting to use Binder for the first time, the build process may take several minutes. 
   It is best to wait and do not close the browser window until it is finished. Future launches 
   will be faster as the build environment is saved as a temporary internet file. 
