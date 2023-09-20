import glob
import os
from pathlib import Path
from setuptools import setup


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "hopp", "version.py"), encoding="utf-8") as f:
    version = f.read()

version = version.split('=')[-1].strip().strip('"').strip("'")

# copy over packages
directories = ['hopp', "examples"]

pkg_dirs = []


def recursive_directories(dirs):
    for directory in dirs:
        pkg_dirs.append(directory)
        files = glob.glob(directory + '/*')
        for f in files:
            if os.path.isdir(f):
                recursive_directories((f,))


recursive_directories(directories)

# copy over package data

package_data = {
    "hopp": [
        str(Path("hopp") / "hydrogen" / "h2_storage" / "pressure_vessel" / "compressed_gas_storage_model_20221021" / "Tankinator.xlsx"),
        str(Path("hopp") / "hydrogen" / "h2_transport" / "data_tables" / "*.csv"),
        str(Path("hopp") / "tools" / "analysis" / "bos" / "BOSLookup.csv")
    ]
                }

hopp_path = Path("hopp")
flicker_path = hopp_path / "simulation" / "technologies" / "layout" / "flicker_data"

for file in glob.glob(str(flicker_path / "*shadow.txt")):
    package_data["hopp"].append(str(os.path.relpath(file, str(Path("hopp")))))

for file in glob.glob(str(flicker_path / "*flicker.txt")):
    package_data["hopp"].append(str(os.path.relpath(file, str(Path("hopp")))))


pySSC_daotk_path = hopp_path / "simulation" / "technologies" / "pySSC_daotk"

pySSC_data_dirs = ["libs", "tower_data", "trough_data"]
for data_dir in pySSC_data_dirs:
    data_path = pySSC_daotk_path / data_dir
    for file in glob.glob(str(data_path / '*')):
        package_data["hopp"].append(str(os.path.relpath(file, str(Path("hopp")))))

cbc_solver_path = hopp_path / "simulation" / "technologies" / "dispatch" / "cbc_solver" / "cbc-win64"
for file in glob.glob(str(cbc_solver_path / '*')):
    package_data["hopp"].append(str(os.path.relpath(file, str(Path("hopp")))))

setup(name='HOPP',
      version=version,
      url='https://www.https://github.com/NREL/HOPP',
      description='Hybrid Systems Optimization and Performance Platform',
      long_description=open("RELEASE.md").read(),
      long_description_content_type='text/markdown',
      license='BSD 3-Clause',
      author='NREL',
      author_email='dguittet@nrel.gov',
      python_requires='>=3.8',
      packages=pkg_dirs,
      package_data=package_data,
      include_package_data=True,
      install_requires=open("requirements.txt").readlines(),
      tests_require=['pytest']
      )
