from pathlib import Path
from setuptools import setup, find_packages
import re


version = {}

with open("hopp/version.py") as fp:
    exec(fp.read(), version)

# Get package data
base_path = Path("hopp")
package_data_files = [
    base_path / "hydrogen" / "h2_storage" / "pressure_vessel" / "compressed_gas_storage_model_20221021" / "Tankinator.xlsx",
    *base_path.glob("hydrogen/h2_transport/data_tables/*.csv"),
    *base_path.glob("tools/analysis/bos/BOSLookup.csv"),
    *base_path.glob("simulation/technologies/layout/flicker_data/*shadow.txt"),
    *base_path.glob("simulation/technologies/layout/flicker_data/*flicker.txt"),
    *base_path.glob("simulation/technologies/csp/pySSC_daotk/libs/*"),
    *base_path.glob("simulation/technologies/csp/pySSC_daotk/tower_data/*"),
    *base_path.glob("simulation/technologies/csp/pySSC_daotk/trough_data/*"),
    *base_path.glob("simulation/technologies/dispatch/cbc_solver/cbc-win64/*")
]

package_data = {
    "hopp": [str(file.relative_to(base_path)) for file in package_data_files]
}

setup(
    name='HOPP',
    version=version['__version__'],
    url='https://github.com/NREL/HOPP',
    description='Hybrid Systems Optimization and Performance Platform',
    long_description=(base_path.parent / "RELEASE.md").read_text(),
    long_description_content_type='text/markdown',
    license='BSD 3-Clause',
    author='NREL',
    author_email='dguittet@nrel.gov',
    python_requires='>=3.8',
    packages=find_packages(),
    package_data=package_data,
    include_package_data=True,
    install_requires=(base_path.parent / "requirements.txt").read_text().splitlines(),
    tests_require=['pytest', 'pytest-subtests', 'responses']
)
