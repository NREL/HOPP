# hybrid_systems

## Software requirements
- python 3.7 proposed as the high level wrapper (due to deprecation of python2.x in January 2020).  The code should work on any Python 3 version higher than 3.5.

## Wiki
Please see detailed instructions for various tasks at the [wiki](https://github.nrel.gov/jannoni/hybrid_systems/wiki)

## Setup
Someday there will be a startup script, in the meantime, if you have Anaconda installed follow that set of instructions, otherwise, follow the Virtualenv set.

### Clone the git
1. Install Git
https://git-scm.com/
2. Navigate to the directory where the repo will go
3. Clone project: 
- `git clone https://github.nrel.gov/jannoni/hybrid_systems`
3. Naviate to the repo
- `cd hybrid_systems`

### Anaconda
1. Create new conda environment:
- `conda create --name env python=3.7`
2. Activate:
- `conda activate env`
3. Install requirements
- `pip install -r requirements.txt`

### Virtualenv
1. Install Python 3:
- https://www.python.org/downloads/
2. Setup virtual environment (replace `/usr/bin/python3.7` with `C:\Python37` on Windows), and the specific python version number if not 3.7
- `virtualenv --python=/usr/bin/python3.7 env`
3. Activate virtual environment
- `source env/bin/activate` (Linux/Mac)
- `env\Scripts\activate.bat` (Windows)
4. Install requirements
Using virtualenv:
` pip install -r requirements.txt`
5. Run the currently setup example
`python main.py`

## Adding additional Python packages
It's likely that at some point, you'll want to add additional python dependencies to the project.  To do so:

- `pip install <package>`

Then, please add the dependency to the requirements.txt file, which can be done:

- `pip freeze > requirements.txt` (Windows)
- `pip freeze | tee requirements.txt` (Linux/OSX)


