"""
A prototype application of the distributed cross-entropy method to the wind optimization problem.
In this basic implementation, the number of turbines is fixed and the generative distribution is uncorrelated.

TODO:
 + Add boundary constraints / penalties
 + Add proximity constraints
 + Better order turbine locations
 + Investigate turbine number as an attribute
 + Investigate modeling parameter covariances
 + Investigate other distribution types
 + Investigate parameter transformations
 + Add solar
 + Add storage
 + Add cabling, etc
 + investigate organic approach
"""

# matplotlib.use('tkagg')
import os
from typing import Dict
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from tools.optimization import (
    setup_run,
    DataRecorder
    )
from hybrid.sites import SiteInfo, flatirons_site
from hybrid.keys import set_developer_nrel_gov_key

from parametrized_optimization_driver import ParametrizedOptimizationDriver
from wind_optimization_problem import WindOptimizationProblem
from wind_parametrization import WindParametrization

# Set API key
load_dotenv()
NREL_API_KEY = os.getenv("NREL_API_KEY")
set_developer_nrel_gov_key(NREL_API_KEY)  # Set this key manually here if you are not setting it using the .env

# Set printing options
np.set_printoptions(precision=2, threshold=10000, linewidth=240)


def run(default_config: Dict) -> None:
    config, output_path, run_name = setup_run(default_config)
    recorder = DataRecorder.make_data_recorder(output_path)
    
    max_evaluations = config['max_evaluations']
    optimizer_config = config['optimizer_config']
    
    site_info = SiteInfo(flatirons_site)
    inner_problem = WindOptimizationProblem(site_info, config['num_turbines'])
    problem = WindParametrization(inner_problem)
    
    optimizer = ParametrizedOptimizationDriver(problem, recorder=recorder, **optimizer_config)
    
    figure = plt.figure(1)
    ax = figure.add_subplot(111)
    plt.grid()
    plt.tick_params(which='both', labelsize=15)
    plt.xlabel('x (m)', fontsize=15)
    plt.ylabel('y (m)', fontsize=15)
    site_info.plot()
    
    score, evaluation, best_solution = optimizer.central_solution()
    score, evaluation = problem.objective(best_solution) if score is None else score
    
    print(-1, ' ', score)
    
    optimizer.problem.plot_candidate(best_solution, (1.0, 0, 0), .2)
    
    prev = optimizer.best_solution()[1]
    try:
        while optimizer.num_evaluations() < max_evaluations:
            print('step start')
            optimizer.step()
            print('step end')

            proportion = min(1.0, optimizer.num_evaluations() / max_evaluations)
            g = 1.0 * proportion
            b = 1.0 - g
            a = .5
            color = (b, g, b)
            score, eval, best = optimizer.best_solution()
            score = problem.objective(best) if score is None else score
            problem.plot_candidate(best, color, .3)
            prev = best
            print(optimizer.num_iterations(), ' ', optimizer.num_evaluations(), score)

    except:
        raise RuntimeError("Optimizer error encountered. Try modifying the config to use larger generation_size if"
                           " encountering singular matrix errors.")

    print('best: ', optimizer.best_solution().__repr__())
    optimizer.problem.plot_candidate(optimizer.best_solution()[2], (0, 0, 0), 1.0)

    # Create the figure
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=(0, 0, 0), label='Optimal')]
    plt.legend(handles=legend_elements)
    plt.show()
    
    optimizer.close()

if __name__ == '__main__':

    default_config = {
        'name':             'test',
        'num_turbines':     20,
        'max_evaluations':  20,
        'optimizer_config': {
            'method':               'CEM',
            'nprocs':               1,
            'generation_size':      10,
            'selection_proportion': .5,
            'prior_scale':          1.0,
            }
        }

    run(default_config)
