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

from typing import Dict
import matplotlib as mpl

mpl.use('Agg')

import os
from dotenv import load_dotenv

import numpy as np
from matplotlib.animation import (
    PillowWriter,
    )
from matplotlib.lines import Line2D

from hopp.tools.optimization import (
    setup_run,
    DataRecorder
    )
from hopp.simulation.technologies.sites import make_circular_site, make_irregular_site, SiteInfo
from hopp.utilities.log import opt_logger as logger
from hopp.simulation.technologies.sites import locations
from hopp.utilities.keys import set_developer_nrel_gov_key
from hopp.simulation.technologies.layout.plot_tools import *

from examples.optimization.layout_opt.parametrized_optimization_driver import ParametrizedOptimizationDriver
from examples.optimization.layout_opt.hybrid_optimization_problem import HybridOptimizationProblem
from examples.optimization.layout_opt.hybrid_parametrization import HybridParametrization

np.set_printoptions(precision=2, threshold=10000, linewidth=240)

# Set API key
load_dotenv()
NREL_API_KEY = os.getenv("NREL_API_KEY")
set_developer_nrel_gov_key(NREL_API_KEY)  # Set this key manually here if you are not setting it using the .env


def run(default_config: Dict) -> None:
    config, output_path, run_name = setup_run(default_config)
    recorder = DataRecorder.make_data_recorder(output_path)

    max_evaluations = config['max_evaluations']
    
    location_index = config['location']
    location = locations[location_index]
    
    site = config['site']
    site_data = None
    if site == 'circular':
        site_data = make_circular_site(lat=location[0], lon=location[1], elev=location[2])
    elif site == 'irregular':
        site_data = make_irregular_site(lat=location[0], lon=location[1], elev=location[2])
    else:
        raise Exception("Unknown site '" + site + "'")
    
    site_info = SiteInfo(site_data)
    inner_problem = HybridOptimizationProblem(site_info, config['num_turbines'], config['solar_capacity'])
    problem = HybridParametrization(inner_problem)
    
    optimizer = ParametrizedOptimizationDriver(problem, recorder=recorder, **config['optimizer_config'])
    
    figure = plt.figure(1)
    axes = figure.add_subplot(111)
    axes.set_aspect('equal')
    plt.grid()
    plt.tick_params(which='both', labelsize=15)
    plt.xlabel('x (m)', fontsize=15)
    plt.ylabel('y (m)', fontsize=15)
    site_info.plot()

    score, evaluation, best_solution = optimizer.central_solution()
    score, evaluation = problem.objective(best_solution) if score is None else score
    
    print(-1, ' ', score, evaluation)
    
    print('setup 1')
    
    num_substeps = 1
    figure, axes = plt.subplots(dpi=200)
    axes.set_aspect(1)
    animation_writer = PillowWriter(2 * num_substeps)
    animation_writer.setup(figure, os.path.join(output_path, 'trajectory.gif'), dpi=200)
    
    print('setup 2')
    _, _, central_solution = optimizer.central_solution()
    
    print('setup 3')
    bounds = problem.inner_problem.site_info.polygon.bounds
    site_sw_bound = np.array([bounds[0], bounds[1]])
    site_ne_bound = np.array([bounds[2], bounds[3]])
    site_center = .5 * (site_sw_bound + site_ne_bound)
    max_delta = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    reach = (max_delta / 2) * 1.3
    min_plot_bound = site_center - reach
    max_plot_bound = site_center + reach
    
    print('setup 4')
    
    best_score, best_evaluation, best_solution = 0.0, 0.0, None
    
    def plot_candidate(candidate):
        nonlocal best_score, best_evaluation, best_solution
        axes.cla()
        axes.set(xlim=(min_plot_bound[0], max_plot_bound[0]), ylim=(min_plot_bound[1], max_plot_bound[1]))
        wind_color = (153 / 255, 142 / 255, 195 / 255)
        solar_color = (241 / 255, 163 / 255, 64 / 255)
        central_color = (.5, .5, .5)
        conforming_candidate, _, __ = problem.make_conforming_candidate_and_get_penalty(candidate)
        problem.plot_candidate(conforming_candidate, figure, axes, central_color, central_color, alpha=.7)
        
        if best_solution is not None:
            conforming_best, _, __ = problem.make_conforming_candidate_and_get_penalty(best_solution)
            problem.plot_candidate(conforming_best, figure, axes, wind_color, solar_color, alpha=1.0)
            axes.set_xlabel('Best Solution AEP: {}'.format(best_evaluation))
        else:
            axes.set_xlabel('')
        
        axes.legend([
            Line2D([0], [0], color=wind_color, lw=8),
            Line2D([0], [0], color=solar_color, lw=8),
            Line2D([0], [0], color=central_color, lw=8),
            ],
            ['Wind Layout', 'Solar Layout', 'Mean Search Vector'],
            loc='lower left')
        animation_writer.grab_frame()
    
    print('plot candidate')
    
    plot_candidate(central_solution)
    
    central_prev = central_solution
    # TODO: make a smooth transition between points
    # TODO: plot exclusion zones
    print('begin')

    try:
        while optimizer.num_evaluations() < max_evaluations:

            print('step start')
            logger.info("Starting step, num evals {}".format(optimizer.num_evaluations()))
            optimizer.step()
            print('step end')

            proportion = min(1.0, optimizer.num_evaluations() / max_evaluations)
            g = 1.0 * proportion
            b = 1.0 - g
            a = .5
            color = (b, g, b)
            best_score, best_evaluation, best_solution = optimizer.best_solution()
            central_score, central_evaluation, central_solution = optimizer.central_solution()

            a1 = optimizer.converter.convert_from(central_prev)
            b1 = optimizer.converter.convert_from(central_solution)
            a = np.array(a1, dtype=np.float64)
            b = np.array(b1, dtype=np.float64)

            for i in range(num_substeps):
                p = (i + 1) / num_substeps
                c = (1 - p) * a + p * b
                candidate = optimizer.converter.convert_to(c)
                plot_candidate(candidate)

            central_prev = central_solution
            print(optimizer.num_iterations(), ' ', optimizer.num_evaluations(), best_score, best_evaluation)
    except:
        raise RuntimeError("Optimizer error encountered. Try modifying the config to use larger generation_size if"
                           " encountering singular matrix errors.")

    animation_writer.finish()

    optimizer.close()

    print("Results and animation written to " + os.path.abspath(output_path))


if __name__ == '__main__':

    default_config = {
        'name':             't2',
        'location':         1,
        'site':             'irregular',
        'solar_capacity':   50000,  # kW
        'num_turbines':     50,  #
        'max_evaluations':  20,
        'optimizer_config': {
            'method':               'CMA-ES',
            'nprocs': 1,
            'generation_size':      10,
            'selection_proportion': .33,
            'prior_scale':          1.0,
            'prior_params':         {
                # "grid_angle": {
                #     "mu": 0.1
                #     }
                }
            }
        }

    run(default_config)
