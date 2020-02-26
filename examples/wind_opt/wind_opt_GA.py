from math import *
import random

import matplotlib
from shapely import affinity
from shapely.geometry import (
    LineString,
    Point,
    )

from examples.wind_opt import turbine_layout_tools
from examples.wind_opt.wind_optimization_problem import WindOptimizationProblem
from optimization.candidate_converter.passthrough_converter import PassthroughConverter
from optimization.driver.ask_tell_parallel_driver import AskTellParallelDriver
from optimization.driver.ask_tell_serial_driver import AskTellSerialDriver
from optimization.optimizer.GA_optimizer import GAOptimizer
from optimization.optimizer.SPSA_optimizer import SPSAOptimizer
from optimization.optimizer.converting_optimization_driver import ConvertingOptimizationDriver
from optimization.optimizer.dimension.centered_bernoulli_dimension import CenteredBernoulliDimension

# matplotlib.use('tkagg')
# import shapely
# sys.path.append('../examples/flatirons')
# import func_tools
from optimization.optimizer.dimension.gaussian_dimension import GaussianDimension


class WindOptGA(ConvertingOptimizationDriver):
    
    def __init__(self,
                 problem: WindOptimizationProblem,
                 generation_size: int = 100,
                 selection_proportion: float = .5,
                 prior_sigma_scale: float = 1.0,
                 prior_scale: float = 1.0,
                 optimizer=None,
                 conformer=None,
                 **kwargs
                 ) -> None:
        
        self.problem = problem
        num_turbines = problem.num_turbines
        
        boundary = problem.site_info.boundary
        d = boundary.length / num_turbines
        positions = turbine_layout_tools.get_evenly_spaced_points_along_border(boundary, d)
        
        prototype = [None] * (num_turbines * 2)
        for i in range(num_turbines):
            position = positions[i]
            prototype[i] = GaussianDimension(position.x, d * prior_sigma_scale, prior_scale)
            prototype[num_turbines + i] = GaussianDimension(position.y, d * prior_sigma_scale, prior_scale)
        
        optimizer = GAOptimizer(generation_size, selection_proportion) if optimizer is None else optimizer
        super().__init__(
            AskTellParallelDriver(),
            # AskTellSerialDriver(),
            optimizer,
            PassthroughConverter(),
            # ObjectConverter(),
            prototype,
            lambda candidate: problem.objective(candidate),
            conformer=lambda c: problem.make_conforming_candidate(c),
            **kwargs
            )
    
    def plot_distribution(self, ax, color, alpha):
        num_turbines = self.problem.num_turbines
        for i in range(num_turbines):
            e = matplotlib.patches.Ellipse(
                xy=(self._optimizer.dimensions[i].mu, self._optimizer.dimensions[num_turbines + i].mu),
                width=self._optimizer.dimensions[i].sigma,
                height=self._optimizer.dimensions[num_turbines + i].sigma,
                angle=0)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(alpha)
            e.set_facecolor(color)
            # be = matplotlib.patches.Ellipse(
            #     xy=(self.optimizer.mean[i], self.optimizer.mean[self.num_turbines + i]),
            #     width=self.optimizer.self.standard_deviation[i],
            #     height=self.optimizer.self.standard_deviation[self.num_turbines + i],
            #     angle=0)
            #
            # ax.add_artist(be)
            # be.set_clip_box(ax.bbox)
            # be.set_alpha(alpha)
            # be.set_facecolor('none')
            # be.set_edgecolor(color)
        
        # plt.plot(self.optimizer.mean[0:num_turbines], self.optimizer.mean[num_turbines:], 'x', color=color, alpha=0.9)
