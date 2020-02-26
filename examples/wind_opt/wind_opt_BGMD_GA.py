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
from examples.wind_opt.wind_optimization_problem_bgm import (
    WindOptimizationProblemBGM,
    BGMCandidate,
    )
from examples.wind_opt.wind_optimization_problem_bgmd import (
    BGMDCandidate,
    WindOptimizationProblemBGMD,
    )
from optimization.candidate_converter.object_converter import ObjectConverter
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


class WindOptBGMD_GA(ConvertingOptimizationDriver):
    
    def __init__(self,
                 problem: WindOptimizationProblemBGMD,
                 generation_size: int = 200,
                 selection_proportion: float = .5,
                 prior_sigma_scale: float = 1.0,
                 prior_scale: float = 1.0,
                 optimizer=None,
                 **kwargs
                 ) -> None:
        self.problem = problem
        num_turbines = problem.inner_problem.num_turbines
        
        boundary = problem.inner_problem.site_info.boundary
        d = boundary.length / num_turbines
        
        prior = self.problem.generate_prior(GaussianDimension,
                                            callback_three=lambda: prior_scale)

        # prior.ratio_turbines = GaussianDimension(1.0, .5, prior_scale)
        # prior.border_ratio = GaussianDimension(.45, .5, prior_scale)
        # prior.border_offset = GaussianDimension(.5, 1, prior_scale)
        # prior.theta = GaussianDimension(0, 2 * pi, prior_scale)
        # # prior.dx = GaussianDimension(50, 1000, prior_scale)
        # prior.grid_aspect = GaussianDimension(0.0, 2, prior_scale)
        # prior.b = GaussianDimension(.2, .4, prior_scale)

        optimizer = GAOptimizer(generation_size, selection_proportion) if optimizer is None else optimizer
        super().__init__(
            AskTellParallelDriver(),
            # AskTellSerialDriver(),
            optimizer,
            # PassthroughConverter(),
            ObjectConverter(),
            prior,
            lambda candidate: problem.objective(candidate),
            conformer=lambda candidate: self.problem.make_conforming_candidate_and_get_penalty(candidate)[0],
            **kwargs
            )
