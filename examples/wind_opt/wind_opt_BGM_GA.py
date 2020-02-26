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


class WindOptBGM_GA(ConvertingOptimizationDriver):
    
    def __init__(self,
                 problem: WindOptimizationProblemBGM,
                 generation_size: int = 200,
                 selection_proportion: float = .5,
                 prior_sigma_scale: float = 1.0,
                 prior_scale: float = 2.0,
                 optimizer=None,
                 conformer=None,
                 ) -> None:
        self.problem = problem
        num_turbines = problem.inner_problem.num_turbines
        
        boundary = problem.inner_problem.site_info.boundary
        d = boundary.length / num_turbines
        
        prior = BGMCandidate()
        prior.d = GaussianDimension(2000, 2000, prior_scale)
        prior.border_offset = GaussianDimension(.5, 2, prior_scale)
        prior.theta = GaussianDimension(0, 4*pi, prior_scale)
        prior.dx = GaussianDimension(200, 1000, prior_scale)
        prior.dy = GaussianDimension(200, 1000, prior_scale)
        prior.b = GaussianDimension(0, 2, prior_scale)
        
        optimizer = GAOptimizer(generation_size, selection_proportion) if optimizer is None else optimizer
        super().__init__(
            AskTellParallelDriver(),
            # AskTellSerialDriver(),
            optimizer,
            # PassthroughConverter(),
            ObjectConverter(),
            prior,
            lambda candidate: problem.objective(candidate),
            conformer=lambda candidate: self.problem.make_conforming_candidate_and_get_penalty(candidate)[0]
            )
