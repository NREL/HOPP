import matplotlib
# matplotlib.use('tkagg')
# import shapely

# sys.path.append('../examples/flatirons')
# import func_tools
from examples.solar_wind_opt.solar_wind_optimization_problem import SolarWindOptimizationProblem
from examples.wind_opt.wind_optimization_problem import WindOptimizationProblem
from optimization.candidate_converter.dict_converter import DictConverter
from optimization.candidate_converter.object_converter import ObjectConverter
from optimization.candidate_converter.passthrough_converter import PassthroughConverter
from optimization.driver.ask_tell_parallel_driver import AskTellParallelDriver
from optimization.driver.ask_tell_serial_driver import AskTellSerialDriver
from optimization.optimizer.DCEM import DCEM
from optimization.optimizer.converting_optimization_driver import ConvertingOptimizationDriver


class SolarWindOptDCEM(ConvertingOptimizationDriver):
    
    def __init__(self,
                 problem: SolarWindOptimizationProblem,
                 generation_size: int = 200,
                 selection_proportion: float = .5,
                 prior_sigma_scale: float = 1.0,
                 prior_scale: float = 1.0,
                 optimizer=None,
                 conformer=None,
                 ) -> None:
        self.problem = problem
        num_turbines = problem.num_turbines
        
        boundary = problem.site_info.boundary
        d = boundary.length / problem.num_turbines
        positions = problem.site_info.get_evenly_spaced_points_along_border(boundary, d)
        
        prototype = {}
        prototype['gcr'] = DCEM.GaussianDimension(0.4, .1 * prior_sigma_scale, prior_scale)
        prototype['turb_x'] = [DCEM.GaussianDimension(position.x, d * prior_sigma_scale, prior_scale)
                               for position in positions]
        prototype['turb_y'] = [DCEM.GaussianDimension(position.y, d * prior_sigma_scale, prior_scale)
                               for position in positions]
        
        optimizer = DCEM(generation_size, selection_proportion) if optimizer is None else optimizer
        super().__init__(
            AskTellParallelDriver(),
            # AskTellSerialDriver(),
            optimizer,
            ObjectConverter(),
            prototype,
            lambda candidate: problem.objective(candidate),
            conformer=conformer
            )
    
    def plot_distribution(self, ax, color, alpha):
        num_turbines = self.problem.num_turbines
        for i in range(num_turbines):
            e = matplotlib.patches.Ellipse(
                xy=(self.optimizer.dimensions[i].mu, self.optimizer.dimensions[num_turbines + i].mu),
                width=self.optimizer.dimensions[i].sigma,
                height=self.optimizer.dimensions[num_turbines + i].sigma,
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
