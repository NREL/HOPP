from examples.wind_opt.wind_opt_DCEM import WindOptDCEM
from examples.wind_opt.wind_optimization_problem import WindOptimizationProblem
from optimization.optimizer.DCEM_optimizer import DCEMOptimizer


class WindOptDCEMC(WindOptDCEM):
    
    def __init__(self,
                 problem: WindOptimizationProblem,
                 generation_size: int = 200,
                 selection_proportion: float = .5,
                 **kwargs) -> None:
        super().__init__(problem,
                         optimizer=DCEMOptimizer(generation_size, selection_proportion),
                         conformer=problem.make_conforming_candidate,
                         **kwargs)
