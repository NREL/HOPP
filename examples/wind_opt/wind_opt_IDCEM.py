from examples.wind_opt.wind_opt_DCEM import WindOptDCEM
from examples.wind_opt.wind_optimization_problem import WindOptimizationProblem
from optimization.optimizer.IDCEM import IDCEM


class WindOptIDCEM(WindOptDCEM):
    
    def __init__(self,
                 problem: WindOptimizationProblem,
                 generation_size: int = 200,
                 selection_size: int = 50,
                 **kwargs) -> None:
        super().__init__(problem, optimizer=IDCEM(generation_size, selection_size), **kwargs)
