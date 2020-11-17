from typing import (
    Optional,
    Tuple,
    )

# import shapely
from .DCEM_optimizer import DCEMOptimizer


# sys.path.append('../examples/flatirons')
# import func_tools
# matplotlib.use('tkagg')


class IDCEM(DCEMOptimizer):
    """
    A prototype implementation of an incremental decomposed cross-entropy method.
    """
    
    def __init__(self,
                 generation_size: int,
                 selection_size: int,
                 **kwargs
                 ) -> None:
        super().__init__(generation_size, 1.0, **kwargs)
        self._selection_size: int = selection_size
        self._population: [Tuple[float, any]] = []
    
    def tell(self, evaluations: [Tuple[float, any]]) -> None:
        """
        Updates the optimizer with the objective evaluations of a list of search points
        :param evaluations: a list of tuples of (evaluation, search point)
        """
        # print('eval: ', [sample[0] for sample in evaluations])
        self._population.extend(evaluations)
        self._population.sort(key=lambda evaluation: evaluation[0], reverse=True)
        del self._population[self._selection_size:]
        # print('pop: ', [sample[0] for sample in self.population])
        
        for i, dimension in enumerate(self._dimensions):
            dimension.update([evaluation[1][i] for evaluation in self._population])
    
    def best_solution(self) -> (Optional[float], any):
        """
        :return: the current best solution and (estimated) score
        """
        return self._population[0] if len(self._population) > 0 else super().best_solution()
