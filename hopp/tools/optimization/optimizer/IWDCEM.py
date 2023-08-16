import math
from abc import ABC
from collections import deque
from typing import (
    Optional,
    Tuple,
    )

# import shapely
from .DCEM_optimizer import DCEMOptimizer


# sys.path.append('../examples/flatirons')
# import func_tools
# matplotlib.use('tkagg')


class IWDCEM(DCEMOptimizer, ABC):
    """
    A prototype implementation of an incremental windowed decomposed cross-entropy method.
    """
    
    def __init__(self,
                 generation_size: int,
                 window_size: int,
                 selection_proportion: float,
                 **kwargs
                 ) -> None:
        super().__init__(generation_size, selection_proportion, **kwargs)
        self._window_size: int = window_size
        self._population = deque()
        self._sorted_population: [any] = []
        self._selection_size: int = math.ceil(self._selection_proportion * self._window_size)
        # print('iwdcem: ', self.generation_size, self.window_size, self.selection_proportion)
    
    def tell(self, evaluations: [Tuple[float, any]]) -> None:
        """
        Updates the optimizer with the objective evaluations of a list of search points
        :param evaluations: a list of tuples of (evaluation, search point)
        """
        print('eval: ', [sample[0] for sample in evaluations])
        
        # this could be optimized using a sorted dictionary, binary tree, etc
        for evaluation in evaluations:
            while len(self._population) >= self._window_size:
                self._population.pop()
            self._population.appendleft(evaluation)
        
        self._sorted_population = sorted(self._population, key=lambda evaluation: evaluation[0], reverse=True)
        del self._sorted_population[self._selection_size:]
        print('sel: ', [sample[0] for sample in self._sorted_population])
        
        for i, dimension in enumerate(self._dimensions):
            dimension.update([evaluation[1][i] for evaluation in self._sorted_population])
    
    def best_solution(self) -> (Optional[float], any):
        """
        :return: the current best solution and (estimated) score
        """
        if len(self._sorted_population) <= 0:
            return super().best_solution()
        return self._sorted_population[0]
