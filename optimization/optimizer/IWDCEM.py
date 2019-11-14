import math
from collections import deque
from typing import Tuple

# import shapely
from optimization.optimizer.DCEM import DCEM


# sys.path.append('../examples/flatirons')
# import func_tools
# matplotlib.use('tkagg')


class IWDCEM(DCEM):
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
        self.window_size: int = window_size
        self.population = deque()
        self.sorted_population: [any] = []
        self.selection_size: int = math.ceil(self.selection_proportion * self.window_size)
        # print('iwdcem: ', self.generation_size, self.window_size, self.selection_proportion)
    
    def tell(self, evaluations: [Tuple[float, any]]) -> None:
        print('eval: ', [sample[0] for sample in evaluations])
        
        # this could be optimized using a sorted dictionary, binary tree, etc
        for evaluation in evaluations:
            while len(self.population) >= self.window_size:
                self.population.pop()
            self.population.appendleft(evaluation)
        
        self.sorted_population = sorted(self.population, key=lambda evaluation: evaluation[0], reverse=True)
        del self.sorted_population[self.selection_size:]
        print('sel: ', [sample[0] for sample in self.sorted_population])
        
        for i, dimension in enumerate(self.dimensions):
            dimension.update([evaluation[1][i] for evaluation in self.sorted_population])
    
    def best(self) -> any:
        if len(self.sorted_population) <= 0:
            return super().best()
        return self.sorted_population[0][1]
