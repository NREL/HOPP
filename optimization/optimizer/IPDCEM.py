import random
from typing import (
    Optional,
    Tuple,
    )

# import shapely
from optimization.optimizer.DCEM import DCEM


# sys.path.append('../examples/flatirons')
# import func_tools
# matplotlib.use('tkagg')


class IPDCEM(DCEM):
    """
    A prototype implementation of an incremental decomposed cross-entropy method.
    """
    
    def __init__(self, generation_size: int, selection_size: int, scale: float,
                 **kwargs
                 ) -> None:
        super().__init__(generation_size, 1.0, **kwargs)
        self.selection_size: int = selection_size
        self.scale: float = scale
        self.population: [Tuple[float, any]] = []
    
    def ask(self, num: Optional[int] = None) -> [any]:
        
        if len(self.population) == 0:
            return super().ask(num)
        
        if num is None:
            num = self.generation_size
        
        population = []
        for _ in range(num):
            base = self.population[random.randrange(len(self.population))][1]
            candidate = [0.0] * len(self.dimensions)
            for i, dimension in enumerate(self.dimensions):
                candidate[i] = base[i] + (dimension.sample() - dimension.best()) * self.scale
            population.append(candidate)
        
        return population
    
    def tell(self, evaluations: [Tuple[float, any]]) -> None:
        print('eval: ', [sample[0] for sample in evaluations])
        self.population.extend(evaluations)
        self.population.sort(key=lambda evaluation: evaluation[0], reverse=True)
        # self.population = self.population[0:self.selection_size]
        del self.population[self.selection_size:]
        print('pop: ', [sample[0] for sample in self.population])
        
        for i, dimension in enumerate(self.dimensions):
            dimension.update([evaluation[1][i] for evaluation in self.population])
    
    def best(self) -> any:
        return self.population[0][1] if len(self.population) > 0 else super().best()
