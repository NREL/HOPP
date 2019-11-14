from typing import (
    Tuple,
    )

# import shapely
from optimization.optimizer.DCEM import DCEM


# sys.path.append('../examples/flatirons')
# import func_tools
# matplotlib.use('tkagg')


class IDCEM(DCEM):
    """
    A prototype implementation of an incremental decomposed cross-entropy method.
    """
    
    def __init__(self,
                 generation_size: int,
                 selection_size: int,
                 **kwargs
                 ) -> None:
        super().__init__(generation_size, 1.0, **kwargs)
        self.selection_size: int = selection_size
        self.population: [Tuple[float, any]] = []
    
    def tell(self, evaluations: [Tuple[float, any]]) -> None:
        # print('eval: ', [sample[0] for sample in evaluations])
        self.population.extend(evaluations)
        self.population.sort(key=lambda evaluation: evaluation[0], reverse=True)
        del self.population[self.selection_size:]
        # print('pop: ', [sample[0] for sample in self.population])
        
        for i, dimension in enumerate(self.dimensions):
            dimension.update([evaluation[1][i] for evaluation in self.population])
    
    def best(self) -> any:
        return self.population[0][1] if len(self.population) > 0 else super().best()
    
