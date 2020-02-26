import functools
import math
import random
from typing import (
    Optional,
    Tuple,
    )

# import shapely
from optimization.optimizer.DCEM_optimizer import DCEMOptimizer


# sys.path.append('../examples/flatirons')
# import func_tools
# matplotlib.use('tkagg')


class ParticleGradientOptimizer(DCEMOptimizer):
    """
    A prototype implementation of a particle gradient method
    """
    
    def __init__(self,
                 generation_size: int,
                 population_size: int,
                 scale: float,
                 **kwargs
                 ) -> None:
        super().__init__(generation_size, 1.0, **kwargs)
        self._population_size: int = population_size
        self._scale: float = scale
        self._population = []
    
    def ask(self, num: Optional[int] = None) -> [any]:
        if len(self._population) == 0:
            return super().ask(num)
        
        if num is None:
            num = self._generation_size
        
        # population = []
        # for _ in range(num):
        #     candidate = [0.0] * len(self.dimensions)
        #     for i, dimension in enumerate(self.dimensions):
        #         base = self.get_random_sample()
        #         second = self.get_random_sample()
        #
        #         if base[0] > second[0]:
        #             a = base
        #             b = second
        #         else:
        #             a = second
        #             b = base
        #
        #         extension = random.expovariate(1.0 / self.scale)
        #         # print(extension)
        #         delta = extension * (a[1][i] - b[1][i])
        #         candidate[i] = base[1][i] + delta
        #     population.append(candidate)
        population = []
        for _ in range(num):
            base = self.get_random_sample()
            
            candidate = [0.0] * len(self._dimensions)
            for i, dimension in enumerate(self._dimensions):
                
                # second = self.get_random_sample()
                
                second = functools.reduce(
                    lambda a, b: a if 0 < math.fabs(a[1][i] - base[1][i]) <= math.fabs(b[1][i] - base[1][i]) else b,
                    [self.get_random_sample() for _ in range(1)], self.get_random_sample())
                
                # print('del ', base[1][i] - second[1][i])
                
                if base[0] > second[0]:
                    a = base
                    b = second
                else:
                    a = second
                    b = base
                
                extension = random.expovariate(1.0 / self._scale)
                difference = a[1][i] - b[1][i]
                # print(extension, difference)
                delta = extension * difference
                candidate[i] = base[1][i] + delta
            population.append(candidate)
        return population
    
    def tell(self, evaluations: [Tuple[float, any]]) -> None:
        print('eval: ', [sample[0] for sample in evaluations[:10]])
        self._population.extend(evaluations)
        self._population.sort(key=lambda evaluation: evaluation[0], reverse=True)
        self._population = self._population[0:self._population_size]
        print('pop: ', [sample[0] for sample in self._population])
        
        for i, dimension in enumerate(self._dimensions):
            dimension.update([evaluation[1][i] for evaluation in self._population])
    
    def best_solution(self) -> (Optional[float], any):
        return self._population[0] if len(self._population) > 0 else super().best_solution()
    
    def get_random_sample(self):
        return self._population[random.randrange(len(self._population))]
