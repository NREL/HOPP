import functools
import math
import random
from typing import (
    Optional,
    Tuple,
    List,
    )

# import shapely
from optimization.optimizer.DCEM import DCEM


# sys.path.append('../examples/flatirons')
# import func_tools
# matplotlib.use('tkagg')


class ParticleGradientOptimizer(DCEM):
    """
    A prototype implementation of a particle graident method
    """
    
    def __init__(self,
                 generation_size: int,
                 population_size: int,
                 scale: float,
                 **kwargs
                 ) -> None:
        super().__init__(generation_size, 1.0, **kwargs)
        self.population_size: int = population_size
        self.scale: float = scale
        self.population = []
    
    def ask(self, num: Optional[int] = None) -> [any]:
        if len(self.population) == 0:
            return super().ask(num)
        
        if num is None:
            num = self.generation_size
        
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
            
            candidate = [0.0] * len(self.dimensions)
            for i, dimension in enumerate(self.dimensions):
                
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
                
                extension = random.expovariate(1.0 / self.scale)
                difference = a[1][i] - b[1][i]
                # print(extension, difference)
                delta = extension * difference
                candidate[i] = base[1][i] + delta
            population.append(candidate)
        return population
    
    def tell(self, evaluations: [Tuple[float, any]]) -> None:
        print('eval: ', [sample[0] for sample in evaluations[:10]])
        self.population.extend(evaluations)
        self.population.sort(key=lambda evaluation: evaluation[0], reverse=True)
        self.population = self.population[0:self.population_size]
        print('pop: ', [sample[0] for sample in self.population])
        
        for i, dimension in enumerate(self.dimensions):
            dimension.update([evaluation[1][i] for evaluation in self.population])
    
    def best(self) -> any:
        return self.population[0][1] if len(self.population) > 0 else super().best()
    
    def get_random_sample(self):
        return self.population[random.randrange(len(self.population))]
