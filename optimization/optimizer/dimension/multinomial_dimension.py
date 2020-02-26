import bisect
import random
from typing import Union

from optimization.optimizer.dimension.dimension_info import DimensionInfo


class MultinomialDimension(DimensionInfo):
    
    def __init__(self, probabilities: [float]):
        self.cumulative_distribution: [float] = self._get_cumulative_distribution(probabilities)
    
    def update(self, samples: [int]) -> None:
        counts = [0] * len(self.cumulative_distribution)
        
        for sample in samples:
            counts[sample] = counts[sample] + 1
        
        cumulative_counts = self._get_cumulative_distribution(counts)
        
        total = float(len(samples))
        for i in range(len(cumulative_counts)):
            self.cumulative_distribution[i] = cumulative_counts[i] / total
    
    def sample(self) -> int:
        return bisect.bisect_right(self.cumulative_distribution, random.random())
    
    def best(self) -> Union[float, int]:
        # could be memoized at update time
        last = 0.0
        largest = (0.0, 0)
        for i, c in enumerate(self.cumulative_distribution):
            p = c - last
            last = c
            if p >= largest[0]:
                largest = (p, i)
        return largest[1]
    
    def distribution(self):
        result = [0.0] * len(self.cumulative_distribution)
        last = 0.0
        for i, c in enumerate(self.cumulative_distribution):
            p = c - last
            result[i] = p
        return result
    
    def _get_cumulative_distribution(self, probabilities):
        cumulative = [0] * len(probabilities)
        acc = 0
        for i, p in enumerate(probabilities):
            acc = acc + p
            self.cumulative_distribution[i] = acc
        return cumulative
    
    def mean(self) -> Union[float, int]:
        acc = 0.0
        for i, p in enumerate(self.distribution()):
            acc += i * p
        return acc
    
    def variance(self) -> Union[float, int]:
        raise Exception('Not Implemented')
