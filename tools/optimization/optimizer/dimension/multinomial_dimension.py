import bisect
import random
from typing import Union

from tools.optimization.optimizer import DimensionInfo


class Multinomial(DimensionInfo):
    
    def __init__(self, probabilities: [float]):
        """
        Multinominal distribution for a single trial
        :param probabilities: event probabilities, sum(probabilities) = 1
        """
        self.cumulative_distribution: [float] = self._get_cumulative_distribution(probabilities)
    
    def update(self, samples: [int]) -> None:
        """
        Update parameters of the pdf with new samples
        :param samples: list of best candidates from optimizer
        """
        counts = [0] * len(self.cumulative_distribution)
        
        for sample in samples:
            counts[sample] = counts[sample] + 1
        
        cumulative_counts = self._get_cumulative_distribution(counts)
        
        total = float(len(samples))
        for i in range(len(cumulative_counts)):
            self.cumulative_distribution[i] = cumulative_counts[i] / total
    
    def sample(self) -> int:
        """
        :return: Sample from the pdf
        """
        return bisect.bisect_right(self.cumulative_distribution, random.random())
    
    def best(self) -> Union[float, int]:
        """
        :return: Most likely sample
        """
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
        """
        :return: return event probabilities
        """
        result = [0.0] * len(self.cumulative_distribution)
        last = 0.0
        for i, c in enumerate(self.cumulative_distribution):
            p = c - last
            result[i] = p
        return result
    
    def _get_cumulative_distribution(self, probabilities):
        """
        :param probabilities: event probabilities
        """
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
