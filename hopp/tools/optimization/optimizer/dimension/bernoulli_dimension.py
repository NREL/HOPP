import random
from typing import Union

from tools.optimization.optimizer import DimensionInfo


class Bernoulli(DimensionInfo):
    
    def __init__(self, p: float = .5, a: Union[float, int] = 0, b: Union[float, int] = 1):
        """
        Bernoulli Distribution
        :param p: probability of value a, else value b
        """
        self.p: float = p
        self.a: Union[float, int] = a
        self.b: Union[float, int] = b
    
    def update(self, samples: [float]) -> None:
        """
        Update parameters of the pdf with new samples
        :param samples: list of best candidates from optimizer
        """
        pass
    
    def sample(self) -> Union[float, int]:
        """
        :return: Sample from the pdf
        """
        return self.a if random.random() < self.p else self.b
    
    def best(self) -> float:
        """
        :return: Most likely sample
        """
        return self.p
    
    def mean(self) -> Union[float, int]:
        return self.p
    
    def variance(self) -> Union[float, int]:
        return self.p * (1.0 - self.p)
