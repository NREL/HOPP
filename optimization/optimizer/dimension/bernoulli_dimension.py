import random
from typing import Union

from optimization.optimizer.dimension.dimension_info import DimensionInfo


class BernoulliDimension(DimensionInfo):
    
    def __init__(self, p: float = .5, a: Union[float, int] = 0, b: Union[float, int] = 1):
        self.p: float = p
        self.a: Union[float, int] = a
        self.b: Union[float, int] = b
    
    def update(self, samples: [float]) -> None:
        pass
    
    def sample(self) -> Union[float, int]:
        return self.a if random.random() < self.p else self.b
    
    def best(self) -> float:
        return self.p
    
    def mean(self) -> Union[float, int]:
        return self.p
    
    def variance(self) -> Union[float, int]:
        return self.p * (1.0 - self.p)
