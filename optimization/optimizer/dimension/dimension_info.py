from abc import abstractmethod
from typing import Union


class DimensionInfo:
    
    @abstractmethod
    def update(self, samples: [any]) -> None:
        pass
    
    @abstractmethod
    def sample(self) -> Union[float, int]:
        pass
    
    @abstractmethod
    def best(self) -> Union[float, int]:
        pass
    
    @abstractmethod
    def mean(self) -> Union[float, int]:
        pass
    
    @abstractmethod
    def variance(self) -> Union[float, int]:
        pass
