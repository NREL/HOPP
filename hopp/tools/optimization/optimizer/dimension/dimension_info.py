from abc import abstractmethod
from typing import Union


class DimensionInfo:
    """
    Interface for probability distribution functions
    """
    
    @abstractmethod
    def update(self, samples: [any]) -> None:
        """
        Update parameters of the pdf with new samples
        :param samples: list of best candidates from optimizer
        """
        pass
    
    @abstractmethod
    def sample(self) -> Union[float, int]:
        """
        :return: Sample from the pdf
        """
        pass
    
    @abstractmethod
    def best(self) -> Union[float, int]:
        """
        :return: Most likely sample
        """
        pass
    
    @abstractmethod
    def mean(self) -> Union[float, int]:
        pass
    
    @abstractmethod
    def variance(self) -> Union[float, int]:
        pass
