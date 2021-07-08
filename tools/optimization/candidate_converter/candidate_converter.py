from abc import abstractmethod
from typing import (
    Generic,
    TypeVar,
    )

from tools.optimization.data_logging.data_recorder import DataRecorder

From = TypeVar('From')
To = TypeVar('To')


class CandidateConverter(Generic[From, To]):
    """
    Converts candidates between two representations
    """
    
    @abstractmethod
    def setup(self, prototype: From, recorder: DataRecorder) -> None:
        pass
    
    @abstractmethod
    def convert_from(self, candidate: From) -> list:
        """
        Converts candidate representation into a list of values
        """
        pass
    
    @abstractmethod
    def convert_to(self, candidate: To) -> From:
        """
        Converts from list of values to another representation
        """
        pass
