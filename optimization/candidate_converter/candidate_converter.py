from abc import abstractmethod
from typing import (
    Generic,
    TypeVar,
    )

from optimization.data_logging.data_recorder import DataRecorder

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
    def convert_from(self, candidate: From) -> To:
        pass
    
    @abstractmethod
    def convert_to(self, candidate: To) -> From:
        pass
