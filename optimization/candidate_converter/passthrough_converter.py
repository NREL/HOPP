from typing import (
    Optional,
    TypeVar,
    )

from optimization.candidate_converter.candidate_converter import CandidateConverter
from optimization.data_logging.data_recorder import DataRecorder

To = TypeVar('To')


class PassthroughConverter(CandidateConverter[To, To]):
    """
    Passthrough converter
    """
    
    def __init__(self, prototype: Optional[To] = None):
        pass
    
    def setup(self, prototype: To, recorder: DataRecorder) -> None:
        pass
    
    def convert_from(self, candidate: To) -> To:
        return candidate
    
    def convert_to(self, candidate: To) -> To:
        return candidate
