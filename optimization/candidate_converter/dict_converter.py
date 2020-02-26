from typing import Optional

from optimization.candidate_converter.candidate_converter import CandidateConverter
from optimization.data_logging.data_recorder import DataRecorder


class DictConverter(CandidateConverter[dict, list]):
    """
    Converts between maps or and numpy arrays.
    """
    
    def __init__(self, prototype: Optional[dict] = None):
        self._map: [any] = []
        if prototype is not None:
            self.setup(prototype)
    
    def setup(self, prototype: dict, recorder: DataRecorder) -> None:
        self._map = list([key for key in sorted(prototype.keys())])
    
    def convert_from(self, candidate: dict) -> list:
        result = [None] * len(self._map)
        for index, key in enumerate(self._map):
            result[index] = candidate[key]
        return result
    
    def convert_to(self, candidate: list) -> dict:
        result = {}
        for index, key in enumerate(self._map):
            result[key] = candidate[index]
        return result
