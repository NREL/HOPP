from typing import Optional

import numpy as np

from optimization.candidate_converter.candidate_converter import CandidateConverter


class DictConverter(CandidateConverter[dict, list]):
    """
    Converts between maps or and numpy arrays.
    """
    
    def __init__(self, prototype: Optional[dict] = None):
        self.map: [any] = []
        if prototype is not None:
            self.setup(prototype)
    
    def setup(self, prototype: dict) -> None:
        self.map = list([key for key in sorted(prototype.keys())])
    
    def convert_from(self, candidate: dict) -> list:
        result = [None] * len(self.map)
        for index, key in enumerate(self.map):
            result[index] = candidate[key]
        return result
    
    def convert_to(self, candidate: list) -> dict:
        result = {}
        for index, key in enumerate(self.map):
            result[key] = candidate[index]
        return result
