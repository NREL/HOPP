import math
from abc import ABC
from typing import (
    List,
    Optional,
    Tuple,
    )

# matplotlib.use('tkagg')
import numpy as np

# sys.path.append('../examples/flatirons')
# import func_tools
from ..data_logging.data_recorder import DataRecorder
from .ask_tell_optimizer import AskTellOptimizer
# import shapely
from .dimension.dimension_info import DimensionInfo


class DCEMOptimizer(AskTellOptimizer, ABC):
    """
    A prototype implementation of the decomposed cross-entropy method.
    """
    
    def __init__(self,
                 generation_size: int = 100,
                 selection_proportion: float = .33,
                 dimensions: Optional[List[DimensionInfo]] = None,
                 ) -> None:
        self._recorder: Optional[DataRecorder] = None
        self._dimensions: [DimensionInfo] = [] if dimensions is None else dimensions
        self._generation_size: int = generation_size
        self._selection_proportion: float = selection_proportion
        self._best_candidate: Optional[Tuple[float, any]] = None  # self.mu()
    
    def setup(self, dimensions: [DimensionInfo], recorder: DataRecorder) -> None:
        """
        Setup parameters given initial conditions of the candidate
        :param dimensions: list of search dimensions
        :param recorder: data recorder
        """
        self._dimensions = dimensions
        self._recorder = recorder
        self._recorder.add_columns('generation', 'mean', 'variance')
    
    def stop(self) -> bool:
        """
        :return: True when the optimizer thinks it has reached a stopping point
        """
        return False
    
    def ask(self, num: Optional[int] = None) -> [any]:
        """
        :param num: the number of search points to return. If undefined, the optimizer will choose how many to return.
        :return: a list of search points generated by the optimizer
        """
        num = self._generation_size if num is None else num
        
        population = []
        for _ in range(num):
            # candidate = [0.0] * len(self.dimensions)
            candidate = np.empty(self.get_num_dimensions())
            for i, dimension in enumerate(self._dimensions):
                candidate[i] = dimension.sample()
            population.append(candidate)
        
        return population
    
    def tell(self, evaluations: [Tuple[float, any]]) -> None:
        """
        Updates the optimizer with the objective evaluations of a list of search points
        :param evaluations: a list of tuples of (evaluation, search point)
        """
        evaluations.sort(key=lambda evaluation: evaluation[0], reverse=True)
        if self._best_candidate is None or evaluations[0][0] > self._best_candidate[0]:
            self._best_candidate = evaluations[0]
        
        selection_size = math.ceil(self._selection_proportion * len(evaluations))
        del evaluations[selection_size:]
        
        for i, dimension in enumerate(self._dimensions):
            dimension.update([evaluation[1][i] for evaluation in evaluations])
        
        self._recorder.accumulate(evaluations, self.mean(), self.variance())
    
    def best_solution(self) -> (Optional[float], any):
        """
        :return: the current best solution
        """
        return (None, self.mean()) if self._best_candidate is None else self._best_candidate
    
    def get_num_candidates(self) -> int:
        return self._generation_size
    
    def get_num_dimensions(self) -> int:
        return len(self._dimensions)
    
    def mean(self) -> any:
        return np.fromiter((dimension.mean() for dimension in self._dimensions), float)
    
    def variance(self) -> any:
        return np.fromiter((dimension.variance() for dimension in self._dimensions), float)
