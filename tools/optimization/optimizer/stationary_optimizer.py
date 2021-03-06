from typing import (
    List,
    Optional,
    Tuple,
    )

# matplotlib.use('tkagg')
import numpy as np

# sys.path.append('../examples/flatirons')
from ..data_logging.data_recorder import DataRecorder
from .ask_tell_optimizer import AskTellOptimizer
from .dimension.gaussian_dimension import DimensionInfo, Gaussian


class StationaryOptimizer(AskTellOptimizer):
    """
    A random search from a stationary distribution
    """
    
    def __init__(self,
                 generation_size: int = 100,
                 selection_proportion: float = .33,
                 dimensions: Optional[List[DimensionInfo]] = None,
                 ) -> None:
        self._recorder: Optional[DataRecorder] = None
        self._generation_size: int = generation_size
        self._selection_proportion: float = selection_proportion
        self._best_candidate: Optional[Tuple[float, float, any]] = None
        
        self._mean = np.empty(0)
        self._covariance = np.empty(0)
        self._variance_scales = self._mean
        
        if dimensions is not None:
            self.setup(dimensions)
    
    def setup(self, dimensions: [Gaussian], recorder: DataRecorder) -> None:
        """
        Setup parameters given initial conditions of the candidate
        :param dimensions: list of search dimensions
        :param recorder: data recorder
        """
        num_dimensions = len(dimensions)
        self._mean = np.fromiter((d.mean() for d in dimensions), float)
        self._covariance = np.zeros((num_dimensions, num_dimensions))
        variance_scales = np.empty(num_dimensions)
        for i, d in enumerate(dimensions):
            self._covariance[i, i] = d.variance()
            variance_scales[i] = d.scale
        self._variance_scales = np.diag(variance_scales)
        self._recorder = recorder
        self._recorder.add_columns('generation', 'mean', 'variance', 'covariance')
    
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
        return [np.random.multivariate_normal(self._mean, self._covariance) for _ in range(num)]
    
    def tell(self, evaluations: [Tuple[float, float, any]]) -> None:
        """
        Updates the optimizer with the objective evaluations of a list of search points
        :param evaluations: a list of tuples of (evaluation, search point)
        """

        def best_key(e):
            return e[1], e[0]
        
        best = max(evaluations, key=best_key)
        self._best_candidate = best if self._best_candidate is None else max((self._best_candidate, best), key=best_key)
        self._recorder.accumulate(evaluations, self.mean(), self.variance(), self._covariance)
    
    def best_solution(self) -> Optional[Tuple[float, float, any]]:
        """
        :return: the current best solution
        """
        return self._best_candidate
    
    def central_solution(self) -> (Optional[float], Optional[float], any):
        return None, None, self._mean
    
    def get_num_candidates(self) -> int:
        return self._generation_size
    
    def get_num_dimensions(self) -> int:
        return self._mean.size
    
    def mean(self) -> any:
        return self._mean
    
    def variance(self) -> any:
        return self._covariance.diagonal()
