import abc
import math
import random
from abc import ABC
from typing import (
    List,
    Optional,
    Tuple,
    Union,
    )

# matplotlib.use('tkagg')
import numpy as np

# sys.path.append('../examples/flatirons')
# import func_tools
from ..data_logging.data_recorder import DataRecorder
from .ask_tell_optimizer import AskTellOptimizer


class KFDCEM(AskTellOptimizer, ABC):
    """
    A prototype implementation of a Kalman-filter based decomposed cross-entropy method.
    """
    
    class Dimension:
        
        @abc.abstractmethod
        def update(self, samples: [float]) -> None:
            pass
        
        @abc.abstractmethod
        def step(self):
            pass
        
        @abc.abstractmethod
        def sample(self) -> Union[float, int]:
            pass
        
        @abc.abstractmethod
        def best(self) -> Union[float, int]:
            pass
    
    class KFDimension(Dimension):
        """
        Kalman Filter model dynamics:
        x' = ax + by + w
        z = hx + v
        
        here, a = 1, b = 0, h = 1 so:
        
        x' = x + w
        z = x + v
        """
        
        def __init__(self,
                     mu: float,
                     sigma: float,
                     sensor_noise: float,
                     dynamic_noise: float):
            self.mu: float = mu
            self.variance: float = sigma * sigma
            self.sensor_noise: float = sensor_noise
            self.dynamic_noise: float = dynamic_noise
        
        def step(self):
            """
            x' = a*x + b*u
            p' = a^2 * p + Q
            In this case:
                a = 1, b = 0
            """
            # print('kf step ', self.variance, self.dynamic_noise, self.variance + self.dynamic_noise)
            self.variance = self.variance + self.dynamic_noise
        
        def update(self, samples: [float]) -> None:
            """
            kalman gain: k = h*p / (p * h^2 + R)
            mean update: x' = x + k*(z - h*x)
            variance update: p' = p*(1 - h*k)
            
            h = 1 here, so:
            
            kalman gain: k = p / (p + R)
            mean update: x' = x + k*(z - x)
            variance update: p' = p*(1 - k)
            """
            if len(samples) > 1:
                sample_standard_deviation = np.std(samples, 0, ddof=1)
            else:
                sample_standard_deviation = 0.0
            sensor_noise = self.sensor_noise + sample_standard_deviation * sample_standard_deviation
            
            sample_mean = np.mean(samples, 0)
            kalman_gain = self.variance / (self.variance + sensor_noise)
            innovation = sample_mean - self.mu
            print('kfu: ', sample_standard_deviation, sensor_noise, innovation, self.variance,
                  self.variance * (1 - kalman_gain))
            self.mu = self.mu + kalman_gain * innovation
            self.variance = self.variance * (1 - kalman_gain)
        
        def sample(self) -> float:
            return random.gauss(self.mu, math.sqrt(self.variance))
        
        def best(self) -> Union[float, int]:
            return self.mu
    
    class KFParameterDimension(Dimension):
        
        def __init__(self,
                     mu: 'KFDCEM.KFDimension',
                     sigma: 'KFDCEM.KFDimension'):
            self.mu = mu
            self.sigma = sigma
        
        def step(self):
            self.mu.step()
            self.sigma.step()
        
        def update(self, samples: [float]) -> None:
            sample_mean = np.mean(samples, 0)
            sample_standard_deviation = np.std(samples, 0, ddof=1)
            self.mu.update([sample_mean])
            self.sigma.update([sample_standard_deviation])
        
        def sample(self) -> float:
            return random.gauss(self.mu.best(), self.sigma.best())
        
        def best(self) -> Union[float, int]:
            return self.mu.best()
    
    def __init__(self,
                 generation_size: int = 100,
                 selection_proportion: float = .33,
                 dimensions: Optional[List[Dimension]] = None
                 ):
        self._dimensions: [KFDCEM.Dimension] = [] if dimensions is None else dimensions
        self._generation_size: int = generation_size
        self._selection_proportion: float = selection_proportion
    
    def setup(self, dimension: [Dimension], recorder: DataRecorder) -> None:
        """
        Setup parameters given initial conditions of the candidate
        :param dimensions: list of search dimensions
        :param recorder: data recorder
        """
        self._dimensions = dimension
    
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
        if num is None:
            num = self._generation_size
        
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
        selection_size = math.ceil(self._selection_proportion * len(evaluations))
        del evaluations[selection_size:]
        
        for i, dimension in enumerate(self._dimensions):
            dimension.step()
            dimension.update([evaluation[1][i] for evaluation in evaluations])
    
    def best_solution(self) -> (Optional[float], any):
        """
        :return: the current best solution
        """
        return None, [dimension.best() for dimension in self._dimensions]
    
    def get_num_dimensions(self) -> int:
        """
        :return: number of dimensions being optimized over, or None if not implemented or applicable
        """
        return len(self._dimensions)
