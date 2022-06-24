from multiprocessing import Pool, cpu_count
from typing import (
    Callable,
    Tuple,
    )

from ..data_logging.data_recorder import DataRecorder
from ..driver.ask_tell_driver import AskTellDriver
from ..optimizer.ask_tell_optimizer import AskTellOptimizer
from .ask_tell_parallel_driver_fns import *


class AskTellParallelDriver(AskTellDriver):
    
    def __init__(self,
                 nprocs: int = cpu_count()):
        self._num_evaluations: int = 0
        self._num_iterations: int = 0
        self._nprocs = nprocs
        self._pool = None
        
        # self.evaluations = []
    
    def __getstate__(self):
        """
        This prevents the pool from being pickled when using the pool...
        """
        self_dict = self.__dict__.copy()
        if 'pool' in self_dict:
            del self_dict['pool']
        return self_dict
    
    def __setstate__(self, state):
        """
        This prevents the pool from being pickled when using the pool...
        """
        self.__dict__.update(state)
    
    def __del__(self):
        """
        This prevents the pool from being pickled when using the pool...
        """
        if hasattr(self, 'pool') and self._pool is not None:
            self._pool.join() 
            self._pool.close() 
            self._pool = None 
    
    def setup(
            self,
            objective: Callable[[any], Tuple[float, float, any]],
            recorder: DataRecorder,
            ) -> None:
        """
        Must be called before calling step() or run().
        Sets the objective function for this driver and the data recorder.
        :param objective: objective function for evaluating candidate solutions
        :param recorder: data recorder
        :return:
        """
        self._pool = Pool(
            initializer=make_initializer(objective),
            processes=self._nprocs)
    
    def step(self,
             optimizer: AskTellOptimizer,
             ) -> bool:
        """
        Steps the optimizer through one iteration of generating candidates, evaluating them, and updating with their
        evaluations.
        :param optimizer: the optimizer to use
        :return: True if the optimizer reached a stopping point (via calling optimizer.stop())
        """
        # print('step()')
        num_candidates = optimizer.get_num_candidates()
        candidates = optimizer.ask(num_candidates)
        evaluations = self._pool.map(evaluate, candidates)
        num_candidates = len(evaluations)
        # print('telling')
        # self.evaluations = list(evaluations)
        optimizer.tell(evaluations)
        
        self._num_evaluations += num_candidates
        self._num_iterations += 1
        # print('done')
        return optimizer.stop()
    
    def get_num_evaluations(self) -> int:
        return self._num_evaluations
    
    def get_num_iterations(self) -> int:
        return self._num_iterations
