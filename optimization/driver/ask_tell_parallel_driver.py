from typing import (
    Callable,
    Tuple,
    )

import multiprocessing

from optimization.driver.ask_tell_driver import AskTellDriver
from optimization.optimizer.ask_tell_optimizer import AskTellOptimizer
from .ask_tell_parallel_driver_fns import *
from ..data_logging.data_recorder import DataRecorder


class AskTellParallelDriver(AskTellDriver):
    
    def __init__(self):
        self._num_evaluations: int = 0
        self._num_iterations: int = 0
        self._pool = None
    
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
            self._pool.close()
    
    def setup(
            self,
            objective: Callable[[any], Tuple[float, any]],
            recorder: DataRecorder,
            ) -> None:
        self._pool = multiprocessing.Pool(
            initializer=make_initializer(objective),
            processes=multiprocessing.cpu_count())
    
    def step(self,
             optimizer: AskTellOptimizer,
             ) -> bool:
        # print('step()')
        num_candidates = optimizer.get_num_candidates()
        candidates = optimizer.ask(num_candidates)
        evaluations = self._pool.map(evaluate, candidates)
        # print('telling')
        optimizer.tell(evaluations)
        
        self._num_evaluations += len(evaluations)
        self._num_iterations += 1
        # print('done')
        return optimizer.stop()
    
    def get_num_evaluations(self) -> int:
        return self._num_evaluations
    
    def get_num_iterations(self) -> int:
        return self._num_iterations
