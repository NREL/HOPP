from typing import (
    Callable,
    Tuple,
    )

from optimization.data_logging.data_recorder import DataRecorder
from optimization.driver.ask_tell_driver import AskTellDriver
from optimization.optimizer.ask_tell_optimizer import AskTellOptimizer


class AskTellSerialDriver(AskTellDriver):
    
    def __init__(self):
        self._num_evaluations: int = 0
        self._num_iterations: int = 0
        self._objective = None
    
    def setup(
            self,
            objective: Callable[[any], Tuple[float, any]],
            recorder: DataRecorder,
            ) -> None:
        self._objective = objective
    
    def step(self,
             optimizer: AskTellOptimizer,
             ) -> bool:
        candidates: [any] = optimizer.ask()
        evaluations: [Tuple[float, any]] = [self._objective(candidate) for candidate in candidates]
        optimizer.tell(evaluations)
        self._num_evaluations += len(evaluations)
        self._num_iterations += 1
        return optimizer.stop()
    
    def get_num_evaluations(self) -> int:
        return self._num_evaluations
    
    def get_num_iterations(self) -> int:
        return self._num_iterations
