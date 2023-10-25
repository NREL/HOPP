from typing import (
    Callable,
    Tuple,
    )

from ..data_logging.data_recorder import DataRecorder
from ..driver.ask_tell_driver import AskTellDriver
from ..optimizer.ask_tell_optimizer import AskTellOptimizer


class AskTellSerialDriver(AskTellDriver):
    
    def __init__(self):
        self._num_evaluations: int = 0
        self._num_iterations: int = 0
        self._objective = None
        # self.evaluations = []
    
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
        self._objective = objective
    
    def step(self,
             optimizer: AskTellOptimizer,
             ) -> bool:
        """
        Steps the optimizer through one iteration of generating candidates, evaluating them, and updating with their
        evaluations.
        :param optimizer: the optimizer to use
        :return: True if the optimizer reached a stopping point (via calling optimizer.stop())
        """
        candidates: [any] = optimizer.ask()
        evaluations: [Tuple[float, float, any]] = [self._objective(candidate) for candidate in candidates]
        # self.evaluations = list(evaluations)
        optimizer.tell(evaluations)
        self._num_evaluations += len(evaluations)
        self._num_iterations += 1
        return optimizer.stop()
    
    def get_num_evaluations(self) -> int:
        return self._num_evaluations
    
    def get_num_iterations(self) -> int:
        return self._num_iterations
