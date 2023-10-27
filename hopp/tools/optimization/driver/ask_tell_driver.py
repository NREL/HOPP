from abc import abstractmethod
from typing import (
    Callable,
    Optional,
    Tuple,
    )

from ..data_logging.data_recorder import DataRecorder
from ..optimizer.ask_tell_optimizer import AskTellOptimizer


class AskTellDriver:
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def step(self,
             optimizer: AskTellOptimizer,
             ) -> bool:
        """
        Steps the optimizer through one iteration of generating candidates, evaluating them, and updating with their
        evaluations.
        :param optimizer: the optimizer to use
        :return: True if the optimizer reached a stopping point (via calling optimizer.stop())
        """
        pass
    
    def run(self,
            optimizer: AskTellOptimizer,
            max_iter: Optional[int] = None,
            ) -> int:
        """
        Runs the optimizer through max_iter iterations.
        May stop early if the optimizer returns True to a call to stop().
        :param optimizer: the optimizer to use
        :param max_iter: maximum number of iterations, or None to use no maximum
        :return: number of iterations (calls to step()) applied
        """
        i: int = 0
        while self.step(optimizer) and (max_iter is None or max_iter > i):
            i += 1
        return i
    
    @abstractmethod
    def get_num_evaluations(self) -> int:
        pass
    
    @abstractmethod
    def get_num_iterations(self) -> int:
        pass
