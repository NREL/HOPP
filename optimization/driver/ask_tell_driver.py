from abc import abstractmethod
from typing import (
    Callable,
    Optional,
    Tuple,
    )

from optimization.optimizer.ask_tell_optimizer import AskTellOptimizer


class AskTellDriver:
    
    @abstractmethod
    def step(self,
             optimizer: AskTellOptimizer,
             objective: Callable[[any], Tuple[any, float]],
             ) -> bool:
        """
        Steps the optimizer through one iteration of generating candidates, evaluating them, and updating with their
        evaluations.
        :return: True if the optimizer reached a stopping point (via calling optimizer.stop())
        """
        pass
    
    def run(self,
            optimizer: AskTellOptimizer,
            objective: Callable[[any], Tuple[any, float]],
            max_iter: Optional[int] = None,
            ) -> int:
        """
        Runs the optimizer through max_iter iterations.
        May stop early if the optimizer returns True to a call to stop().
        :param optimizer: the optimizer to use
        :param objective: objective function for evaluating candidate solutions
        :param max_iter: maximum number of iterations, or None to use no maximum
        :return: number of iterations (calls to step()) applied
        """
        i: int = 0
        while self.step(optimizer, objective) and (max_iter is None or max_iter > i):
            i += 1
        return i
    
    @abstractmethod
    def get_num_evaluations(self) -> int:
        pass

    @abstractmethod
    def get_num_iterations(self) -> int:
        pass
