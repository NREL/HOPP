from typing import (
    Callable,
    Tuple,
    Optional,
    )

import joblib

from optimization.driver.ask_tell_driver import AskTellDriver
from optimization.optimizer.ask_tell_optimizer import AskTellOptimizer


class AskTellSerialDriver(AskTellDriver):
    
    def __init__(self):
        self.num_evaluations: int = 0
        self.num_iterations: int = 0
    
    def step(self,
             optimizer: AskTellOptimizer,
             objective: Callable[[any], Tuple[float, any]],
             ) -> bool:
        candidates: [any] = optimizer.ask()
        evaluations: [Tuple[float, any]] = [objective(candidate) for candidate in candidates]
        optimizer.tell(evaluations)
        self.num_evaluations += len(evaluations)
        self.num_iterations += 1
        return optimizer.stop()
    
    def get_num_evaluations(self) -> int:
        return self.num_evaluations
    
    def get_num_iterations(self) -> int:
        return self.num_iterations
