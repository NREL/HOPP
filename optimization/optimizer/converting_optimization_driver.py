from typing import (
    Callable,
    Optional,
    Tuple,
    )

from optimization.candidate_converter.candidate_converter import CandidateConverter
from optimization.driver.ask_tell_driver import AskTellDriver
from optimization.optimizer.ask_tell_optimizer import AskTellOptimizer


class ConvertingOptimizationDriver:
    """
    A composition model based driver for combining different:
        + optimizers
        + objective functions
        + drivers for running & parallelizing the generation-evaluation-update optimization cycle
    Each combination of objective function and optimizer will require a compatible set of initial conditions which
    should be provided by the prototype, and converted to and from a (usually vectorized) format acceptable to the
    optimizer by the given converter (dict to vector for example).
    """
    
    def __init__(self,
                 driver: AskTellDriver,
                 optimizer: AskTellOptimizer,
                 converter: CandidateConverter,
                 prototype: any,
                 objective: Callable[[any], float],
                 conformer: Optional[Callable[[any], any]] = None,
                 ) -> None:
        self.driver: AskTellDriver = driver
        self.optimizer: AskTellOptimizer = optimizer
        self.converter: CandidateConverter = converter
        self.prototype: any = prototype
        self.objective: Callable[[any], float] = objective
        
        if conformer is None:
            self.converted_objective: Callable[[any], Tuple[float, any]] = \
                lambda candidate: (self.objective(self.converter.convert_to(candidate)), candidate)
        else:
            def composite_objective(candidate) -> Tuple[float, any]:
                converted_candidate = self.converter.convert_to(candidate)
                conforming_candidate = conformer(converted_candidate)
                evaluation = self.objective(conforming_candidate)
                return_candidate = self.converter.convert_from(conforming_candidate)
                return evaluation, return_candidate
            
            self.converted_objective: Callable[[any], Tuple[float, any]] = composite_objective
        
        self.converter.setup(prototype)
        self.optimizer.setup(self.converter.convert_from(prototype))
    
    def step(self) -> bool:
        """
        Steps the optimizer through one iteration of generating candidates, evaluating them, and updating
        with their
        evaluations.
        :return: True if the optimizer reached a stopping point (via calling optimizer.stop())
        """
        return self.driver.step(self.optimizer, self.converted_objective)
    
    def run(self, max_iter: Optional[int] = None) -> int:
        """
        Runs the optimizer through max_iter iterations.
        May stop early if the optimizer returns True to a call to stop().
        :param optimizer: the optimizer to use
        :param objective: objective function for evaluating candidate solutions
        :param max_iter: maximum number of iterations, or None to use no maximum
        :return: number of iterations (calls to step()) applied
        """
        return self.driver.run(self.optimizer, self.converted_objective, max_iter)
    
    def best(self) -> any:
        """
        :return: the current best solution
        """
        return self.converter.convert_to(self.optimizer.best())
