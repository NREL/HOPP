from typing import (
    Callable,
    Optional,
    Tuple,
    )

from optimization.candidate_converter.candidate_converter import CandidateConverter
from optimization.data_logging.data_recorder import DataRecorder
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
    optimizer by the given converter (dict to vector, for example).
    """
    
    def __init__(self,
                 driver: AskTellDriver,
                 optimizer: AskTellOptimizer,
                 converter: CandidateConverter,
                 prototype: any,
                 objective: Callable[[any], float],
                 conformer: Optional[Callable[[any], any]] = None,
                 recorder: DataRecorder = DataRecorder(),
                 ) -> None:
        self.recorder: DataRecorder = recorder
        
        self._driver: AskTellDriver = driver
        self._optimizer: AskTellOptimizer = optimizer
        self._converter: CandidateConverter = converter
        self._prototype: any = prototype
        self._objective: Callable[[any], float] = objective
        self._conformer: Callable[[any], any] = lambda x: x
        if conformer is not None:
            self._conformer = conformer
        
        if conformer is None:
            objective = self._objective
            convert_to = converter.convert_to
            self.converted_objective: Callable[[any], Tuple[float, any]] = \
                lambda candidate: (self._objective(self._converter.convert_to(candidate)), candidate)
        else:
            def composite_objective(candidate) -> Tuple[float, any]:
                converted_candidate = self._converter.convert_to(candidate)
                conforming_candidate = conformer(converted_candidate)
                evaluation = self._objective(conforming_candidate)
                return_candidate = self._converter.convert_from(conforming_candidate)
                return evaluation, return_candidate
            
            self.converted_objective: Callable[[any], Tuple[float, any]] = composite_objective
        
        self._converter.setup(prototype, recorder)
        self._optimizer.setup(self._converter.convert_from(prototype), recorder)
        self._driver.setup(self.converted_objective, recorder)
        
        self.recorder.add_columns('iteration', 'num_evaluations', 'score', 'best_solution')
        self.recorder.set_schema()
    
    def __del__(self):
        # noinspection PyBroadException
        try:
            self.close()
        except:
            pass
    
    def step(self) -> bool:
        """
        Steps the optimizer through one iteration of generating candidates, evaluating them, and updating
        with their evaluations.
        :return: True if the optimizer reached a stopping point (via calling optimizer.stop())
        """
        result = self._driver.step(self._optimizer)
        self.recorder.accumulate(self._driver.get_num_iterations(), self._driver.get_num_evaluations(),
                                 *self.best_solution())
        self.recorder.store()
        return result
    
    def run(self, max_iter: Optional[int] = None) -> int:
        """
        Runs the optimizer through max_iter iterations.
        May stop early if the optimizer returns True to a call to stop().
        :param optimizer: the optimizer to use
        :param objective: objective function for evaluating candidate solutions
        :param max_iter: maximum number of iterations, or None to use no maximum
        :return: number of iterations (calls to step()) applied
        """
        return self._driver.run(self._optimizer, max_iter)
    
    def best_solution(self) -> Tuple[Optional[float], any]:
        """
        :return: the current best solution
        """
        score, solution = self._optimizer.best_solution()
        converted_solution = self._converter.convert_to(solution)
        conforming_solution = self._conformer(converted_solution)
        return score, conforming_solution
    
    def close(self) -> None:
        self.recorder.close()
