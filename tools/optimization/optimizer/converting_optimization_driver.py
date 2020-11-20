from typing import (
    Callable,
    Optional,
    Tuple,
    )

from ..data_logging.data_recorder import DataRecorder
from ..data_logging.null_data_recorder import NullDataRecorder
from ..candidate_converter.candidate_converter import CandidateConverter
from ..driver.ask_tell_driver import AskTellDriver
from ..optimizer.ask_tell_optimizer import AskTellOptimizer


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
                 objective: Callable[[any], Tuple[float, float]],
                 conformer: Optional[Callable[[any], any]] = None,
                 recorder: DataRecorder = NullDataRecorder(),
                 ) -> None:
        self.recorder: DataRecorder = recorder
        
        self._driver: AskTellDriver = driver
        self._optimizer: AskTellOptimizer = optimizer
        self._converter: CandidateConverter = converter
        self._prototype: any = prototype
        self._objective: Callable[[any], Tuple[float, float]] = objective
        self._conformer: Callable[[any], (any, float)] = \
            conformer if conformer is not None else lambda candidate: (candidate, 0.0)
        
        def converted_objective(candidate) -> Tuple[float, float, any]:
            """
            Composes the actions of converting candidate representation, constraint conformation,
            objective evaluation and re-conversion to candidate values
            :param candidate: list of values
            :return: objective performance, list of values of conforming candidate
            """
            converted_candidate = self._converter.convert_to(candidate)
            conforming_candidate, conformation_penalty, conformation_penalty_sq = self._conformer(converted_candidate)
            score, evaluation = self._objective(conforming_candidate)
            score -= conformation_penalty
            # print('score', score, 'evaluation', evaluation)
            return score, evaluation, candidate
        
        self.converted_objective: Callable[[any], Tuple[float, float, any]] = converted_objective
        
        self._converter.setup(prototype, recorder)
        self._optimizer.setup(self._converter.convert_from(prototype), recorder)
        self._driver.setup(self.converted_objective, recorder)
        
        self.recorder.add_columns('iteration', 'num_evaluations', 'best_score', 'best_evaluation', 'best_solution')
        self.recorder.set_schema()
    
    def __del__(self):
        # noinspection PyBroadException
        try:
            self.close()
        except:
            pass
    
    def num_evaluations(self) -> float:
        return self._driver.get_num_evaluations()
    
    def num_iterations(self) -> float:
        return self._driver.get_num_iterations()
    
    def step(self) -> bool:
        """
        Steps the optimizer through one iteration of generating candidates, evaluating them, and updating
        with their evaluations.
        :return: True if the optimizer reached a stopping point (via calling optimizer.stop())
        """
        result = self._driver.step(self._optimizer)
        self.recorder.accumulate(self._driver.get_num_iterations(),
                                 self._driver.get_num_evaluations(),
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
    
    def best_solution(self) -> [Tuple[float, float, any]]:
        """
        :return: the current best solution
        """
        result = self._optimizer.best_solution()
        if result is None:
            return None, None, None
        return result[0], result[1], self._convert_and_conform(result[2])
    
    def central_solution(self) -> (Optional[float], Optional[float], any):
        """
        :return: the mean search position, or other representative solution
        """
        score, evaluation, solution = self._optimizer.central_solution()
        return score, evaluation, self._convert_and_conform(solution)
    
    def _convert_and_conform(self, solution):
        return self._conformer(self._converter.convert_to(solution))[0]
    
    def close(self) -> None:
        self.recorder.close()
    
    @property
    def converter(self):
        return self._converter
