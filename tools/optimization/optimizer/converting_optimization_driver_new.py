from typing import (
    Callable,
    Optional,
    Tuple,
)

from ..data_logging.data_recorder import DataRecorder
from ..data_logging.null_data_recorder import NullDataRecorder
from ..driver.ask_tell_driver import AskTellDriver
from ..optimizer.ask_tell_optimizer import AskTellOptimizer


class ConvertingOptimizationDriver:
    """
    A composition model based driver for combining different:
        + optimizers
        + objective functions
        + drivers for running & parallelizing the generation-evaluation-update optimization cycle
    Each combination of objective function and optimizer will require a compatible set of initial conditions which
    should be provided by the prototype
    """

    def __init__(self,
                 driver: AskTellDriver,
                 optimizer: AskTellOptimizer,
                 prototype: any,
                 conformer: Optional[Callable[[any], Tuple[object, any]]],
                 objective: Callable[[any], Tuple[float, float, any]],
                 recorder: DataRecorder = NullDataRecorder(),
                 ) -> None:
        self.recorder: DataRecorder = recorder

        self._driver: AskTellDriver = driver
        self._optimizer: AskTellOptimizer = optimizer
        self._prototype: any = prototype
        self._conformer: Optional[Callable[[any], Tuple[object, any]]] = conformer
        self._objective: Callable[[any], Tuple[float, float, any]] = objective

        self._optimizer.setup(self._prototype, recorder)
        self._driver.setup(self._objective, recorder)

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
        return result[0], result[1], self._conformer(result[2])[0]

    def central_solution(self) -> (Optional[float], Optional[float], any):
        """
        :return: the mean search position, or other representative solution
        """
        score, evaluation, solution = self._optimizer.central_solution()
        return score, evaluation, self._conformer(solution)[0]

    def close(self) -> None:
        self.recorder.close()
