from typing import (
    Optional,
    Callable,
    Tuple
)

from .data_logging.data_recorder import DataRecorder
from .data_logging.null_data_recorder import NullDataRecorder

from .optimization_problem import OptimizationProblem
from .driver.ask_tell_parallel_driver import AskTellDriver, AskTellParallelDriver
from .driver.ask_tell_serial_driver import AskTellSerialDriver
from .optimizer.CEM_optimizer import CEMOptimizer
from .optimizer.CMA_ES_optimizer import CMAESOptimizer
from .optimizer.GA_optimizer import GAOptimizer
from .optimizer.SPSA_optimizer import (
    SPSADimensionInfo,
    SPSAOptimizer,
)
from .optimizer.ask_tell_optimizer import AskTellOptimizer
from .optimizer.dimension.gaussian_dimension import Gaussian
from .optimizer.stationary_optimizer import StationaryOptimizer


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


class OptimizationDriver(ConvertingOptimizationDriver):
    """
    Creates a ConvertingOptimizationDriver with the given optimizer method and initial conditions from the
    problem's prior
    """

    def __init__(self,
                 problem: OptimizationProblem,
                 method: str,
                 recorder: DataRecorder,
                 nprocs: Optional[int] = None,
                 **kwargs
                 ) -> None:
        self.problem: OptimizationProblem = problem

        optimizer: AskTellOptimizer
        prior: object
        if method == 'GA':
            optimizer, prior = self.genetic_algorithm(**kwargs)
        elif method == 'CEM':
            optimizer, prior = self.cross_entropy(**kwargs)
        elif method == 'CMA-ES':
            optimizer, prior = self.CMA_ES(**kwargs)
        elif method == 'SPSA':
            optimizer, prior = self.genetic_algorithm(**kwargs)
        elif method == 'Stationary':
            optimizer, prior = self.stationary_optimizer(**kwargs)
        else:
            raise ValueError('Unknown optimizer: "' + method + '"')

        driver = AskTellSerialDriver() if nprocs == 1 else AskTellParallelDriver(nprocs)
        super().__init__(
            driver,
            optimizer,
            # ObjectConverter(),
            prior,
            conformer=self.problem.conform_candidate_and_get_penalty,
            objective=self.problem.objective,
            recorder=recorder,
        )

    @staticmethod
    def check_kwargs(inputs: tuple,
                     **kwargs
                     ) -> None:
        """
        Checks that the inputs are in **kwargs
        :param inputs: tuple of strings
        :param kwargs: keyword arguments
        """
        for i in inputs:
            if i not in kwargs:
                raise ValueError(i + " argument required")

    def create_prior(self,
                     dimension_type: type,
                     conf_prior_params: dict = None
                     ):
        """
        Create a prior candidate with information about each dimension's distribution
        :param dimension_type: the distribution type of each dimension of the prior
        :param conf_prior_params: a nested dictionary containing key: value pairs where the key is the dimension name
                    and the value is a dictionary of the distributions' parameters that should replace the default ones
                    that are stored in the problem's get_prior_params function. Parameters that are not attributes of
                    dimension_type are not used

        Example:
            config_prior_params = { "border_spacing": {"mu": 3, "beta": 4}}
                This will replace the prior's border_spacing distribution's mu parameter to be 3, but beta is ignored
        :return:
        """
        prior_params = self.problem.get_prior_params(dimension_type)

        if conf_prior_params:
            for conf_dimension in conf_prior_params.keys():
                if conf_dimension in prior_params.keys():
                    prior_params[conf_dimension].update({k: v for k, v in conf_prior_params[conf_dimension].items()
                                                         if k in prior_params[conf_dimension].keys()})

        # for conf_dimension, v in prior_params.items():
        #     prior_params[conf_dimension] = dimension_type(**v)

        # self.problem.convert_to_candidate(prior_params)
        return list(dimension_type(**v) for _, v in prior_params.items())

    def cross_entropy(self,
                      **kwargs
                      ) -> (AskTellOptimizer, object):
        """
        Create a cross entropy optimizer using a Gaussian sampling distribution with required keyword arguments:
            prior_scale: float = scaling factor
            generation_size: int = number of candidates to generate
            selection_proportion: float = proportion of candidates
        :param kwargs: keyword arguments
        :return: optimizer, prior
        """
        args = ("prior_scale", "generation_size", "selection_proportion")
        self.check_kwargs(args, **kwargs)

        prior = self.create_prior(Gaussian, kwargs.get("prior_params"))
        optimizer = CEMOptimizer(kwargs.get("generation_size"), kwargs.get("selection_proportion"))
        return optimizer, prior

    def CMA_ES(self,
               **kwargs
               ) -> (AskTellOptimizer, object):
        """
        Create a cross entropy optimizer using a Gaussian sampling distribution with required keyword arguments:
            prior_scale: float = scaling factor
            generation_size: int = number of candidates to generate
            selection_proportion: float = proportion of candidates
        :param kwargs: keyword arguments
        :return: optimizer, prior
        """
        args = ("prior_scale", "generation_size", "selection_proportion")
        self.check_kwargs(args, **kwargs)

        prior = self.create_prior(Gaussian, kwargs.get("prior_params"))
        optimizer = CMAESOptimizer(kwargs.get("generation_size"), kwargs.get("selection_proportion"))
        return optimizer, prior

    def genetic_algorithm(self,
                          **kwargs
                          ) -> (AskTellOptimizer, object):
        """
        Create a genetic algorithm optimizer using a Gaussian sampling distribution with required keyword arguments:
            prior_scale: float = scaling factor
            generation_size: int = number of candidates to generate
            selection_proportion: float = proportion of candidates
        :param kwargs: keyword arguments
        :return: optimizer, prior
        """
        args = ("prior_scale", "generation_size", "selection_proportion")
        self.check_kwargs(args, **kwargs)

        prior = self.create_prior(Gaussian, kwargs.get("prior_params"))
        optimizer = GAOptimizer(kwargs.get("generation_size"), kwargs.get("selection_proportion"))
        return optimizer, prior

    def simultaneous_perturbation_stochastic_approximation(self,
                                                           **kwargs
                                                           ) -> (AskTellOptimizer, object):
        """
        Create a SPSA optimizer using a SPSA sampling distribution with required keyword arguments:
            prior_scale: float = scaling factor
            generation_size: int = number of candidates to generate
        :param kwargs: keyword arguments
        :return: optimizer, prior
        """
        args = ("prior_scale", "generation_size")
        self.check_kwargs(args, **kwargs)

        prior = self.create_prior(SPSADimensionInfo, kwargs.get("prior_params"))
        optimizer = SPSAOptimizer(.2, num_estimates=kwargs.get("generation_size"))
        return optimizer, prior

    def stationary_optimizer(self,
                             **kwargs
                             ) -> (AskTellOptimizer, object):
        """
        Create a stationary optimizer using a Gaussian sampling distribution with required keyword arguments:
            prior_scale: float = scaling factor
            generation_size: int = number of candidates to generate
            selection_proportion: float = proportion of candidates
        :param kwargs: keyword arguments
        :return: optimizer, prior
        """
        args = ("prior_scale", "generation_size", "selection_proportion")
        self.check_kwargs(args, **kwargs)

        prior = self.create_prior(Gaussian, kwargs.get("prior_params"))
        optimizer = StationaryOptimizer(kwargs.get("generation_size"), kwargs.get("selection_proportion"))
        return optimizer, prior
