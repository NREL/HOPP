from __future__ import annotations
from typing import Optional, Union, Callable

from tools.optimization.data_logging.data_recorder import NullRecordLogger, DataRecorder
from tools.optimization.problem_parametrization import ProblemParametrization
from tools.optimization.candidate_converter.object_converter import CandidateConverter, ObjectConverter
from tools.optimization.driver.ask_tell_parallel_driver import AskTellDriver, AskTellParallelDriver
from tools.optimization.driver.ask_tell_serial_driver import AskTellSerialDriver
from tools.optimization.optimizer.CEM_optimizer import CEMOptimizer
from tools.optimization.optimizer.CMA_ES_optimizer import CMAESOptimizer
from tools.optimization.optimizer.GA_optimizer import GAOptimizer
from tools.optimization.optimizer.SPSA_optimizer import (
    SPSADimensionInfo,
    SPSAOptimizer,
    )
from tools.optimization.optimizer.ask_tell_optimizer import AskTellOptimizer
from tools.optimization.optimizer.dimension.gaussian_dimension import Gaussian
from tools.optimization.optimizer.stationary_optimizer import StationaryOptimizer
from examples.optimization.layout_opt.parametrized_optimization_problem import ParametrizedOptimizationProblem


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
                 objective: Callable[[any], tuple[float, float]],
                 conformer: Optional[Callable[[any], any]] = None,
                 recorder: DataRecorder = NullRecordLogger(),
                 ) -> None:
        self.recorder: DataRecorder = recorder

        self._driver: AskTellDriver = driver
        self._optimizer: AskTellOptimizer = optimizer
        self._converter: CandidateConverter = converter
        self._prototype: any = prototype
        self._objective: Callable[[any], tuple[float, float]] = objective
        self._conformer: Callable[[any], (any, float)] = \
            conformer if conformer is not None else lambda candidate: (candidate, 0.0)

        def converted_objective(candidate) -> tuple[float, float, any]:
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

        self.converted_objective: Callable[[any], tuple[float, float, any]] = converted_objective

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

    def run(self, max_iter:Optional[int] = None) -> int:
        """
        Runs the optimizer through max_iter iterations.
        May stop early if the optimizer returns True to a call to stop().
        :param optimizer: the optimizer to use
        :param objective: objective function for evaluating candidate solutions
        :param max_iter: maximum number of iterations, or None to use no maximum
        :return: number of iterations (calls to step()) applied
        """
        return self._driver.run(self._optimizer, max_iter)

    def best_solution(self) -> tuple[tuple[float, float, any]]:
        """
        :return: the current best solution
        """
        result = self._optimizer.best_solution()
        if result is None:
            return None, None, None
        return result[0], result[1], self._convert_and_conform(result[2])

    def central_solution(self) -> tuple[Optional[float], Optional[float], any]:
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


class ParametrizedOptimizationDriver(ConvertingOptimizationDriver):
    """
    Creates a ConvertingOptimizationDriver with the given optimizer method and initial conditions from the
    problem's prior
    """
    
    def __init__(self,
                 problem: Union[ParametrizedOptimizationProblem, ProblemParametrization],
                 method: str,
                 recorder: DataRecorder,
                 nprocs: Optional[int] = None,
                 **kwargs
                 ) -> None:
        self.problem: Union[ParametrizedOptimizationProblem, ProblemParametrization] = problem
        
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
            ObjectConverter(),
            prior,
            lambda candidate: problem.objective(candidate),
            conformer=lambda candidate: self.problem.make_conforming_candidate_and_get_penalty(candidate),
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
        prior = self.problem.candidate_type()
        prior_params = self.problem.get_prior_params(dimension_type)
        
        if conf_prior_params:
            for conf_dimension in conf_prior_params.keys():
                if conf_dimension in prior_params.keys():
                    prior_params[conf_dimension].update({k: v for k, v in conf_prior_params[conf_dimension].items()
                                                         if k in prior_params[conf_dimension].keys()})
        
        for conf_dimension, v in prior_params.items():
            prior[conf_dimension] = dimension_type(**v)
        return prior
    
    def cross_entropy(self,
                      **kwargs
                      ) -> tuple[AskTellOptimizer, object]:
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
               ) -> tuple[AskTellOptimizer, object]:
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
                          ) -> tuple[AskTellOptimizer, object]:
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
                                                           ) -> tuple[AskTellOptimizer, object]:
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
               ) -> tuple[AskTellOptimizer, object]:
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
