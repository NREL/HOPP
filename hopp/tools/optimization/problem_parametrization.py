from abc import abstractmethod
from typing import (
    Tuple,
    Type,
    )


class ProblemParametrization:
    """
    Interface between optimization drivers and the simulations being optimized that uses a parametrization of the
    original problem.
    """
    
    def __init__(self,
                 inner_problem,
                 inner_problem_type: Type,
                 candidate_type: Type
                 ) -> None:
        self.inner_problem: inner_problem_type = inner_problem
        self.candidate_type: Type = candidate_type
    
    @abstractmethod
    def get_prior_params(self,
                         distribution_type: Type
                         ) -> dict:
        """
        Returns the parameters for each parameter's distribution for a given distribution type
        :param distribution_type: str identifier ("Gaussian", "Bernoulli", ...)
        :return: dictionary of parameters
        """
        pass
    
    @abstractmethod
    def make_inner_candidate_from_parameters(self,
                                             candidate: object,
                                             ) -> Tuple[float, any]:
        """
        Transforms parametrized into inner problem candidate
        :param candidate:
        :return: a penalty and an inner candidate
        """
        pass
    
    @abstractmethod
    def make_conforming_candidate_and_get_penalty(self,
                                                  candidate: object
                                                  ) -> tuple:
        """
        Modifies a candidate's parameters so that it falls within range
        :param candidate: optimization candidate
        :return: conforming candidate, parameter error values
        """
        pass
    
    def objective(self,
                  parameters: object
                  ) -> (float, float):
        """
        Returns inner problem performance of parametrized candidate
        :param parameters: parametrized candidate
        :return: performance
        """
        penalty, inner_candidate = self.make_inner_candidate_from_parameters(parameters)
        if isinstance(inner_candidate, list) or isinstance(inner_candidate, tuple):
            inner_candidate = inner_candidate[0]
        # print('ProblemParametrization::objective() inner_candidate', inner_candidate)
        evaluation = self.inner_problem.objective(inner_candidate)
        score = evaluation - penalty
        # print('ProblemParametrization::objective()', evaluation, penalty, evaluation - penalty)
        return score, evaluation
    
    def plot_candidate(self,
                       parameters: object,
                       *args,
                       **kwargs) -> None:
        pass
