from __future__ import annotations
from abc import abstractmethod
from collections import OrderedDict
import numpy as np

from hybrid.layout.layout_tools import clamp
from hybrid.hybrid_simulation import HybridSimulation


class OptimizationProblem:
    """
    Interface between optimization drivers and the simulations being optimized that uses a parametrization of the
    original problem.
    """

    def __init__(self,
                 ) -> None:
        self.candidate_dict: OrderedDict = OrderedDict()

    def get_prior_params(self,
                         distribution_type
                         ) -> dict:
        """
        Returns the parameters for each parameter's distribution for a given distribution type
        :param distribution_type: str identifier ("Gaussian", "Bernoulli", ...)
        :return: dictionary of parameters
        """

        if distribution_type.__name__ == "Gaussian":
            priors = dict()
            for k, v in self.candidate_dict.items():
                priors[k] = dict()
                for p, vv in v["prior"].items():
                    priors[k][p] = vv
            return priors
        else:
            raise NotImplementedError

    def check_candidate(self,
                        candidate: np.ndarray):
        if len(candidate) != len(self.candidate_dict):
            raise ValueError("Candidate vector is of incorrect length. Check design variables.")
        # for n, k in enumerate(self.candidate_dict.keys()):
        #     if type(candidate[n]) != self.candidate_dict[k]["type"]:
        #         raise ValueError("{} variable should be of type {} not {}".format(k,
        #                                                                           self.candidate_dict[k]["type"],
        #                                                                           type(candidate[n])))

    def convert_to_candidate(self,
                             parameters: dict) -> np.ndarray:
        candidate = np.zeros(len(self.candidate_dict))
        for n, k in enumerate(self.candidate_dict.keys()):
            if k not in parameters.keys():
                raise ValueError("Design variables require {} parameter".format(k))
            candidate[n] = parameters[k]
        self.check_candidate(candidate)
        return candidate

    @abstractmethod
    def _set_simulation_to_candidate(self,
                                     candidate: np.ndarray,
                                     ) -> tuple[float, any]:
        """
        Transforms parametrized into inner problem candidate
        :param candidate:
        :return: a penalty and an inner candidate
        """
        pass

    def conform_candidate_and_get_penalty(self,
                                          candidate: np.ndarray
                                          ) -> tuple:
        """
        Modifies a candidate's parameters so that it falls within range
        :param candidate: optimization candidate
        :return: conforming candidate, parameter error values
        """
        conforming_candidate = np.copy(candidate)
        parameter_error: float = 0.0

        for n, var in enumerate(self.candidate_dict.keys()):
            conforming_candidate[n], parameter_error = \
                clamp(candidate[n],
                      parameter_error,
                      self.candidate_dict[var]["min"],
                      self.candidate_dict[var]["max"])

        return conforming_candidate, parameter_error

    @abstractmethod
    def objective(self,
                  candidate: np.ndarray
                  ) -> tuple[float, float, any]:
        """
        Returns simulated performance of candidate
        :param candidate: optimization candidate
        :return: performance
        """
        pass

    def plot_candidate(self,
                       parameters: object,
                       *args,
                       **kwargs) -> None:
        pass
