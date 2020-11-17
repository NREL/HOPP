import random
from typing import Union

import numpy as np

from tools.optimization.optimizer.dimension.dimension_info import DimensionInfo


# class WindOptKFDCE(WindOpt):
#
#     def __init__(self,
#                  site_info: SiteInfo,
#                  num_turbines: int,
#                  generation_size: int,
#                  selection_proportion: float,
#                  sensor_noise: float = 0.0,
#                  dynamic_noise: float = 0.0):
#         super().__init__(site_info, num_turbines)
#
#         dimensions = [None] * (num_turbines * 2)
#         for i, dist in enumerate(self.get_starting_distributions()):
#             dimensions[i] = KFDCEM.KFDimension(
#                 dist[0],
#                 dist[1],
#                 sensor_noise,
#                 dynamic_noise)
#             dimensions[num_turbines + i] = KFDCEM.KFDimension(
#                 dist[2],
#                 dist[3],
#                 sensor_noise,
#                 dynamic_noise)
#
#         self.optimizer = KFDCEM(
#             dimensions,
#             generation_size,
#             selection_proportion)
#
#
# class WindOptKFPDCE(WindOpt):
#
#     def __init__(self,
#                  site_info: SiteInfo,
#                  num_turbines: int,
#                  generation_size: int,
#                  selection_proportion: float,
#                  mu_variance: float,
#                  mu_sensor_noise: float,
#                  mu_dynamic_noise: float,
#                  sigma_variance: float,
#                  sigma_sensor_noise: float,
#                  sigma_dynamic_noise: float):
#         super().__init__(site_info, num_turbines)
#
#         dimensions = [None] * (num_turbines * 2)
#         for i, dist in enumerate(self.get_starting_distributions()):
#             dimensions[i] = KFDCEM.KFParameterDimension(
#                 KFDCEM.KFDimension(
#                     dist[0],
#                     mu_variance,
#                     mu_sensor_noise,
#                     mu_dynamic_noise),
#                 KFDCEM.KFDimension(
#                     dist[1],
#                     sigma_variance,
#                     sigma_sensor_noise,
#                     sigma_dynamic_noise))
#             dimensions[num_turbines + i] = KFDCEM.KFParameterDimension(
#                 KFDCEM.KFDimension(
#                     dist[2],
#                     mu_variance,
#                     mu_sensor_noise,
#                     mu_dynamic_noise),
#                 KFDCEM.KFDimension(
#                     dist[3],
#                     sigma_variance,
#                     sigma_sensor_noise,
#                     sigma_dynamic_noise))
#
#         self.optimizer = KFDCEM(
#             dimensions,
#             generation_size,
#             selection_proportion)


class Gaussian(DimensionInfo):
    
    def __init__(self, mu: float, sigma: float, scale: float = 1.0):
        """
        Gaussian distribution
        :param mu: mean
        :param sigma: standard deviation
        :param scale: sigma scaling
        """
        self.mu: float = mu
        self.sigma: float = sigma
        self.scale: float = scale
    
    def update(self, samples: [float]) -> None:
        """
        Update parameters of the pdf with new samples
        :param samples: list of best candidates from optimizer
        """
        self.mu = np.mean(samples, 0)
        self.sigma = np.std(samples, 0, ddof=1) * self.scale
    
    def sample(self) -> float:
        """
        :return: Sample from the pdf
        """
        return random.gauss(self.mu, self.sigma)
    
    def best(self) -> Union[float, int]:
        """
        :return: Most likely sample
        """
        return self.mu
    
    def mean(self) -> Union[float, int]:
        return self.mu
    
    def variance(self) -> Union[float, int]:
        return self.sigma ** 2
