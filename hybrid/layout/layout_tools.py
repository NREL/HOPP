from math import fabs
from typing import (
    Callable,
    Tuple,
    )

import numpy as np
from shapely.geometry import Polygon


def binary_search_float(objective: Callable[[float], any],
                        minimum: float,
                        maximum: float,
                        max_iters: int = 32,
                        threshold: float = 1e-3,
                        ) -> (float, bool):
    """
    :param objective: function for which to find fixed point
    :param minimum: min value of search
    :param maximum: max value of search
    :param max_iters: max iterations
    :param threshold: distance between max and min search points upon which to exit early
    :return: solution
    """
    if fabs(maximum - minimum) < threshold:
        return maximum, True
    if minimum > maximum:
        raise ValueError(f"binary search minimum {minimum} must be less than maximum {maximum}")
    candidate = 0.0
    for i in range(max_iters):
        candidate = (maximum + minimum) / 2
        evaluation = objective(candidate)
        
        if fabs(maximum - minimum) < threshold:
            return candidate, True
        
        if evaluation < 0:  # candidate < target
            minimum = candidate
        elif evaluation > 0:  # candidate > target
            maximum = candidate
    
    return candidate, False


def binary_search_int(objective: Callable[[int], any],
                      minimum: int,
                      maximum: int,
                      ) -> (int, bool):
    """
    :param objective: function for which to find fixed point
    :param minimum: min value of search
    :param maximum: max value of search
    :return: solution
    """
    if minimum > maximum:
        raise ValueError(f"binary search minimum {minimum} must be less than maximum {maximum}")
    candidate = 0
    while minimum < maximum:
        candidate = (maximum + minimum) // 2
        evaluation = objective(candidate)
        
        if evaluation < 0:  # candidate < target
            minimum = candidate + 1
        elif evaluation > 0:  # candidate > target
            maximum = candidate
        else:  # candidate == target
            return candidate, True
    return candidate, False


def make_polygon_from_bounds(sw_bound: np.ndarray,
                             ne_bound: np.ndarray
                             ) -> Polygon:
    return Polygon([
        sw_bound.tolist(),
        [sw_bound[0], ne_bound[1]],
        ne_bound.tolist(),
        [ne_bound[0], sw_bound[1]]])


def clamp(value,
          error,
          minimum,
          maximum
          ) -> Tuple:
    delta = 0.0
    if value > maximum:
        delta = value - maximum
        value = maximum
    elif value < minimum:
        delta = minimum - value
        value = minimum
    return value, error + delta ** 2
