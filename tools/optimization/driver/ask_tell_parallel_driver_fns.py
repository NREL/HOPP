from functools import partial

"""
These functions are helpers for AskTellParallelDriver that have been moved into their own file to reduce the
amount of dependencies pickled and unpickled when interacting with the parallel driver's pool.
"""

__objective = None

def make_initializer(objective):
    """
    Wraps the objective in a function to initialize a pool
    """
    return partial(set_objective, objective=objective)


def set_objective(objective):
    """
    Sets the objective for (this process in) the pool
    """
    global __objective
    __objective = objective


def evaluate(candidate):
    """
    Evaluates the given candidate
    """
    global __objective
    return __objective(candidate)

# def flatten_list(nested_list: [[any]]) -> [any]:
#     result = []
#     for sublist in nested_list:
#         result.extend(sublist)
#     return result
