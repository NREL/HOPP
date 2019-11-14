import functools
from typing import (
    Callable,
    Tuple,
    Optional,
    )

import dill
import pathos.multiprocessing as multiprocessing

# import multiprocessing

from optimization.driver.ask_tell_driver import AskTellDriver
from optimization.optimizer.ask_tell_optimizer import AskTellOptimizer

# def run_dill_encoded(payload):
#     fun, args = dill.loads(payload)
#     return fun(*args)
#
#
# def dill_map(pool, fun, args):
#     payload = dill.dumps((fun, args))
#     return pool.map(run_dill_encoded, (payload,))

        
def evaluate(objective, optimizer, c) -> Tuple[float, any]:
    """
    split out of the class to reduce the amount of data pickled when using multiprocessing
    """
    return objective(optimizer.ask(1)[0])

class AskTellParallelDriver(AskTellDriver):
    
    def __init__(self):
        self.num_evaluations: int = 0
        self.num_iterations: int = 0
        # self.pool = multiprocessing.Pool()
    
    def step(self,
             optimizer: AskTellOptimizer,
             objective: Callable[[any], Tuple[float, any]],
             ) -> bool:
        evaluations: [Tuple[float, any]] = []
        
        num_candidates = optimizer.get_num_candidates()
        
        # if num_candidates is None:
        #     # do a serial ask call
        #     candidates: [any] = optimizer.ask()  # TODO: could parallelize asking as well
        #
        #     def evaluate_one(candidate: any) -> Tuple[float, any]:
        #         return objective(candidate)
        #
        #     evaluations = joblib.Parallel(n_jobs=-1, batch_size='auto', prefer='processes')(
        #         joblib.delayed(evaluate_one)(candidate) for candidate in candidates)
        # else:
        # parallelize asking (candidate generation)
        # def evaluate_one(c: int) -> Tuple[float, any]:
        #     return objective(optimizer.ask(1)[0])
        
        evaluate_one = functools.partial(evaluate, objective, optimizer)
        
        # print('baditems: ', dill.detect.baditems(evaluate_one))
        # print('badobjects: ', dill.detect.badobjects(evaluate_one))
        # print('errors: ', dill.detect.errors(evaluate_one))
        # print('trace: ', dill.detect.trace(evaluate_one))

        pool = multiprocessing.Pool()
        # evaluations = self.pool.map(lambda c : objective(optimizer.ask(1)[0]), range(num_candidates))
        evaluations = pool.map(evaluate_one, range(num_candidates))
        
        # evaluations = joblib.Parallel(n_jobs=-1, batch_size='auto', prefer='processes')(
        #     joblib.delayed(evaluate_one)(c) for c in range(num_candidates))
        
        optimizer.tell(evaluations)
        self.num_evaluations += len(evaluations)
        self.num_iterations += 1
        
        return optimizer.stop()
    
    def get_num_evaluations(self) -> int:
        return self.num_evaluations
    
    def get_num_iterations(self) -> int:
        return self.num_iterations
