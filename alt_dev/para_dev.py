import concurrent.futures as cf
import threading
import multiprocessing
import time
import numpy as np
from skopt.benchmarks import branin
from functools import partial
import warnings
warnings.simplefilter("ignore")
import humpday
warnings.simplefilter("default")
import inspect

class Worker(multiprocessing.Process):
    """
    Process worker to execute objective calculations
    """

    def __init__(self, task_queue, cache):
        super().__init__()
        self.task_queue = task_queue
        self.cache = cache

    def run(self):
        proc_name = self.name
        while True:
            # Get task from queue
            task = self.task_queue.get()

            if task is None:
                # Signal shutdown
                print('%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break

            # Execute task
            candidate, result = task()

            self.task_queue.task_done()
            self.cache[candidate] = result
        return


class Task(object):
    """
    Mock problem class to define the objective calculation
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.lower_bounds = (-5, 0)
        self.upper_bounds = (10, 15)

    def __call__(self):
        # time.sleep(0.5) # pretend to take some time to do the work
        x = [self.a*(self.upper_bounds[0]-self.lower_bounds[0]) + self.lower_bounds[0],
             self.b*(self.upper_bounds[1]-self.lower_bounds[1]) + self.lower_bounds[1]]
        try:
            result = branin(x)
        except Exception as error:
            result = np.nan

        return (self.a, self.b), result


def func(x, name=None, tasks=None, cache=None, lock=None):
    """
    Mock Optimization loop (just loops through a few candidates and exits)
    """
    candidate = tuple(x)
    try:
        # Check if result in cache
        lock.acquire()
        result = cache[candidate]
        lock.release()

        if result is None:
            # In cache but not complete, poll cache
            while (result := cache[candidate]) is None:
                time.sleep(0.01)

            with lock:
                print(f'{name} Cache wait:', candidate, result)
            return result

        else:
            # Result available in cache, no work needed
            with lock:
                print(f'{name} Cache hit:', candidate, result)
            return result

    except KeyError:
        # Candidate not in cache
        cache[candidate] = None # indicates waiting in cache
        print(f'{name} Candidate entering task queue:', candidate)
        tasks.put(Task(*candidate))
        lock.release()

        # Poll cache for available result (should be threading.Condition)
        while (result := cache[candidate]) is None:
            time.sleep(0.01)

        with lock:
            print(f'{name} Task return:', candidate, result)

        return result


def kill_workers(tasks, num_workers):
    # Signal exit for worker processes
    for i in range(num_workers):
        tasks.put(None)

if __name__ == '__main__':
    """
    Create process workers
    Start optimizers to fill task queue
    Wait for everything to finish and clean everything up
    Lots of logging print statements
    """
    times = []
    for num_workers in [12]:#range(1, 6): #range(1, 11):
        # Establish communication queues
        tasks = multiprocessing.JoinableQueue()
        manager = multiprocessing.Manager()
        cache = manager.dict()

        # Start workers
        # num_workers = 3
        print('Creating %d workers' % num_workers)
        workers = [Worker(tasks, cache) for i in range(num_workers)]

        for w in workers:
            w.start()

        # Starting threads that act like optimizers
        start = time.perf_counter()
        lock = threading.Lock()

        n_opt = 12
        obj = [partial(func, name=str(i), tasks=tasks, cache=cache, lock=lock) for i in range(n_opt)]
        for i in range(n_opt):
            obj[i].__name__ = humpday.OPTIMIZERS[i].__name__

        opt = [partial(humpday.OPTIMIZERS[i], n_dim=2, n_trials=50, with_count=True) for i in range(n_opt)]

        with cf.ThreadPoolExecutor(max_workers=n_opt) as executor:
            threads = {executor.submit(opt[i], obj[i]): i for i in range(n_opt)}

            for future in cf.as_completed(threads):
                wait = threads[future]
                result = future.result()

                with lock:
                    print(wait, 'thread finished', result)

        end = time.perf_counter()
        times.append(end-start)
        print(f'\n#### Elapsed time: {end - start:.2f} secs #### \n')

        # End worker processes
        kill_workers(tasks, num_workers)

        # Wait for all of the tasks to finish
        tasks.join()
        for w in workers:
            w.join()

        # Start printing results
        # for key, value in cache.items():
        #     print('Result:', key, value)

    print(times)