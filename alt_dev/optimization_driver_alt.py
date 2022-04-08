import functools
import inspect
import time
import os
from datetime import datetime

import queue
import traceback

import pandas as pd
import numpy as np

from diskcache import Cache, JSONDisk
from functools import wraps, partial

import concurrent.futures as cf
import threading
import multiprocessing
from typing import Callable


def get_best_from_cache(cache: Cache, objective: Callable) -> tuple:
    """
    Helper function for locating the best candidate according to some objective callable in a cache

    :param cache: diskcache Cache, as in the driver cache
    :param objective: callable function accepting a simulation result, and returning a scalar float objective
    :return: tuple containing the best candidate and result
    """
    # init
    best_result = np.inf
    best_candidate = None

    # for each candidate
    for candidate in cache.iterkeys():
        if candidate == 'meta':
            continue

        # try to compute the objective
        try:
            # user defined objective value
            result = objective(cache[candidate])

            # update best candidate and results
            if result < best_result:
                best_result = result
                best_candidate = candidate

        # on error, go to next result
        except:
            continue

    return best_candidate, best_result


# def flatten_dict(result: dict, sep='__', prev_key='') -> dict:
#     """
#     Helper function for flattening a result nested dictionary into a flat dictionary
#
#     :param result: a hybrid simulation result nested dictionary, as in the output from problem.evaluate_objective
#     :param sep: separator string used to concatenate nested keys
#     :param prev_key: combination of all previous keys in the nested dictionary
#     :return: a dictionary with no other dictionaries as values
#     """
#     row = dict()
#
#     # for each key
#     for key, value in result.items():
#         subkey = key if prev_key == '' else sep.join([prev_key, key])
#
#         # if the value is itself a dictionary, recurse
#         if isinstance(value, dict):
#             row.update(flatten_dict(value, sep, prev_key=subkey))
#
#         # add the value to the output dictionaruyu
#         else:
#             row.update({subkey: value})
#
#     return row


class OptimizerInterrupt(Exception):
    """
    Stub exception used by the driver to interrupt optimizers (e.g., if time limit has been exceeded)
    """
    pass


class Worker(multiprocessing.Process):
    """
    Process-contained worker to execute objective calculations.
    """

    def __init__(self, task_queue, cache, setup: Callable) -> None:
        """
        Process-contained worker, having an independent instance of the problem and simulation to evaluate the objective

        :param task_queue: multiprocessing.JoinableQueue()
        :param cache: multiprocessing.manager.dict()
        :param setup: function to create a new instance of the design problem
        """
        super().__init__()
        self.task_queue = task_queue
        self.cache = cache
        self.setup = setup

    def run(self):
        """
        Poll the task queue until a task (candidate, caller_name) are available, (None, None) or a KeyboardInterrupt
            signals shutdown

        :return: None
        """

        # Create a new problem for the worker
        problem = self.setup()

        # proc_name = self.name # not currently used
        candidate = None

        while True:
            try:
                # Get task from queue, this method blocks this process until a task is available
                candidate, caller_name = self.task_queue.get()

                if candidate is None:
                    # Signal shutdown
                    self.task_queue.task_done()
                    break

                # Execute task, measure evaluation time
                start_time = time.time()
                result = problem.evaluate_objective(candidate)
                result['eval_time'] = time.time() - start_time
                result['caller'] = [caller_name]

            except KeyboardInterrupt:
                # Exit cleanly
                self.task_queue.task_done()

                # Signal any waiting optimizer threads to exit
                if candidate is not None:
                    # self.cache[candidate] = OptimizerInterrupt
                    self.cache.set(candidate, 'OptimizerInterrupt', tag='exception')

                break

            # Objective returns normally, mark task as done and return result
            self.task_queue.task_done()
            self.cache.set(candidate, result, tag='result')


class OptimizationDriver():
    """
    Object to interface the HOPP optimization problem with humpday optimizers
    """
    DEFAULT_KWARGS = dict(time_limit=np.inf,  # total time limit in seconds
                          eval_limit=np.inf,  # objective evaluation limit (counts new evaluations only)
                          obj_limit=-np.inf,  # lower bound of objective, exit if best objective is less than this
                          n_proc=multiprocessing.cpu_count() - 4,  # maximum number of objective process workers
                          cache_dir='driver_cache',  # filename for the driver cache file
                          reconnect_cache=False,  # True if the driver should reconnect to a previous result cache
                          write_csv=False,  # True if the cached results should be written to csv format files
                          dataframe_file='study_results.df.gz',  # filename for the driver cache dataframe file
                          csv_file='study_results.csv',  # filename for the driver cache csv file
                          scaled=True,  # True if the sample/optimizer candidates need to be scaled to problem units
                          retry=True)  # True if any evaluations ending in an exception should be retried on restart

    def __init__(self,
                 setup: Callable,
                 **kwargs) -> None:
        """
        Object to interface the HOPP optimization problem with humpday optimizers

        :param setup: Function which creates and returns a new instance of the optimization problem
        :param kwargs: Optional keyword arguments to change driver options (see DEFAULT_KWARGS)
        """

        self.setup = setup
        self.problem = setup()  # The driver needs an instance of the problem to access problem.candidate_from()
        self.parse_kwargs(kwargs)

        self.best_obj = None
        self.cache_info = dict(hits=0, misses=0, size=0, total_evals=0)
        self.meta = dict()

        self.get_candidate = self.problem.candidate_from_unit_array if self.options['scaled'] \
            else self.problem.candidate_from_array  # Function to create formatted design candidates
        self.start_time = None
        self.force_stop = False
        self.eval_count = 0

        if not self.options['reconnect_cache']:
            i = 1
            check_dir = self.options['cache_dir']

            while os.path.isdir(check_dir):
                check_dir = f"{self.options['cache_dir']}_{i}"
                i += 1

            self.options['cache_dir'] = check_dir

        self.cache = Cache(self.options['cache_dir'], disk=JSONDisk, disk_compress_level=9,
                           cull_limit=0, statistics=1, eviction_policy='none')
        self.start_len = len(self.cache)
        self.read_cache()

    def parse_kwargs(self, kwargs: dict) -> None:
        """
        Helper function to set defaults and update options with user-provided input

        :param kwargs: Using ``**kwargs`` this is a dict of keyword arguments provided by the user
        :return: None
        """
        self.options = self.DEFAULT_KWARGS.copy()

        for key, value in kwargs.items():
            if key in self.options:
                self.options[key] = value
            else:
                print(f"Ignoring unknown driver option {key}={value}")

    def init_parallel_workers(self, num_workers: int) -> None:
        """
        Create the communication queue, cache dictionary, thread lock, and worker processes

        :param num_workers: Number of process-independent workers, which evaluate the objective.
        :return:
        """

        if not hasattr(self, 'tasks'):
            self.tasks = multiprocessing.JoinableQueue()
            self.lock = threading.Lock()

        print(f"Creating {num_workers} workers")
        self.workers = [Worker(self.tasks, self.cache, self.setup)
                        for _ in range(num_workers)]

        # Start the workers polling the task queue
        for w in self.workers:
            w.start()

    def cleanup_parallel(self) -> None:
        """
        Cleanup all worker processes, signal them to exit cleanly, mark any pending tasks as complete

        :return: None
        """

        # If the driver receives a KeyboardInterrupt then the task queue needs to be emptied
        if self.force_stop:
            try:
                # Mark all tasks complete
                while True:
                    self.tasks.get(block=False)
                    self.tasks.task_done()

            # Occurs when task queue is empty
            except queue.Empty:
                pass

        else:
            # Exit normally, None task signals each worker to exit
            for i in range(len(self.workers)):
                self.tasks.put((None, 'worker exit'))

        # Wait for all of the tasks to finish
        self.tasks.join()
        for w in self.workers:
            w.join()
            del w

    def check_interrupt(self) -> None:
        """
        Check optional stopping criteria, these are specified by the user in the driver options

        :return: None
        """
        if self.force_stop:
            raise OptimizerInterrupt

        elapsed = time.time() - self.start_time
        if elapsed > self.options['time_limit']:
            # print(f"Driver exiting, time limit: {self.options['time_limit']} secs")
            # logging.info(f"Driver exiting, time limit: {self.options['time_limit']} secs")
            raise OptimizerInterrupt

        if self.eval_count >= self.options['eval_limit']:
            # print(f"Driver exiting, eval limit: {self.options['eval_limit']}")
            # logging.info(f"Driver exiting, eval limit: {self.options['eval_limit']}")
            raise OptimizerInterrupt

        if (self.best_obj is not None) and (self.best_obj <= self.options['obj_limit']):
            # print(f"Driver exiting, obj limit: {self.options['obj_limit']}")
            # logging.info(f"Driver exiting, obj limit: {self.options['obj_limit']}")
            raise OptimizerInterrupt

    def print_log_header(self) -> None:
        """
        Print a linear solver-style log header.

        """
        self.log_headers = ['N_Evals', 'Objective', 'Eval_Time', 'Total_Time']
        self.log_widths = [len(header) + 5 for header in self.log_headers]

        print()
        print("##### HOPP Optimization Driver #####".center(sum(self.log_widths)))
        print("Driver Options:", self.options, sep="\n\t")
        print("Optimizer Options:", self.opt_names, sep="\n\t")
        print()
        print("".join((val.rjust(width) for val, width in zip(self.log_headers, self.log_widths))))

    def print_log_line(self, reason, obj, eval_time) -> None:
        """
        Print a linear solver-style log line.

        :param info: Dictionary containing at least the evaluation time of the last iteration and reason why a log
            line is being printed. Originally lines would be printed for a hit on the cache (denoted by a ``c`` prefix on the
            the line, but this was removed, and lines are now only printed on new evaluations for conciseness.

        :returns: None
        """
        prefix_reasons = {'cache_hit': 'c ', 'new_best': '* ', '': ''}
        prefix = prefix_reasons[reason]
        best_objective_str = f"{obj:8g}" if obj is not None else "NA"

        curr_time = time.time()
        log_values = [prefix + str(self.eval_count),
                      f"{best_objective_str}",
                      f"{eval_time / 60:.2f} min",
                      f"{(curr_time - self.start_time) / 60:.2f} min"]
        print("".join((val.rjust(width) for val, width in zip(log_values, self.log_widths))))

    def print_log_end(self, best_candidate, best_objective):
        candidate_str = str(best_candidate) \
            .replace('[', '[\n    ', 1) \
            .replace('], ', '],\n    ') \
            .replace(']]', ']\n  ]')

        print()
        print(f"Best Objective: {best_objective:.2f}")
        print(f"Best Candidate:\n  {candidate_str}")

    def write_cache(self, pd_filename=None, csv_filename=None) -> None:
        """
        Write driver cache out to pickle file

        :param filename: Optional path of file to write out the cache to
        :return:  None
        """
        if self.start_len == len(self.cache) - 1:
            print(f"no new entries in cache ({len(self.cache) - 1} results), skipping write...")
            return

        dt_string = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        pd_dir = os.path.join(self.options['cache_dir'], '_dataframe', dt_string)
        if not os.path.isdir(pd_dir):
            os.makedirs(pd_dir)

        if pd_filename is None:
            pd_filename = os.path.join(pd_dir, self.options['dataframe_file'])
        else:
            pd_filename = os.path.join(pd_dir, pd_filename)

        if self.options['write_csv'] or csv_filename is not None:
            csv_dir = os.path.join(self.options['cache_dir'], '_csv', dt_string)
            if not os.path.isdir(csv_dir):
                os.makedirs(csv_dir)

            if csv_filename is None:
                csv_filename = os.path.join(csv_dir, self.options['csv_file'])
            else:
                csv_filename = os.path.join(csv_dir, csv_filename)

        # Gather meta data
        self.meta['cache_info'] = self.cache_info.copy()
        self.meta['driver_options'] = self.options
        self.meta['start_time'] = self.start_time
        self.meta['candidate_fields'] = self.problem.candidate_fields
        self.meta['design_variables'] = self.problem.design_variables
        self.meta['fixed_variables'] = self.problem.fixed_variables

        try:
            self.meta['problem_setup'] = inspect.getsource(self.setup)
        except TypeError:
            if isinstance(self.setup, partial):
                self.meta['problem_setup'] = inspect.getsource(self.setup.func) \
                                             + '__args__' + str(self.setup.args) + '\n' \
                                             + '__keywords__' + str(self.setup.keywords) + '\n'

        try:
            self.meta['sim_setup'] = inspect.getsource(self.problem.init_simulation)
        except TypeError:
            if isinstance(self.problem.init_simulation, partial):
                self.meta['sim_setup'] = inspect.getsource(self.problem.init_simulation.func) \
                                             + '__args__' + str(self.problem.init_simulation.args) + '\n' \
                                             + '__keywords__' + str(self.problem.init_simulation.keywords) + '\n'

        try:
            self.meta['eval_obj'] = inspect.getsource(self.problem.evaluate_objective)
        except TypeError:
            if isinstance(self.problem.evaluate_objective, partial):
                self.meta['eval_obj'] = inspect.getsource(self.problem.evaluate_objective.func) \
                                             + '__args__' + str(self.problem.evaluate_objective.args) + '\n' \
                                             + '__keywords__' + str(self.problem.evaluate_objective.keywords) + '\n'

        self.cache['meta'] = self.meta.copy()

        self.start_len = len(self.cache) - 1
        print(f"writing {len(self.cache) - 1} results to dataframe {pd_filename}...")

        data_list = []
        candidate_sep = self.problem.sep
        pandas_sep = '__'

        for candidate in self.cache:
            if candidate == 'meta':
                continue

            result = self.cache.get(candidate)

            if not isinstance(result, dict):
                self.cache.delete(candidate)
                continue

            # row = dict()
            #
            # for key, value in candidate:
            #     key = key.replace(candidate_sep, pandas_sep)
            #     row[key] = value
            #
            # row.update(flatten_dict(result))
            data_list.append(result)

        df = pd.DataFrame(data_list)
        df.attrs = self.meta
        df.to_pickle(pd_filename)

        ### new code to write to csv(s)
        if csv_filename is not None:
            all_cols = sorted(df.columns)
            scalar_cols = [col for col in all_cols
                           if isinstance(df[col].loc[0], np.float64)]
            iterable_cols = [col for col in all_cols
                             if not isinstance(df[col].loc[0], np.float64)]

            df[scalar_cols].to_csv(csv_filename)

            for i in range(len(df)):
                df_row = pd.concat([pd.Series(df[col].loc[i]) for col in iterable_cols],
                                   axis=1, keys=iterable_cols)
                row_filename = os.path.join(csv_dir, f"{i}.csv")
                df_row.to_csv(row_filename)

    def read_cache(self) -> None:
        """
        Read the driver cache from file

        :param filename: Optional path of file to read the cache from
        :return: None
        """
        if len(self.cache) > 0:
            try:
                self.cache_info.update(self.cache['meta']['cache_info'])
                self.meta.update(self.cache['meta'].copy())

            except KeyError:
                pass

        return

    def wrapped_parallel_objective(self):
        """
        This method implements the logic to check if a candidate is in the cache, or is pending evaluation, or neither.
        Each optimizer thread needs its own copy of this method since they don't have access to the driver object, we
        can implement this by wrapping this method and returning the wrapped function. This allows the optimizer threads
        to share the driver object without explicitly passing it to them, and allows them to all use the shared task
        queue and driver cache.

        :return: None
        """
        eval_count = 0

        @wraps(self.wrapped_parallel_objective)
        def p_wrapper(*args, name=None, idx=None, objective=None):
            """
            Objective function the optimizer threads call, assumes a parallel structure and avoids any re-calculations
                - Check if candidate is in cache, if so return objective stored in cache
                - If not, check if candidate is in queue (indicated by integer value in cache), wait for signal
                - If not, objective needs to be calculated, add candidate to task queue, poll cache for return,
                    and finally signal any threads waiting on the same candidate

            :param args: Follows the optimizer's convention of objective inputs (typically an array of floats)
            :param name: Caller name to insert into the result dictionary
            :param idx: Thread index, used for signal conditions
            :param objective_keys: Ordered list of keys to get the objective from the result dictionary
            :return: the numeric value being optimized
            """
            nonlocal eval_count
            eval_count += 1

            self.check_interrupt()
            candidate = self.get_candidate(*args)
            self.cache_info['total_evals'] += 1
            obj = None

            try:
                # Check if result in cache, throws KeyError if not
                self.lock.acquire()
                result = self.cache[candidate]
                # print(f"cache hit {self.cache_info['total_evals']}")
                self.lock.release()
                self.cache_info['hits'] += 1

                if isinstance(result, int):
                    # In cache but not complete, wait for complete signal
                    signal = self.conditions[result]
                    with signal:
                        signal.wait()

                    result = self.cache[candidate]

                if not isinstance(result, dict):
                    self.force_stop = True
                    self.check_interrupt()

                if 'exception' in result.keys():
                    if self.options['retry']:
                        with self.lock:
                            self.cache.delete(candidate)

                        raise KeyError

                # Result available in cache, no work needed
                # Append this caller name to the result dictionary
                with self.lock:
                    result['caller'].append((name, eval_count))
                    self.cache[candidate] = result

            except KeyError:
                # Candidate not in cache, nor waiting in queue
                self.cache[candidate] = idx  # indicates waiting condition for any other thread

                # Insert candidate and caller information into task queue
                self.tasks.put((candidate, (name, eval_count)))

                self.lock.release()
                self.cache_info['misses'] += 1

                # Poll cache for available result (unclear how this could be a threading.Condition signal)
                result = self.cache[candidate]
                while isinstance(result, int):
                    time.sleep(10)
                    result = self.cache[candidate]

                # Signal any other threads waiting on the same candidate
                signal = self.conditions[idx]
                with signal:
                    signal.notifyAll()

                # KeyboardInterrupt places a OptimizerInterrupt in the cache to signal a force_stop
                if not isinstance(result, dict):
                    self.force_stop = True
                    self.check_interrupt()

                # Update best best objective if needed, and print a log line to console
                if objective is not None:
                    obj = objective(result)

                    if (self.best_obj is None) or (obj < self.best_obj):
                        self.best_obj = obj
                        reason = 'new_best'
                    else:
                        reason = ''
                else:
                    reason = ''

                with self.lock:
                    self.eval_count += 1
                    self.print_log_line(reason, obj, result['eval_time'])

                self.cache_info['size'] += 1

            return obj

        return p_wrapper

    def wrapped_objective(self):
        """
        This method implements the logic to check if a candidate is in the cache, or is pending evaluation, or neither.
        Each optimizer thread needs its own copy of this method since they don't have access to the driver object, we
        can implement this by wrapping this method and returning the wrapped function. This allows the optimizer threads
        to share the driver object without explicitly passing it to them, and allows them to all use the shared task
        queue and driver cache.

        :return: None
        """
        eval_count = 0

        @wraps(self.wrapped_objective)
        def s_wrapper(*args, name=None, objective=None):
            """
            Objective function the optimizer threads call, assumes a parallel structure and avoids any re-calculations
                - Check if candidate is in cache, if so return objective stored in cache
                - If not, check if candidate is in queue (indicated by integer value in cache), wait for signal
                - If not, objective needs to be calculated, add candidate to task queue, poll cache for return,
                    and finally signal any threads waiting on the same candidate

            :param args: Follows the optimizer's convention of objective inputs (typically an array of floats)
            :param name: Caller name to insert into the result dictionary
            :param idx: Thread index, used for signal conditions
            :param objective_keys: Ordered list of keys to get the objective from the result dictionary
            :return: the numeric value being optimized
            """
            nonlocal eval_count
            eval_count += 1

            self.check_interrupt()
            candidate = self.get_candidate(*args)
            self.cache_info['total_evals'] += 1
            obj = None

            try:
                result = self.cache[candidate]
                # print(f"cache hit {self.cache_info['total_evals']}")

                if 'exception' in result.keys():
                    if self.options['retry']:
                        self.cache.delete(candidate)
                        raise KeyError

                self.cache_info['hits'] += 1

                # Result available in cache, no work needed
                # Append this caller name to the result dictionary
                result['caller'].append((name, eval_count))
                self.cache[candidate] = result

            except KeyError:
                # Execute task, measure evaluation time
                start_time = time.time()
                result = self.problem.evaluate_objective(candidate)
                result['eval_time'] = time.time() - start_time
                result['caller'] = [(name, eval_count)]

                self.cache[candidate] = result
                self.cache_info['misses'] += 1

                # KeyboardInterrupt places a OptimizerInterrupt in the cache to signal a force_stop
                if not isinstance(result, dict):
                    self.force_stop = True
                    self.check_interrupt()

                # Update best objective if needed, and print a log line to console
                if objective is not None:
                    obj = objective(result)

                    if (self.best_obj is None) or (obj < self.best_obj):
                        self.best_obj = obj
                        reason = 'new_best'

                    else:
                        reason = ''

                else:
                    reason = ''

                self.eval_count += 1
                self.print_log_line(reason, obj, result['eval_time'])

                self.cache_info['size'] += 1

            return obj

        return s_wrapper

    def execute(self, callables, inputs):
        """
        Execute each pairwise callable given input in separate threads, using up to n_processors or the number of
        callables whichever is less.

        :param callables: A list of callable functions (e.g. a list of optimizer functions)
        :param inputs: A list of inputs, one for each callable (e.g. a list of wrapped problem objectives)
        :param objective_keys: A list of keys for the result nested dictionary structure
        :return: Either the best objective found, corresponding to objective_keys, or the number of
                successful evaluations if objective_keys is None
        """
        # setup
        self.start_time = time.time()
        self.eval_count = 0
        self.force_stop = False

        # Begin parallel execution
        self.print_log_header()
        output = dict()
        
        try:
            for f, input, name in zip(callables, inputs, self.opt_names):
                try:
                    output[name] = f(input)

                # On an OptimizerInterrupt cancel all pending futures
                except OptimizerInterrupt:
                    break

                # Print any others
                except Exception as exc:
                    err_str = traceback.format_exc()
                    print(f"{name} generated an exception: {err_str}")

                # Optimizer thread exits normally
                else:
                    # Print done message if input is a function (as in an optimization run)
                    if callable(inputs[0]):
                        print(f"Optimizer {name} finished")

        # Allows clean exit on KeyboardInterrupt
        except KeyboardInterrupt:
            pass

        self.write_cache()
        self.cache.close()
        return output

    def parallel_execute(self, callables, inputs):
        """
        Execute each pairwise callable given input in separate threads, using up to n_processors or the number of
        callables whichever is less.

        :param callables: A list of callable functions (e.g. a list of optimizer functions)
        :param inputs: A list of inputs, one for each callable (e.g. a list of wrapped problem objectives)
        :param objective_keys: A list of keys for the result nested dictionary structure
        :return: Either the best objective found, corresponding to objective_keys, or the number of
                successful evaluations if objective_keys is None
        """
        # setup
        self.start_time = time.time()
        self.eval_count = 0
        self.force_stop = False

        # Establish communication queues and execution workers-
        num_workers = min(self.options['n_proc'], len(callables))  # optimizers are assumed to be serial
        self.init_parallel_workers(num_workers)

        # Add thread conditions to allow signaling between threads waiting on the same candidate
        self.conditions = [threading.Condition() for _ in range(len(callables))]

        # Begin parallel execution
        self.print_log_header()
        output = dict()
        
        with cf.ThreadPoolExecutor(max_workers=num_workers) as executor:
            try:
                threads = {executor.submit(callables[i], inputs[i]): name for i, name in enumerate(self.opt_names)}

                for future in cf.as_completed(threads):
                    name = threads[future]
                    try:
                        output[name] = future.result()

                    # On an OptimizerInterrupt cancel all pending futures
                    except OptimizerInterrupt:
                        for f, name in threads.items():
                            f.cancel()
                        break

                    # Print any others
                    except Exception as exc:
                        err_str = traceback.format_exc()
                        print(f"{name} generated an exception: {err_str}")

                    # Optimizer thread exits normally
                    else:
                        # Print done message if input is a function (as in an optimization run)
                        if callable(inputs[0]):
                            print(f"Optimizer {name} finished")

            # Allows clean exit on KeyboardInterrupt
            except KeyboardInterrupt:
                for future, name in threads.items():
                    future.cancel()

        # End worker processes
        self.cleanup_parallel()
        self.write_cache()
        self.cache.close()
        
        return output
        

    def sample(self, candidates, design_name='Sample') -> int:
        """
        Execute the objective function on each candidate in a sample in parallel, using yp to n_processors or the
        number of candidates threads.

        :param candidates: A list of unit arrays corresponding to the samples of a design.
        :return: The number of successful evaluations.
        """
        n_candidates = len(candidates)
        self.opt_names = [f"{design_name}-{i}" for i in range(n_candidates)]

        callables = [partial(self.wrapped_objective(), name=name)
                     for i, name in enumerate(self.opt_names)]

        evaluations = self.execute(callables, candidates)

        return evaluations

    def parallel_sample(self, candidates, design_name='Sample') -> int:
        """
        Execute the objective function on each candidate in a sample in parallel, using yp to n_processors or the
        number of candidates threads.

        :param candidates: A list of unit arrays corresponding to the samples of a design.
        :return: The number of successful evaluations.
        """
        n_candidates = len(candidates)
        self.opt_names = [f"{design_name}-{i}" for i in range(n_candidates)]

        callables = [partial(self.wrapped_parallel_objective(), name=name, idx=i)
                     for i, name in enumerate(self.opt_names)]

        output = self.parallel_execute(callables, candidates)

        return output

    def optimize(self, optimizers, opt_configs, objectives) -> tuple:
        """
        Execute the the list of optimizers on an instance of the wrapped objective function, using up to n_processors
        or the number of optimizers.

        :param optimizers: A list of optimization callable functions, taking the function to be optimized and config.
        :param opt_config: The common optimizer configuration, shared between all optimization functions.
        :param objective_keys: A list of keys for the result nested dictionary structure
        :return: The best candidate and best simulation result found.
        """
        n_opt = len(optimizers)
        self.opt_names = [f"{opt.__name__}-{obj.__name__}" for opt, obj in zip(optimizers, objectives)]
        self.meta['obj'] = {f"{obj.__name__}-{i}":inspect.getsource(obj) for i,obj in enumerate(objectives)}

        # Defining optimizer thread callables and inputs
        # The wrapped objective function is the input to the optimizer
        callables = [partial(opt, **config) for opt,config in zip(optimizers, opt_configs)]
        inputs = [partial(self.wrapped_objective(), name=name, objective=obj)
                  for i, (name, obj) in enumerate(zip(self.opt_names, objectives))]

        # Some optimizers need the threads to have a __name__ attribute, partial objects do not
        for i in range(n_opt):
            inputs[i].__name__ = self.opt_names[i]

        # best_candidate, best_result = self.execute(callables, inputs, objective=objective)
        output = self.execute(callables, inputs)

        return output

    def parallel_optimize(self, optimizers, opt_configs, objectives) -> tuple:
        """
        Execute the the list of optimizers on an instance of the wrapped objective function, using up to n_processors
        or the number of optimizers.

        :param optimizers: A list of optimization callable functions, taking the function to be optimized and config.
        :param opt_config: The common optimizer configuration, shared between all optimization functions.
        :param objective_keys: A list of keys for the result nested dictionary structure
        :return: The best candidate and best simulation result found.
        """
        n_opt = len(optimizers)
        self.opt_names = [f"{opt.__name__}-{obj.__name__}" for opt, obj in zip(optimizers, objectives)]
        self.meta['obj'] = {f"{obj.__name__}-{i}":inspect.getsource(obj) for i,obj in enumerate(objectives)}

        # Defining optimizer thread callables and inputs
        # The wrapped objective function is the input to the optimizer
        callables = [partial(opt, **config) for opt,config in zip(optimizers, opt_configs)]
        inputs = [partial(self.wrapped_parallel_objective(), name=name, objective=obj, idx=i)
                  for i, (name, obj) in enumerate(zip(self.opt_names, objectives))]

        # Some optimizers need the threads to have a __name__ attribute, partial objects do not
        for i in range(n_opt):
            inputs[i].__name__ = self.opt_names[i]

        # best_candidate, best_result = self.parallel_execute(callables, inputs, objective=objective)
        output = self.parallel_execute(callables, inputs)

        return output
