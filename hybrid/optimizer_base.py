import ast
import csv
import sys
import numpy as np

from hybrid.scenario import Scenario


class Optimizer:

    """
    Class that iteratively generates and evaluates sample scenarios according to an optimization algorithm
    Attributes:
            scenario:               A Scenario instance for running system simulations
            n_dim:                  Dimension of the search space, ie length of coordinates vector
            r_dim:                  Dimension of the output space, ie length of output vector
            system_behavior:        Dict where k:v = system: func(systems) -> None that encodes the system behavior
            sampling_function:      func(dim, param) -> n-dim vector, normalized
            output_eval_function:   func(r-dim vector) -> float
            stopping_condition:     func(n_iters, max_output, max_output_prev) -> bool
            output_save_function:   func(n_it, sample, output_val) -> None
    """
    def __init__(self, defaults, input_data, output_info, system_behavior):
        """
        :param defaults: A nested dictionary with k:v = system: model: group: variable: value
        :param input_data: A nested dictionary with k:v = system: model: group: variable: range
        :param output_info: A nested dictionary with k:v = system: model: group: [values]
        :param system_behavior: A dict with k:v = system: func(systems) -> None that encodes the system behavior
        """

        self.scenario = Scenario(defaults, input_data, output_info, system_behavior)
        self.n_dim = self.scenario.vec_length
        self.r_dim = len(self.scenario.outputs_names)
        self.output_info = output_info

        self.sampling_function = None
        self.output_eval_function = None
        self.stopping_condition = None
        self.output_store_function = None

    def setup(self, sampling_func=None, output_eval_func=None, stopping_cond=None, output_store_func=None):
        self.setup_sampling_function(sampling_func)
        self.setup_output_eval_function(output_eval_func)
        self.setup_stopping_condition(stopping_cond)
        self.output_store_function = output_store_func

    def setup_sampling_function(self, func=None):
        """
        Method to create a sample as a n-dim vector
        :param func: func(dim, param) -> n-dim vector, normalized
        """
        if not func:
            self.sampling_function = self.random_sampling
        else:
            self.sampling_function = func

    @staticmethod
    def random_sampling(dim, n):
        return np.random.rand(dim)

    def setup_output_eval_function(self, func=None):
        """
        function to weight output vector as a scalar value
        :param func: func(r-dim vector) -> float
        """
        if not func:
            self.output_eval_function = self.average_all
        else:
            self.output_eval_function = func

    @staticmethod
    def average_all(output):
        if len(output) == 1:
            return output[0]
        return sum([i if type(i) == float else sum(i)/len(i) for i in output]) / len(output)

    def setup_stopping_condition(self, func=None):
        """
        function to decide loop termination
        :param func: func(n_iters, max_output, max_output_prev) -> bool
        """
        if not func:
            self.stopping_condition = self.tol_limit
        else:
            self.stopping_condition = func

    @staticmethod
    def tol_limit(n_iters, max_output, max_output_prev):
        if n_iters > 0 and abs(max_output-max_output_prev)/max_output_prev < 0.001:
            return True
        if n_iters > 100:
            return True
        return False

    def evaluate_scenario(self, coordinates):
        """
        Assigns the scenario's parameters and evaluates the simulation output
        :param coordinates: n-length vector, normalized
        :return: scalar value of output
        """
        self.scenario.setup_single(coordinates)
        output_vec = self.scenario.run_single()
        return self.output_eval_function(output_vec)

    def optimize(self):
        """
        Basic optimization routine with stopping condition, sample generation & sample evaluation functions
        :return: tuple with max_output, max_coords, n_iteration, output_vec
        """
        n_iterations = 0
        max_output = -sys.float_info.max
        max_output_prev = max_output
        max_coords = []
        max_vec = []
        while not self.stopping_condition(n_iterations, max_output, max_output_prev):
            sample = self.sampling_function(self.n_dim, n_iterations)
            self.scenario.setup_single(sample)
            output_vec = self.scenario.run_single()
            output_val = self.output_eval_function(output_vec )
            if self.output_store_function:
                self.output_store_function(n_iterations, sample, output_val)
            if output_val > max_output:
                max_output_prev = max_output
                max_output = output_val
                max_coords = sample
                max_vec = output_vec
            n_iterations += 1
        return max_output, max_coords, n_iterations, max_vec


def csv_to_dict(csvfile):
    """
    Given a CSV file with one header row, create a dictionary
    where each key corresponds to a column label, and the values
    for that key are a list

    Parameters
        ---------
        csvfile: string
            path to a csv file to read into a dict
    """
    with open(csvfile) as f:
        reader = csv.reader(f)
        data = dict()
        header = list()
        for i, row_data in enumerate(reader):
            if i == 0:
                for col in row_data:
                    header.append(col)
                    data[col] = list()
            else:
                for j, col_data in enumerate(row_data):
                    data[header[j]].append(ast.literal_eval(col_data))

    return data

