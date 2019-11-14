"""
Scenario setup

Relies on the SAM:
1. Code generation in JSON format, saved as .json
2. Parametric case generation in csv format
"""
import importlib

from hybrid.solar.layout import calculate_solar_extent
from collections import OrderedDict
from parameters.parameter_utils import *


class Scenario:
    """
    Holds a set of technology-financial systems, and the PySAM classes required to simulate them.
    Keeps an ordered list of input parameters and outputs, both mapped to normalized vectors (coordinates).
    Assigns the values of the tech-fin systems from a coordinate vector, and evalutes the results using
    a system_behavior function per system.

        Attributes:
            system_names:           List of the technologies to model
            systems:                OrderedDict of PySAM models, nested as system: compute_module: group: variable
            system_behavior:        Dict where k:v = system: func(systems) -> None that encodes the system behavior
            parameter_map:          Orders the input parameters into a tuple of (system, model, group, input)
            parameter_range:        Orders the range of the input parameters, or "binary" for a discrete variable
            parameter_names:        List of the parameter variables by name
            vec_length:             Length of a coordinate vector
            coordinate_names:       List of the coordinate variables by name
            outputs_map:            Orderes the output variables into a tuple of (system, model, output)
    """
    
    def __init__(self, defaults, input_data, output_info, system_behavior):
        """
        :param defaults: A nested dictionary with k:v = system: model: group: variable: value
        :param input_data: A nested dictionary with k:v = system: model: group: variable: range
        :param output_info: A nested dictionary with k:v = system: model: group: [values]
        :param system_behavior: A dict with k:v = system: func(systems) -> None that encodes the system behavior
        """
        self.system_names = system_behavior.keys()
        self.system_behavior = system_behavior
        self.systems = OrderedDict()
        self.initialize_default_models(defaults)
        self.defaults = defaults

        self.parameter_map = list()
        self.parameter_range = list()
        self.parameter_names = list()
        self.coordinate_names = list()
        self.outputs_map = list()
        self.outputs_names = list()
        
        self.map_parameters(input_data)
        self.map_outputs(output_info)
        self.vec_length = len(self.coordinate_names)
    
    def initialize_default_models(self, defaults):
        """
        Creates the PySAM models and initializes their values with the provided defaults
        Modules stored in self.systems as nested dictionary with k:v = system: model: <pysam_class>
        :param defaults: a nested dictionary with k:v = system: model: group: variable: value
        """
        for system in self.system_names:
            default_dict = defaults[system]
            self.systems[system] = OrderedDict()
            for model, defs in default_dict.items():
                pysam_module = importlib.import_module("PySAM." + model)
                if type(defs) == str:
                    compute_module = pysam_module.default(defs)
                else:
                    compute_module = pysam_module.new()
                    compute_module.assign(defs)
                self.systems[system][model] = compute_module
    
    def map_parameters(self, input_data):
        """
        Creates the mapping required to transform a dictionary of inputs into a normalized coordinate vector
        and saving the axis names in self.coordinate_names
        :param input_data: a nested dictionary with k:v = system: model: group: variable: range
        """
        for system in self.system_names:
            if system not in input_data.keys():
                continue
            param_dict = input_data[system]
            for model, values in param_dict.items():
                for group, variables in values.items():
                    for var, range_info in variables.items():
                        self.parameter_range.append(range_info)
                        self.parameter_names.append(var)
                        self.parameter_map.append((system, model, group, var))
                        if type(range_info) == list:
                            self.coordinate_names.append(var)
                        # if discrete, use one-hot encoding
                        elif type(range_info) == tuple:
                            for option in range_info:
                                binary_var = var + "-" + str(option)
                                self.coordinate_names.append(binary_var)
                        else:
                            raise ValueError("range info for a variable must be a list or a tuple")
    
    def map_outputs(self, output_info):
        """
        Creates the mapping required to transform a dictionary of outputs into a normalized coordinate vector
        :param output_info: a nested dictionary with k:v = system: model: group: [values]
        """
        for system in self.system_names:
            if system not in output_info.keys():
                continue
            system_outputs = output_info[system]
            for model, values in system_outputs.items():
                for value in values:
                    self.outputs_map.append((system, model, value))
                    self.outputs_names.append(value)
    
    def undo_one_hot_encoding(self, coordinates, c):
        """
        Transforms the group of binary variables produced by one-hot back into a single discrete variable
        :param coordinates: the vector of coordinates, normalized
        :param c: entry of vector where the binary variables begin
        :return: name of the discrete variable and its (integer) value
        """
        name_arr = self.coordinate_names[c].split('-')
        options = self.parameter_range[self.parameter_names.index(name_arr[0])]
        argmax = np.argmax(coordinates[c:c + len(options)])
        return name_arr[0], options[argmax]
    
    def values_from_coordinates(self, coordinates):
        """
        From a normalized coordinate vector, compute the actual variable values
        :param coordinates: normalized vec, length same as self.vec_length
        :return: a list of variable values
        """
        coordinates = np.array(coordinates)
        if min(coordinates) < 0 or max(coordinates) > 1:
            raise ValueError("coordinates must be scaled to unit length")
        parameter_values = []
        c = 0
        for p in range(len(self.parameter_names)):
            if type(self.parameter_range[p]) == tuple:
                name, opt = self.undo_one_hot_encoding(coordinates, c)
                parameter_values.append(opt)
                c += len(self.parameter_range[p])
            else:
                parameter_values.append((self.parameter_range[p][1] - self.parameter_range[p][0]) * coordinates[c]
                                        + self.parameter_range[p][0])
                c += 1
        return parameter_values
    
    def input_data_from_coordinates(self, coordinates):
        """
        From a normalized coordinate vector, get a dictionary of input parameters
        :param coordinates: normalized vec, length same as self.vec_length
        :return: dict where k:v = system: model: group: variable: value
        """
        sample = self.values_from_coordinates(coordinates)
        inputs = dict()
        
        for i in range(len(self.parameter_names)):
            var_keys = self.parameter_map[i]
            if var_keys[0] not in inputs.keys():
                inputs[var_keys[0]] = dict()
            if var_keys[1] not in inputs[var_keys[0]].keys():
                inputs[var_keys[0]][var_keys[1]] = dict()
            
            inputs_model = inputs[var_keys[0]][var_keys[1]]
            if var_keys[2] not in inputs_model.keys():
                inputs_model[var_keys[2]] = dict()
            
            inputs_model[var_keys[2]][var_keys[3]] = sample[i]
        return inputs
    
    def setup_single(self, coordinates):
        """
        Assign the values of the system simulations
        :param coordinates: normalized vec, length same as self.vec_length
        """
        sample_params = self.values_from_coordinates(coordinates)
        i = 0
        while i < len(sample_params):
            param = self.parameter_map[i]
            system = self.systems[param[0]]
            model = system[param[1]]
            group = model.__getattribute__(param[2])
            if type(self.parameter_range[i]) == tuple:
                name = self.parameter_map[i][3].split('-')[0]
                group.__setattr__(name, sample_params[i])
            else:
                group.__setattr__(param[3], sample_params[i])
            i += 1
    
    def run_single(self, get_output=True):
        """
        Run a scenario.

        Note 1:
        Currently SAM is set to solve for the PPA price required to achieve an 11% IRR in year 20
        But realistically, energy markets will not support this PPA price, so we'll probably want to switch
        to a mode where SAM solves for the IRR given a reasonable PPA price, and then add any capacity value of the plant
        """
        
        for system_name, models in self.systems.items():
            self.system_behavior[system_name](self.systems)

        output_vec = []
        for output_info in self.outputs_map:
            model = self.systems[output_info[0]][output_info[1]]
            try:
                output_vec.append(model.__getattribute__("Outputs").__getattribute__(output_info[2]))
            # have to do this because battery model gen output appears to be going into "System" group
            except AttributeError:
                output_vec.append(model.__getattribute__("System").__getattribute__(output_info[2]))

        return output_vec

    def output_values(self, output_vec):
        """
        Translates the output vector to a dictionary of variable:value pairs
        :param output_vec: same dimension as outputs_map
        :return: dictionary
        """
        output_dict = dict()
        for i in range(len(self.outputs_map)):
            output_info = self.outputs_map[i]
            
            if not output_info[0] in output_dict:
                output_dict[output_info[0]] = dict()
            output_dict[output_info[0]][output_info[2]] = output_vec[i]
        return output_dict

def run_default_scenario(defaults, input_info, output_info, run_systems, print_status=True, save_data=True, filename=None):
    """
    Run default scenario and print outputs
    """
    default_scenario = Scenario(defaults, input_info, output_info, run_systems)
    output_vec = default_scenario.run_single()
    outputs = default_scenario.output_values(output_vec)
    
    if print_status:
        print("Running Default Scenario:\n=========================\n")
        print('Outputs:')
        print_output_vals(outputs)
        
        if 'Solar' in run_systems:
            solar_extent = calculate_solar_extent(defaults['Solar']['Pvsamv1'])
            print('Solar extent: %s' % (solar_extent,) + ' meters')

    if save_data:
        if filename is None:
            import os
            path = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(path, '..', 'results', 'scenario.csv')
            save_output_array_vals(outputs, filename)

    return default_scenario, outputs
