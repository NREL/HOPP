import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ctypes import *
c_number = c_double  # must be c_double or c_float depending on how defined in sscapi.h
import abc
import importlib
import copy

PYSAM_MODULE_NAME = 'PySAM_DAOTk'
# PYSAM_MODULE_NAME = 'PySAM'
SSCDLL_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "libs", "ssc.dll")
# SSCDLL_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "libs", "sscd.dll")

# SSCDLL_PATH = os.path.join(os.environ.get('SAMNTDIR'), 'deploy/x64/ssc.dll')             # release
# SSCDLL_PATH = os.path.join(os.environ.get('SAMNTDIR'),'deploy/x64/sscd.dll')            # debug

def ssc_wrap(wrapper, tech_name, financial_name, defaults_name=None, defaults=None):
    """Factory method for ssc wrappers
    Returns an SscWrap object for an ssc interface via either PySSC or PySAM
    """
    #TODO:  Flatten the dictionaries returned by PysamWrap::execute and PysamWrap::export_params
    #TODO:  Add a function to replace the defaults parameter. If there's a defaults_name specified and
    #       wrapper is pyssc, create a PysamWrap first and run export_params.

    if wrapper == 'pyssc':
        return PysscWrap(tech_name, financial_name, defaults)
    elif wrapper == 'pysam':
        return PysamWrap(tech_name, financial_name, defaults_name)


class SscWrap(metaclass=abc.ABCMeta):
    """Abstract parent class of ssc wrappers
    Includes wrapper functions for calls to ssc (any models)
    """
    @abc.abstractmethod
    def set(self, param_dict):
        return

    @abc.abstractmethod
    def get(self, name):
        return

    @abc.abstractmethod
    def execute(self):
        return

    @abc.abstractmethod
    def export_params(self):
        return


class PysscWrap(SscWrap):
    def __init__(self, tech_name, financial_name, defaults=None):
        self.ssc = PySSC()
        self.wrapper = 'pyssc'
        self.tech_name = tech_name
        self.financial_name = financial_name
        if defaults is not None:
            self.defaults = copy.deepcopy(defaults)
            self.params = copy.deepcopy(self.defaults)
        else:
            self.defaults = {}
            self.params = {}
        self.params['tech_model'] = self.tech_name
        self.params['financial_model'] = self.financial_name
	
    def set(self, param_dict):
        if 'is_elec_heat_dur_off' in param_dict and type(param_dict['is_elec_heat_dur_off']) == list:
            param_dict['is_elec_heat_dur_off'] = param_dict['is_elec_heat_dur_off'][0]

        self.params.update(param_dict)

    def get(self, name):
        return self.params[name]

    def execute(self):
        results = ssc_sim_from_dict(self.ssc, self.params)
        return results

    def export_params(self):
        return copy.deepcopy(self.params)

    def create_lk_inputs_file(self, filename: str, weather_file):
        file = open(filename, "w")
        file.write("clear();\n")
        # Handling weather file differently due to the solar_resource_data
        clean_path = str(os.path.normpath(weather_file))
        solar_file_map = {'tcsmolten_salt': 'solar_resource_file', 'trough_physical': 'file_name'}
        solar_file_name = solar_file_map[self.params['tech_model']]
        file.write("var( '" + solar_file_name + "', '" + clean_path.replace('\\', '/') + "' );\n")

        # Inputs
        for key, value in self.params.items():
            if key == 'tech_model' or key == 'financial_model':
                continue
            elif any([type(value) is scalar_type for scalar_type in [int, float, str]]):
                file.write("var( '" + key + "', " + str(value) + " );\n")
            elif type(value) is bool:
                if value:
                    file.write("var( '" + key + "', " + str(1) + " );\n")
                else:
                    file.write("var( '" + key + "', " + str(0) + " );\n")
            elif type(value) is list:
                if type(value[0]) is list:
                    file.write("var( '" + key + "', \n")
                    file.write("[ ")
                    # Matrix
                    for row_count, row in enumerate(value):
                        file.write("[ ")
                        for item_count, item in enumerate(row):
                            suffix = ", "
                            if item_count == len(row) - 1:
                                suffix = " ],\n"
                                if row_count == len(value) - 1:
                                    suffix = " ] ] );\n"
                            file.write(str(item) + suffix)
                else:
                    file.write("var( '" + key + "', ")
                    file.write("[ ")
                    # List
                    for item_count, item in enumerate(value):
                        suffix = ", "
                        if item_count == len(value) - 1:
                            suffix = " ] );\n"
                        file.write(str(item) + suffix)
        # Run calls
        file.write("run('" + str(self.params['tech_model']) + "');\n")
        if self.params['financial_model'] is not None:
            file.write("run('" + str(self.params['financial_model']) + "');\n")

        # Outputs:
        file.write("\n")
        file.write("outln('Annual energy (year 1) ' + var('annual_energy'));\n")
        file.write("outln('Capacity factor (year 1) ' + var('capacity_factor'));\n")
        file.write("outln('Annual Water Usage ' + var('annual_total_water_use'));\n")
        file.close()


class PysamWrap(SscWrap):
    def __init__(self, tech_name, financial_name, defaults_name=None):
        self.wrapper = 'pysam'
        self.tech_name = tech_name
        self.financial_name = financial_name
        self.defaults_name = defaults_name

        def format_name(model_name):
            if model_name is None:
                return None
            result = model_name.replace('_', ' ')
            result = result.title()
            result = result.replace(' ', '')
            return result

        tech_module = importlib.import_module(PYSAM_MODULE_NAME + '.' + format_name(self.tech_name))
        if defaults_name is not None:
            self.tech_model = tech_module.default(defaults_name)
        else:
            self.tech_model = tech_module.new()
        if financial_name is not None:
            financial_module = importlib.import_module(PYSAM_MODULE_NAME + '.' + format_name(self.financial_name))
            self.financial_model = financial_module.from_existing(self.tech_model, self.defaults_name)

    def set(self, param_dict):
        for key,value in param_dict.items():
            key = key.replace('.', '_')
            key = key.replace('adjust:', '')    # These set the values in tech_model.AdjustmentFactors
            if value == []: value = [0]         # setting an empty list crashes PySAM
            if 'is_elec_heat_dur_off' in param_dict and type(param_dict['is_elec_heat_dur_off']) != list:
                param_dict['is_elec_heat_dur_off'] = [param_dict['is_elec_heat_dur_off']]

            try:
                self.tech_model.value(key, value)
            except Exception as err:
                try:
                    self.financial_model.value(key, value)
                except:
                    print("Cannot set parameter: %s" % key)
                    pass            # continue
        return 0

    def get(self, name):
        key = name.replace('.', '_')
        try:
            return self.tech_model.value(key)
        except Exception as err:
            raise(err)

    def execute(self):
        self.tech_model.execute(1)
        results = self.tech_model.Outputs.export()
        if self.financial_name is not None:
            self.financial_model.execute(1)
            results.update(self.financial_model.Outputs.export())
        return results

    def export_params(self):
        result = self.tech_model.export()
        result.update(self.financial_model.export())
        return result


# TODO: make these few following functions into member functions
# Functions to simulate compute modules through dictionaries
def ssc_sim_from_dict(ssc, data_pydict):
    """ Run a technology compute module using parameters in a dict.

    Parameters
    ----------
    data_pydict: dict
        Required keys are:
            tech_model: str
                name of the compute module to run.
            financial_model: str or None
                name of the financial model to apply. If None, no financial
                model is used.
        Other keys are names of args for the selected tech_model or
        financial_model.

    Returns
    -------
    (dict): dict
        keys are outputs from the selected compute module.
    """
    tech_model_name = data_pydict["tech_model"]
    # Convert python dictionary into ssc var info table
    # ssc = PySSC()
    data_ssc_tech_model = dict_to_ssc_table(ssc, data_pydict, tech_model_name)

    financial_model_name = data_pydict["financial_model"]
    if financial_model_name is None:
        data_ssc = data_ssc_tech_model
    else:
        data_ssc = dict_to_ssc_table_dat(ssc, data_pydict, financial_model_name,
                                         data_ssc_tech_model)

    return ssc_sim(ssc, data_ssc, tech_model_name, financial_model_name)


def ssc_sim(ssc, data_ssc, tech_model_name, financial_model_name):

    # Run the technology model compute module
    tech_model_return = ssc_cmod(ssc, data_ssc, tech_model_name)
    tech_model_success = tech_model_return[0]
    tech_model_dict = tech_model_return[1]

    # Add tech and financial models back to dictionary
    tech_model_dict["tech_model"] = tech_model_name
    tech_model_dict["financial_model"] = financial_model_name

    if (tech_model_success == 0):
        tech_model_dict["cmod_success"] = 0
        return tech_model_dict

    if financial_model_name in [None, "none"]:
        tech_model_dict["cmod_success"] = 1
        return tech_model_dict

    # Run the financial model
    financial_model_return = ssc_cmod(ssc, data_ssc, financial_model_name)
    financial_model_success = financial_model_return[0]
    financial_model_dict = financial_model_return[1]

    if (financial_model_success == 0):
        financial_model_dict["cmod_success"] = 0
        out_err_dict = tech_model_dict.copy()
        return out_err_dict.update(financial_model_dict)

    # If all models are successful, set boolean true
    financial_model_dict["cmod_success"] = 1

    # Combine tech and financial model dictionaries
    out_dict = tech_model_dict.copy()
    out_dict.update(financial_model_dict)

    return out_dict

def ssc_cmod(ssc, dat, name):
    # ssc = PySSC()

    cmod = ssc.module_create(name.encode("utf-8"))
    ssc.module_exec_set_print(0)

    # Run compute module
    # Check for simulation errors
    if ssc.module_exec(cmod, dat) == 0:
        print(name + ' simulation error')
        idx = 1
        msg = ssc.module_log(cmod, 0)
        while msg is not None:
            print(' : ' + msg.decode("utf - 8"))
            msg = ssc.module_log(cmod, idx)
            idx = idx + 1
        cmod_err_dict = ssc_table_to_dict(ssc, cmod, dat)
        return [False, cmod_err_dict]

    # Get python dictionary representing compute module with all inputs/outputs defined
    return [True, ssc_table_to_dict(ssc, cmod, dat)]


def dict_to_ssc_table(ssc, py_dict, cmod_name):
    # ssc = PySSC()
    dat = ssc.data_create()
    return dict_to_ssc_table_dat(ssc, py_dict, cmod_name, dat)


def dict_to_ssc_table_dat(ssc, py_dict, cmod_name, dat):
    # ssc = PySSC()

    cmod = ssc.module_create(cmod_name.encode("utf-8"))

    dict_keys = list(py_dict.keys())
    # dat = ssc.data_create()

    ii = 0
    while (True):

        p_ssc_entry = ssc.module_var_info(cmod, ii)

        ssc_input_data_type = ssc.info_data_type(p_ssc_entry)

        # 1 = String, 2 = Number, 3 = Array, 4 = Matrix, 5 = Table
        if (ssc_input_data_type <= 0 or ssc_input_data_type > 5):
            break

        ssc_input_var_type = ssc.info_var_type(p_ssc_entry)

        # If the variable type is INPUT (1) or INOUT (3)
        if (ssc_input_var_type == 1 or ssc_input_var_type == 3):

            # Get name of iith variable in compute module table
            ssc_input_data_name = str(ssc.info_name(p_ssc_entry).decode("ascii"))

            # Find corresponding 'des_par' dictionary item
            is_str_test_key = False
            for i in range(len(dict_keys)):
                if (dict_keys[i] == ssc_input_data_name):
                    is_str_test_key = True
                    # print ("Found key")
                    break

            # Helpful for debugging:
            # if(is_str_test_key == False):
            #    print ("Did not find key: ", ssc_input_data_name)

            # Set compute module data to dictionary value
            if is_str_test_key:
                set_ssc_var(ssc_input_data_type, ssc, dat, ssc_input_data_name, py_dict[ssc_input_data_name])

        ii = ii + 1

    return dat


def set_ssc_var(ssc_input_data_type, ssc, dat, ssc_input_data_name, value):
    if (ssc_input_data_type == 1):
        ssc.data_set_string(dat, ssc_input_data_name.encode("ascii"),
                            value.encode("ascii"))
    elif (ssc_input_data_type == 2):
        ssc.data_set_number(dat, ssc_input_data_name.encode("ascii"), value)
    elif (ssc_input_data_type == 3):
        if len(value) > 0:
            ssc.data_set_array(dat, ssc_input_data_name.encode("ascii"), value)
    elif (ssc_input_data_type == 4):
        if len(value) > 0:
            ssc.data_set_matrix(dat, ssc_input_data_name.encode("ascii"), value)
    elif (ssc_input_data_type == 5):
        if len(value) > 0:
            table = ssc.data_create()
            for k,v in value.items():
                set_ssc_var(ssc_data_type(v), ssc, table, k, v)
            ssc.data_set_table(dat, ssc_input_data_name.encode("ascii"), table)
            ssc.data_free(table)


# Returns SSC data type of Python data type
def ssc_data_type(v):
    if type(v) is str:
        ssc_data_type = 1       # string
    elif type(v) is int or type(v) is float or type(v) is bool:
        ssc_data_type = 2       # number
    elif type(v) is list:
        if type(v[0]) is list:
            ssc_data_type = 4   # matrix
        else:
            ssc_data_type = 3   # array
    elif type(v) is dict:
        ssc_data_type = 5       # table
    else:
        print("Unknown SSC variable type for " + str(v))
    return ssc_data_type


# Returns python dictionary representing SSC compute module w/ all required inputs/outputs defined
def ssc_table_to_dict(ssc, cmod, dat):
    # ssc = PySSC()
    i = 0
    ssc_out = {}
    while (True):
        p_ssc_entry = ssc.module_var_info(cmod, i)
        ssc_output_data_type = ssc.info_data_type(p_ssc_entry)
        if (ssc_output_data_type <= 0 or ssc_output_data_type > 5):
            break
        ssc_output_data_name = str(ssc.info_name(p_ssc_entry).decode("ascii"))
        ssc_data_query = ssc.data_query(dat, ssc_output_data_name.encode("ascii"))
        if (ssc_data_query > 0):
            if (ssc_output_data_type == 1):
                ssc_out[ssc_output_data_name] = ssc.data_get_string(dat,
                                                                    ssc_output_data_name.encode("ascii")).decode(
                    "ascii")
            elif (ssc_output_data_type == 2):
                ssc_out[ssc_output_data_name] = ssc.data_get_number(dat, ssc_output_data_name.encode("ascii"))
            elif (ssc_output_data_type == 3):
                ssc_out[ssc_output_data_name] = ssc.data_get_array(dat, ssc_output_data_name.encode("ascii"))
            elif (ssc_output_data_type == 4):
                ssc_out[ssc_output_data_name] = ssc.data_get_matrix(dat, ssc_output_data_name.encode("ascii"))
            elif (ssc_output_data_type == 5):
                ssc_out[ssc_output_data_name] = ssc.data_get_table(dat, ssc_output_data_name.encode("ascii"))
        i = i + 1

    ssc.data_free(dat)
    ssc.module_free(cmod)
    return ssc_out

#TODO: verify darwin and linux paths work
class PySSC:
    def __init__(self):
        this_directory = os.path.abspath(os.path.dirname(__file__))

        if sys.platform == 'win32' or sys.platform == 'cygwin':
            # self.pdll = CDLL(os.path.join(this_directory, "ssc.dll"))
            # self.pdll = CDLL(os.path.join(os.environ.get('SAMNTDIR'),'deploy/x64/sscd.dll'))
            self.pdll = CDLL(SSCDLL_PATH)
        elif sys.platform == 'darwin':
            self.pdll = CDLL(os.path.join(this_directory, "libs", "ssc.dylib"))
        elif sys.platform == 'linux':
            self.pdll = CDLL(os.path.join(this_directory, "libs", 'libssc.so'))
        else:
            print('Platform not supported ', sys.platform)
        print('Process ID = ' + str(os.getpid()))       # attach to process will not work until after the above CDLL() call
        pass

    INVALID = 0
    STRING = 1
    NUMBER = 2
    ARRAY = 3
    MATRIX = 4
    INPUT = 1
    OUTPUT = 2
    INOUT = 3

    def version(self):
        self.pdll.ssc_version.restype = c_int
        return self.pdll.ssc_version()

    def data_create(self):
        self.pdll.ssc_data_create.restype = c_void_p
        return self.pdll.ssc_data_create()

    def data_free(self, p_data):
        self.pdll.ssc_data_free(c_void_p(p_data))

    def data_clear(self, p_data):
        self.pdll.ssc_data_clear(c_void_p(p_data))

    def data_unassign(self, p_data, name):
        self.pdll.ssc_data_unassign(c_void_p(p_data), c_char_p(name))

    def data_query(self, p_data, name):
        self.pdll.ssc_data_query.restype = c_int
        return self.pdll.ssc_data_query(c_void_p(p_data), c_char_p(name))

    def data_first(self, p_data):
        self.pdll.ssc_data_first.restype = c_char_p
        return self.pdll.ssc_data_first(c_void_p(p_data))

    def data_next(self, p_data):
        self.pdll.ssc_data_next.restype = c_char_p
        return self.pdll.ssc_data_next(c_void_p(p_data))

    def data_set_string(self, p_data, name, value):
        self.pdll.ssc_data_set_string(c_void_p(p_data), c_char_p(name), c_char_p(value))

    def data_set_number(self, p_data, name, value):
        self.pdll.ssc_data_set_number(c_void_p(p_data), c_char_p(name), c_number(value))

    def data_set_array(self, p_data, name, parr):
        count = len(parr)
        arr = (c_number * count)()
        arr[:] = parr  # set all at once instead of looping
        return self.pdll.ssc_data_set_array(c_void_p(p_data), c_char_p(name), pointer(arr), c_int(count))

    def data_set_array_from_csv(self, p_data, name, fn):
        f = open(fn, 'rb')
        data = []
        for line in f:
            data.extend([n for n in map(float, line.split(b','))])
        f.close()
        return self.data_set_array(p_data, name, data)

    def data_set_matrix(self, p_data, name, mat):
        nrows = len(mat)
        ncols = len(mat[0])
        size = nrows * ncols
        arr = (c_number * size)()
        idx = 0
        for r in range(nrows):
            for c in range(ncols):
                arr[idx] = c_number(mat[r][c])
                idx = idx + 1
        return self.pdll.ssc_data_set_matrix(c_void_p(p_data), c_char_p(name), pointer(arr), c_int(nrows), c_int(ncols))

    def data_set_matrix_from_csv(self, p_data, name, fn):
        f = open(fn, 'rb')
        data = []
        for line in f:
            lst = ([n for n in map(float, line.split(b','))])
            data.append(lst)
        f.close()
        return self.data_set_matrix(p_data, name, data)

    def data_set_table(self, p_data, name, tab):
        return self.pdll.ssc_data_set_table(c_void_p(p_data), c_char_p(name), c_void_p(tab))

    def data_get_string(self, p_data, name):
        self.pdll.ssc_data_get_string.restype = c_char_p
        return self.pdll.ssc_data_get_string(c_void_p(p_data), c_char_p(name))

    def data_get_number(self, p_data, name):
        val = c_number(0)
        self.pdll.ssc_data_get_number(c_void_p(p_data), c_char_p(name), byref(val))
        return val.value

    def data_get_array(self, p_data, name):
        count = c_int()
        self.pdll.ssc_data_get_array.restype = POINTER(c_number)
        parr = self.pdll.ssc_data_get_array(c_void_p(p_data), c_char_p(name), byref(count))
        arr = parr[0:count.value]  # extract all at once
        return arr

    def data_get_matrix(self, p_data, name):
        nrows = c_int()
        ncols = c_int()
        self.pdll.ssc_data_get_matrix.restype = POINTER(c_number)
        parr = self.pdll.ssc_data_get_matrix(c_void_p(p_data), c_char_p(name), byref(nrows), byref(ncols))
        idx = 0
        mat = []
        for r in range(nrows.value):
            row = []
            for c in range(ncols.value):
                row.append(float(parr[idx]))
                idx = idx + 1
            mat.append(row)
        return mat

    # don't call data_free() on the result, it's an internal
    # pointer inside SSC
    def data_get_table(self, p_data, name):
        return self.pdll.ssc_data_get_table(c_void_p(p_data), name)

    def module_entry(self, index):
        self.pdll.ssc_module_entry.restype = c_void_p
        return self.pdll.ssc_module_entry(c_int(index))

    def entry_name(self, p_entry):
        self.pdll.ssc_entry_name.restype = c_char_p
        return self.pdll.ssc_entry_name(c_void_p(p_entry))

    def entry_description(self, p_entry):
        self.pdll.ssc_entry_description.restype = c_char_p
        return self.pdll.ssc_entry_description(c_void_p(p_entry))

    def entry_version(self, p_entry):
        self.pdll.ssc_entry_version.restype = c_int
        return self.pdll.ssc_entry_version(c_void_p(p_entry))

    def module_create(self, name):
        self.pdll.ssc_module_create.restype = c_void_p
        return self.pdll.ssc_module_create(c_char_p(name))

    def module_free(self, p_mod):
        self.pdll.ssc_module_free(c_void_p(p_mod))

    def module_var_info(self, p_mod, index):
        self.pdll.ssc_module_var_info.restype = c_void_p
        return self.pdll.ssc_module_var_info(c_void_p(p_mod), c_int(index))

    def info_var_type(self, p_inf):
        return self.pdll.ssc_info_var_type(c_void_p(p_inf))

    def info_data_type(self, p_inf):
        return self.pdll.ssc_info_data_type(c_void_p(p_inf))

    def info_name(self, p_inf):
        self.pdll.ssc_info_name.restype = c_char_p
        return self.pdll.ssc_info_name(c_void_p(p_inf))

    def info_label(self, p_inf):
        self.pdll.ssc_info_label.restype = c_char_p
        return self.pdll.ssc_info_label(c_void_p(p_inf))

    def info_units(self, p_inf):
        self.pdll.ssc_info_units.restype = c_char_p
        return self.pdll.ssc_info_units(c_void_p(p_inf))

    def info_meta(self, p_inf):
        self.pdll.ssc_info_meta.restype = c_char_p
        return self.pdll.ssc_info_meta(c_void_p(p_inf))

    def info_group(self, p_inf):
        self.pdll.ssc_info_group.restype = c_char_p
        return self.pdll.ssc_info_group(c_void_p(p_inf))

    def info_uihint(self, p_inf):
        self.pdll.ssc_info_uihint.restype = c_char_p
        return self.pdll.ssc_info_uihint(c_void_p(p_inf))

    def info_required(self, p_inf):
        self.pdll.ssc_info_required.restype = c_char_p
        return self.pdll.ssc_info_required(c_void_p(p_inf))

    def info_constraints(self, p_inf):
        self.pdll.ssc_info_constraints.restype = c_char_p
        return self.pdll.ssc_info_constraints(c_void_p(p_inf))

    def module_exec(self, p_mod, p_data):
        self.pdll.ssc_module_exec.restype = c_int
        return self.pdll.ssc_module_exec(c_void_p(p_mod), c_void_p(p_data))

    def module_exec_simple_no_thread(self, modname, data):
        self.pdll.ssc_module_exec_simple_nothread.restype = c_char_p
        return self.pdll.ssc_module_exec_simple_nothread(c_char_p(modname), c_void_p(data))

    def module_log(self, p_mod, index):
        log_type = c_int()
        time = c_float()
        self.pdll.ssc_module_log.restype = c_char_p
        return self.pdll.ssc_module_log(c_void_p(p_mod), c_int(index), byref(log_type), byref(time))

    def module_exec_set_print(self, prn):
        return self.pdll.ssc_module_exec_set_print(c_int(prn))


###########################################################################################
if __name__ == '__main__':
    # # Run PySAM normally
    # import PySAM.TcsmoltenSalt as t
    # import PySAM.Singleowner as s
    # model_name = 'MSPTSingleOwner'
    # tech_model = t.default(model_name)
    # tech_model.SolarResource.solar_resource_file = weather_file
    # financial_model = s.from_existing(tech_model, model_name)       # must be called before tech_model.execute(1) !!
    # tech_model.execute(1)
    # tech_attributes = tech_model.export()                           # not necessary
    # results_pysam = tech_model.Outputs.export()
    # financial_model.execute(1)
    # financial_attributes = financial_model.export()                 # not necessary
    # results_pysam.update(financial_model.Outputs.export())

    # # Run PySSC normally
    # from PySAM import PySSC
    # from data.mspt_2021_develop_defaults import default_ssc_params  # exported from latest open-source develop as JSON inputs
    # import copy
    # default_params = copy.deepcopy(default_ssc_params)
    # default_params['solar_resource_file'] = weather_file
    # default_params['tech_model'] = 'tcsmolten_salt'
    # default_params['financial_model'] = 'singleowner'
    # results_pyssc = PySSC.ssc_sim_from_dict(default_params)

    # # Run PySAM through wrap
    # ssc = ssc_wrap(
    #     wrapper='pysam',
    #     tech_name='tcsmolten_salt',
    #     financial_name='singleowner',
    #     defaults_name='MSPTSingleOwner',
    #     defaults=default_ssc_params         # not applicable here but doesn't hurt
    #     )
    # ssc.set({'solar_resource_file': weather_file})
    # results_pysam_wrap = ssc.execute()

    # Run PySSC through wrap
    # ssc = ssc_wrap(
    #     wrapper='pyssc',
    #     tech_name='tcsmolten_salt',
    #     financial_name='singleowner',
    #     defaults_name='MSPTSingleOwner',    # not applicable here but doesn't hurt
    #     defaults=default_ssc_params
    #     )
    # ssc.set({'solar_resource_file': weather_file})
    # results_pyssc_wrap = ssc.execute()

    # print("PySAM annual energy = {:.3e} kWh".format(results_pysam['annual_energy']))
    # print("PySAM PPA price = {:.2f} cents/kWh".format(results_pysam['ppa_price']))
    # ##
    # print("PySSC annual energy = {:.3e} kWh".format(results_pyssc['annual_energy']))
    # print("PySSC PPA price = {:.2f} cents/kWh".format(results_pyssc['ppa_price']))
    # ##
    # print("PySAM-wrap annual energy = {:.3e} kWh".format(results_pysam_wrap['annual_energy']))
    # print("PySAM-wrap PPA price = {:.2f} cents/kWh".format(results_pysam_wrap['ppa_price']))
    # ##
    # print("PySSC-wrap annual energy = {:.3e} kWh".format(results_pyssc_wrap['annual_energy']))
    # print("PySSC-wrap PPA price = {:.2f} cents/kWh".format(results_pyssc_wrap['ppa_price']))

    pass
