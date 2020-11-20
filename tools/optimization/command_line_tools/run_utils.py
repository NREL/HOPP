import errno
import json
import os
import sys
from datetime import datetime
from typing import Optional

from . import config_tools


def makedir_if_not_exists(filename: str) -> None:
    try:
        os.makedirs(filename)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_run(
        default_config: {},
        output_path: Optional[str] = None,
        run_suffix: Optional[str] = None,
        place_in_subdir: bool = True,
        output_dir_name: str = 'log',
        ) -> ({}, str):
    '''
    Sets up a config and logging prefix for this run. Writes the config to a log file.
    :param default_config: the base configuration that the command line params override
    :param output_path: path to place output files including logs and configuration record. If None, './output' is used.
    :param run_suffix: appended to the run name. If None, then the ISO 8601 datetime is used with '.' instead of ':'.
    :param place_in_subdir: True to output into a subdir of output_path, False to directly into output_path
    :return: config, output_path, run_name
    '''
    
    config = config_tools.parse_config_from_args(sys.argv[1:], default_config)
    run_name = config['name']
    run_suffix = '_' + datetime.now().isoformat().replace(':', '.') if run_suffix is None else run_suffix
    run_name += run_suffix
    
    output_path = os.path.join(os.path.curdir, output_dir_name) if output_path is None else output_path
    output_path = output_path if place_in_subdir and run_name is None else os.path.join(output_path, run_name)
    makedir_if_not_exists(output_path)
    
    print('setup_run() run_name: "' + run_name + '"')
    print('setup_run() output_path: "' + output_path + '"')
    # print('setup_run() config:')
    # pprint(config)
    
    write_config_log(config, output_path, run_name)
    print('setup_run() complete.')
    return config, output_path, run_name


def write_config_log(
        config: {},
        output_path: str,
        run_name: str,
        config_name: str = 'config') -> None:
    '''
    Writes a json log file containing the configuration of this run
    :param output_path: where to write the file
    :param run_name: prefix of the filename
    :param config_name: suffix of the filename
    '''
    # config_filename = os.path.join(output_path, run_name + config_name + '.json')
    config_filename = os.path.join(output_path, config_name + '.json')
    with open(config_filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2, sort_keys=True)

# def setup_sub_run(default_config: {}) -> ({}, DataRecorder):
#     '''
#     Sets up a config, logging prefix, and recorder for this run.
#     :param default_config:
#     :return:
#     '''
#     config = command_line_config.parse_config_from_args(sys.argv[1:], default_config)
#     pprint(config)
#     run_prefix = get_run_prefix(config, None)
#     write_config_log(config, run_prefix)
#     recorder = make_data_recorder(run_prefix)
#     return config, run_prefix, recorder
#
# def setup_run(default_config: {}, subrun_name: Optional[str] = None):
#     '''
#     Sets up a config and logging prefix for this run. Writes the config to a log file.
#     :param default_config:
#     :param subrun_name:
#     :return:
#     '''
#     config = config_tools.parse_config_from_args(sys.argv[1:], default_config)
#     pprint(config)
#     run_prefix = get_run_prefix(config, subrun_name)
#     write_config_log(config, run_prefix)
#     return config, run_prefix
#
#
# def make_data_recorder(run_prefix: str) -> DataRecorder:
#     '''
#     Makes a DataRecorder for logging this run
#     :param run_prefix: the run log file prefix
#     :return: a DataRecorder for this run
#     '''
#     log_filename = os.path.join(log_path, run_prefix + 'log' + '.jsonl')
#     return DataRecorder(JSONLinesRecordLogger(log_filename))
#
#
# def write_config_log(config: {}, run_prefix: str, suffix: str = 'config') -> None:
#     '''
#     Writes a json log file containing the configuration of this run
#     :param config: run config
#     :param run_prefix: run prefix
#     '''
#     config_filename = os.path.join(log_path, run_prefix + suffix + '.json')
#     with open(config_filename, 'w', encoding='utf-8') as f:
#         json.dump(config, f, ensure_ascii=False, indent=2, sort_keys=True)
#
#
# def get_run_prefix(config: {}, subrun_name: Optional[str] = None) -> str:
#     '''
#     Makes the run log file prefix string by concatenating the run name, subrun name (if it exists), and the current
#     time
#     :param config: run config
#     :param run_prefix: run prefix
#     :return: run log file prefix
#     '''
#     start_time = int(time.time() * 10000)
#     subrun_name = '' if subrun_name is None else '_' + subrun_name
#     return config['name'] + subrun_name + '_' + str(start_time) + '_'

# def plot_from_logger(recorder, x_axis, y_axis):
#     plt.figure()
#     plt.plot(recorder.get_column(x_axis), recorder.get_column(y_axis))
#     print(x_axis, recorder.get_column(x_axis))
#     print(y_axis, recorder.get_column(y_axis))
#     plt.xlabel(x_axis, fontsize=15)
#     plt.ylabel(y_axis, fontsize=15)
