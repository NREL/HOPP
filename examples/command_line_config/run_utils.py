import json
import sys
import time
from pprint import pprint

import matplotlib.pyplot as plt

from examples.command_line_config import command_line_config
from optimization.data_logging.JSON_lines_record_logger import JSONLinesRecordLogger
from optimization.data_logging.data_recorder import DataRecorder


def setup_run(default_config: {}) -> ({}, DataRecorder):
    config = command_line_config.parse_config_from_args(sys.argv, default_config)
    pprint(config)
    
    start_time = int(time.time() * 10000)
    run_name = config['run_name'] + '_' + str(start_time) + '_'
    log_filename = run_name + 'log' + '.jsonl'
    config_filename = run_name + 'config' + '.json'
    with open(config_filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2, sort_keys=True)
    
    recorder = DataRecorder(JSONLinesRecordLogger(log_filename))
    return config, recorder


def plot_from_logger(recorder, x_axis, y_axis):
    plt.figure()
    plt.plot(recorder.get_column(x_axis), recorder.get_column(y_axis))
    print(x_axis, recorder.get_column(x_axis))
    print(y_axis, recorder.get_column(y_axis))
    plt.xlabel(x_axis, fontsize=15)
    plt.ylabel(y_axis, fontsize=15)
