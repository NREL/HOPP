import json
from copy import deepcopy


def parse_config_from_args(
        args: [str],
        default_config: {},
        ) -> {}:
    """
    Makes a configuration map given a list of (command line) override args and a default configuration
    """
    config = deepcopy(default_config)
    
    for i in range(int((len(args) - 1) / 2)):
        base = 1 + 2 * i
        option = args[base]
        value = args[base + 1]
        
        target = config
        keys = option.split('.')
        for j in range(len(keys) - 1):
            target = target[keys[j]]
        
        key = keys[-1]
        
        if key == '':
            # if the key sequence ends in '.', try to parse the value as JSON
            values = json.loads(value)
            merge_configs(target, values)
        else:
            # try to convert the value to the same type as in the default config
            if key in target:
                default = target[key]
                if isinstance(default, bool):
                    value = bool(value)
                elif isinstance(default, int):
                    value = int(value)
                elif isinstance(default, float):
                    value = float(value)
            target[key] = value
    return config


def merge_configs(target, overrides):
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if key in target:
                target[key] = merge_configs(target[key], value)
            else:
                target[key] = value
        return target
    else:
        return overrides
