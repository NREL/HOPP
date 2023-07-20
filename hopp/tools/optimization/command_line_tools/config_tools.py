import json
import re
from copy import deepcopy


def parse_config_from_args(
        args: [str],
        default_config: {},
        ) -> {}:
    """
    Makes a configuration map given a list of (command line) override args and a default configuration
    """
    config = deepcopy(default_config)
    
    arg = ''.join(args).strip()
    print('c1: ', arg)
    
    # add enclosing brackets if they are missing
    if not arg.startswith('{'):
        arg = '{' + arg + '}'
    
    print('c2: ', arg)
    
    # convert bare true, false, and null's to lowercase
    arg = re.sub(r'(?i)(?<=[\t :{},])["\']?(true|false|null)["\']?(?=[\t :{},])',
                 lambda match: match.group(0).lower(),
                 arg)
    
    print('c4: ', arg)
    
    # replace bare or single quoted strings with quoted ones
    arg = re.sub(
        r'(?<=[\t :{},])["\']?(((?<=")((?!(?<!\\)").)*(?="))|(?<=\')((?!\').)*(?=\')|(?!(true|false|null).)(['
        r'a-zA-Z_][a-zA-Z_0-9]*))["\']?(?=[\t :{},])',
        r'"\1"',
        arg)
    
    print('c5: ', arg)
    
    overrides = json.loads(arg)
    config = merge_configs(config, overrides)
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
