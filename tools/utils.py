def flatten_dict(d):
    def get_key_values(d):
        for key, value in d.items():
            if isinstance(value, dict):
                yield from get_key_values(value)
            else:
                yield key, value

    return {key:value for (key,value) in get_key_values(d)}
