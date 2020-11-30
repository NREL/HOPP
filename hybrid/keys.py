developer_nrel_gov_key = ""


def set_developer_nrel_gov_key(key: str):
    global developer_nrel_gov_key
    developer_nrel_gov_key = key


def get_developer_nrel_gov_key():
    global developer_nrel_gov_key
    if len(developer_nrel_gov_key) == 0:
        raise ValueError("Please provide NREL Developer key. `from hybrid.keys import set_developer_nrel_gov_key`")
    return developer_nrel_gov_key
