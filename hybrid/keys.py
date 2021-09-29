from dotenv import load_dotenv
import os

developer_nrel_gov_key = ""


def set_developer_nrel_gov_key(key: str):
    global developer_nrel_gov_key
    developer_nrel_gov_key = key


def get_developer_nrel_gov_key():
    global developer_nrel_gov_key
    if len(developer_nrel_gov_key) != 40:
        raise ValueError("Please provide NREL Developer key using `set_developer_nrel_gov_key`. "
                         "`from hybrid.keys import set_developer_nrel_gov_key`")
    return developer_nrel_gov_key


def set_nrel_key_dot_env(path=None):
    if path:
        load_dotenv(path)
    else:
        load_dotenv()
    NREL_API_KEY = os.getenv("NREL_API_KEY")
    set_developer_nrel_gov_key(NREL_API_KEY)
