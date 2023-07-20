from dotenv import load_dotenv, find_dotenv
import os

developer_nrel_gov_key = ""


def set_developer_nrel_gov_key(key: str):
    global developer_nrel_gov_key
    developer_nrel_gov_key = key


def get_developer_nrel_gov_key():
    global developer_nrel_gov_key
    if developer_nrel_gov_key is None or len(developer_nrel_gov_key) != 40:
        raise ValueError("Please provide NREL Developer key using `set_developer_nrel_gov_key`"
                         "(`from hopp.keys import set_developer_nrel_gov_key`) \n"
                         " Ensure your Developer key is set either as a `NREL_API_KEY` Environment Variable or"
                         " using the .env file method. For details on setting up .env, "
                         "please see Section 7 of 'Installing from Source' or "
                         "Section 2 of 'Installing from Package Repositories' in the README.md")
    return developer_nrel_gov_key


def set_nrel_key_dot_env(path=None):
    if path and os.path.exists(path):
        load_dotenv(path)
    else:
        r = find_dotenv(usecwd=True)
        load_dotenv(r)
    NREL_API_KEY = os.getenv("NREL_API_KEY")
    if NREL_API_KEY is not None:
        set_developer_nrel_gov_key(NREL_API_KEY)
