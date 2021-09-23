from dotenv import load_dotenv, find_dotenv
import os

developer_nrel_gov_key = ""


def set_developer_nrel_gov_key(key: str):
    global developer_nrel_gov_key
    developer_nrel_gov_key = key


def get_developer_nrel_gov_key():
    global developer_nrel_gov_key
    if len(developer_nrel_gov_key) != 40:
        raise ValueError("Please provide NREL Developer key using `set_developer_nrel_gov_key`"
                         "(`from hybrid.keys import set_developer_nrel_gov_key`) \n"
                         " - or ensure your Developer key is set using the .env file method."
                         " For details on how to do this, "
                         "please see Section 7 of Readme.md")
    return developer_nrel_gov_key


def set_nrel_key_dot_env(path=None):
    if path:
        load_dotenv(path)
    else:
        r = load_dotenv()
        print(r)
    NREL_API_KEY = os.getenv("NREL_API_KEY")
    print("API Key: ".format(NREL_API_KEY))
    set_developer_nrel_gov_key(NREL_API_KEY)