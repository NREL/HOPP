import json
import os
import requests

from hopp.utilities.keys import get_developer_nrel_gov_key

URDB_BASE_URL = "https://api.openei.org/utility_rates"

class UtilityRate:
    """
    Class to define a utility rate and interact with the Utility Rate Database (URDB)
    https://api.openei.org/utility_rates?version=7&format=json&detail=full&getpage={urdb_label}&api_key={api_key}'
    """
    def __init__(self, path_rates, urdb_label):
        self.path_rates = path_rates
        self.urdb_label = urdb_label
        self.api_key = get_developer_nrel_gov_key()

    @property
    def urdb_url(self):
        return f"{URDB_BASE_URL}?version=7&format=json&detail=full&getpage={self.urdb_label}&api_key={self.api_key}"

    def get_urdb_response(self):
        file_exists = False
        if self.path_rates:
            file_urdb_json = os.path.join(self.path_rates, str(self.urdb_label) + '.json')
            file_exists = os.path.exists(file_urdb_json)
        results = None
        if not file_exists and self.urdb_label is not None:
            # since NREL can't figure out its certificate
            resp = requests.get(self.urdb_url, verify=False)

            if resp.ok:
                results = json.loads(resp.text, strict=False)
                results = results['items'][0]
                if self.path_rates:
                    with open(file_urdb_json, 'w') as fp:
                        json.dump(obj=results, fp=fp)
                self.urdb_response = results
        elif file_exists:
            with open(file_urdb_json, 'r') as fp:
                results = json.load(fp=fp)
        self.results = results
        return results
