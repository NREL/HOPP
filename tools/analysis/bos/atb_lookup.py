from scipy.interpolate import LinearNDInterpolator as interp
from pathlib import Path
import pandas as pd
import numpy as np

from .bos_model import BOSCalculator
from hybrid.log import bos_logger as logger

file_path = Path(__file__).parent


class ATBLookup:
    def __init__(self):
        super().__init__()
        self.name = "ATBLookup"

        self.input_parameters = ["Year",
                                 "Scenario"]

        # List of desired output parameters from the JSON lookup
        self.desired_output_parameters = ["Class 1 Wind - Advanced", "Class 1 Wind - Moderate",
                                          "Class 1 Wind - Conservative", "Utility PV - Advanced",
                                          "Utility PV - Moderate", "Utility PV - Conservative",
                                          "BESS kwh - Advanced", "BESS kwh - Moderate", "BESS kwh - Conservative",
                                          "BESS kw - Advanced", "BESS kw - Moderate", "BESS kw - Conservative"]

        # Loads the json data containing all the ATB cost information from json file
        self.atb_data, self.contents = self._load_lookup()

        for p in self.desired_output_parameters:
            if p not in self.atb_data:
                raise KeyError(p + " column missing")

    def _load_lookup(self):
        import json
        file = file_path / "ATBCosts2020.json"
        with open(file, 'r') as f:
            atb_cost_data = json.loads(f.read())
        contents = atb_cost_data#[self.input_parameters].values
        return atb_cost_data, contents

    def _load_interp(self):
        fxns = []
        for p in self.desired_output_parameters:
            f = interp(self.contents, self.data[p].values)
            fxns.append(f)
        return fxns

    def _lookup_costs(self, year, scenario):
        if year < 2018:
            year = 2018

        wind_cost_mw = self.atb_data['Class 1 Wind - {}'.format(scenario)][str(year)] * 1000
        solar_cost_mw = self.atb_data['Utility PV - {}'.format(scenario)][str(year)] * 1000
        storage_cost_mw = self.atb_data['BESS kw - {}'.format(scenario)][str(year)] * 1000
        storage_cost_mwh = self.atb_data['BESS kwh - {}'.format(scenario)][str(year)] * 1000
        # print(wind_cost_mw, solar_cost_mw, storage_cost_mw, storage_cost_mwh)
        return wind_cost_mw, solar_cost_mw, storage_cost_mw, storage_cost_mwh

    def calculate_atb_costs(self, year, scenario='Moderate'):
        """
        Calls the appropriate calculate_bos_costs_x method for the Cost Source data specified

        :param year: ATB scenario year (2018-2050)
        :param scenario: ATB scenario (Advanced, Moderate, Conservative)
        :return: wind, solar, storage cost per mw and mwh
        """
        if scenario == 'Advanced' or 'Moderate' or 'Conservative':
            return self._lookup_costs(year, scenario)
        else:
            raise ValueError("scenario type {} not recognized".format(scenario))
