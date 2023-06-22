from __future__ import annotations
from pathlib import Path
from typing import Union

from hopp.simulation.hopp import Hopp


class HoppInterface():
    def __init__(self, configuration: dict | str | Path):
        self.configuration = configuration

        if isinstance(self.configuration, (str, Path)):
            self.hopp = Hopp.from_file(self.configuration)

        elif isinstance(self.configuration, dict):
            self.hopp = Hopp.from_dict(self.configuration)

    def reinitialize(self):
        pass

    def simulate(self):
        pass