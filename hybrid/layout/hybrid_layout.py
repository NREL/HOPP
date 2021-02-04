from typing import Optional

from hybrid.power_source import PowerSource
from hybrid.layout.wind_layout import WindLayout, WindBoundaryGridParameters
from hybrid.layout.solar_layout import SolarLayout, SolarGridParameters


class HybridLayout:
    def __init__(self,
                 power_sources: dict):
        self.solar: Optional[SolarLayout] = None
        self.wind: Optional[WindLayout] = None
        for source, model in power_sources:
            if source == 'wind':
                self.wind = model.layout
            if source == 'solar':
                self.solar = model.layout

    def reset_layout(self,
                     wind_params: Optional[WindBoundaryGridParameters],
                     solar_params: SolarGridParameters):
        if self.solar:
            self.solar.set_layout_params(solar_params)
        if self.wind:
            # add exclusion
            self.wind.set_layout_params(wind_params, self.solar.buffer_region)
        # do flicker
