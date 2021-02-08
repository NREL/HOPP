from typing import Optional

from hybrid.power_source import PowerSource, SiteInfo
from hybrid.layout.wind_layout import WindLayout, WindBoundaryGridParameters
from hybrid.layout.solar_layout import SolarLayout, SolarGridParameters


class HybridLayout:
    def __init__(self,
                 site: SiteInfo,
                 power_sources: dict):
        self.site: SiteInfo = site
        self.solar: Optional[SolarLayout] = None
        self.wind: Optional[WindLayout] = None
        for source, model in power_sources.items():
            if source == 'wind':
                self.wind = model._layout
            if source == 'solar':
                self.solar = model._layout

        self.set_layout()

    def reset_layout(self,
                     wind_params: Optional[WindBoundaryGridParameters],
                     solar_params: SolarGridParameters):
        if self.solar:
            self.solar.set_layout_params(solar_params)
        if self.wind:
            # add exclusion
            self.wind.set_layout_params(wind_params, self.solar.buffer_region)

    def set_layout(self):
        solar_params = None
        wind_params = None
        if self.solar:
            solar_params = self.solar.parameters
        if self.wind:
            wind_params = self.wind.parameters

        # only run hybrid case if both exist
        if solar_params and wind_params:
            self.reset_layout(wind_params, solar_params)

    def plot(self,
             figure=None,
             axes=None,
             wind_color='b',
             solar_color='darkorange',
             site_border_color='k',
             site_alpha=0.95,
             linewidth=4.0
             ):
        if not figure or not axes:
            figure, axes = self.site.plot(figure, axes, site_border_color, site_alpha, linewidth)
        if self.wind:
            self.wind.plot(figure, axes, wind_color)
        if self.solar:
            self.solar.plot(figure, axes, solar_color)
        return figure, axes
