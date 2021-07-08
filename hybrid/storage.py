
from hybrid.sites import SiteInfo
from hybrid.power_source import PowerSource, Singleowner


class Battery(PowerSource):
    system_model: object
    financial_model: Singleowner.Singleowner

    def __init__(self, site: SiteInfo):
        super().__init__(site)
    # TODO: implement
