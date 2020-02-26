from hybrid.site_info import SiteInfo


class OptimizationProblem:
    def __init__(self,
                 site_info: SiteInfo,
                 num_turbines: int,
                 min_spacing: float = 200.0,  # [m]
                 ) -> None:
        self.site_info: SiteInfo = site_info
        self.num_turbines: int = num_turbines
        self.min_spacing: float = min_spacing

    def objective(self, candidate):
        pass