class OptimizationProblem:
    """
    Simulation whose parameters are to be optimized
    """
    
    def __init__(self,
                 site_info,
                 num_turbines: int,
                 min_spacing: float = 200.0,  # [m]
                 ) -> None:
        """
        :param site_info: contains location, site and resource information
        :param num_turbines: desired number of turbines
        :param min_spacing: min spacing between turbines
        """
        self.site_info = site_info
        self.num_turbines: int = num_turbines
        self.min_spacing: float = min_spacing
    
    def _setup_simulation(self
                          ) -> None:
        """
        Initialize simulation and data
        :return:
        """
        pass
    
    def make_conforming_candidate_and_get_penalty(
            self,
            candidate: object
            ) -> tuple:
        """
        Modifies a candidate so that its problem instance respects constraints and returns a penalty for violations
        :param candidate: optimization candidate
        :return: conforming candidate, parameter error values
        """
        pass
    
    def objective(
            self,
            parameters: object
            ) -> float:
        """
        Objective function of the simulation to be optimized
        :param parameters: candidate
        :return: performance
        """
        pass
    
    @staticmethod
    def plot_candidate(
            candidate,
            color='k',
            alpha=.5) -> None:
        pass
