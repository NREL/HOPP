import logging
from abc import ABCMeta, abstractmethod
from hybrid.site_info import SiteInfo
import PySAM.Singleowner as Singleowner

logger = logging.getLogger('hybrid_system')


class PowerSource(metaclass=ABCMeta):
    financial_model: Singleowner.Singleowner

    def __init__(self, site: SiteInfo):
        """
        Abstract class for a renewable energy power plant simulation.
        """
        self.site = site

    @abstractmethod
    def system_capacity_kw(self):
        """
        :return: system capacity in kW
        """

    @abstractmethod
    def simulate(self):
        """
        Run the system and financial model
        """

    @abstractmethod
    def generation_profile(self):
        """
        :return: array of power output in kW
        """

    @abstractmethod
    def copy(self):
        """
        :return: new instance
        """