from pathlib import Path
from typing import Optional, Tuple, Union, Sequence

from attrs import define, field
import numpy as np

from hopp.simulation.technologies.power_source import PowerSource
from hopp.simulation.base import BaseClass
from hopp.simulation.technologies.financial import CustomFinancialModel, FinancialModelType
from hopp.simulation.technologies.sites import SiteInfo
import PySAM.Singleowner as Singleowner

@define
class GhostConfig(BaseClass):
    n_timesteps: float = field(default = 8760)
    system_capacity_kw: float = field(default = 0.0)
    system_capacity_kwac: Optional[float] = field(default = 0.0)
    fin_model: Optional[Union[dict, FinancialModelType]] = field(default=None)
    name: str = field(default="GhostPlant")
    generation_profile_kw: Optional[list[float]] = field(default = None)
    
    # n_ghost_systems: Optional[int] = field(default = 1)
    # sub_systems_name: Optional[Union[list[str],str]] = field(default = "")
    
    # def __attrs_post_init__(self):
    #     if isinstance(self.system_capacity_kw,list):
    #         self.n_ghost_systems = len(self.system_capacity_kw)
    #         if isinstance(self.system_capacity_kwac,list) or self.system_capacity_kwac>0.0:
    #             if len(self.system_capacity_kwac)!=len(self.system_capacity_kw):
    #                 raise UserWarning("Please specify system capacity in kWac for all systems")
    #         if len(self.generation_profile_kw)!=len(self.system_capacity_kw):
    #             if len(self.system_capacity_kwac)!=len(self.system_capacity_kw):
    #                 raise UserWarning("Please specify generation profiles for all systems")
    #     if self.sub_systems_name == "" and self.n_ghost_systems>1:
    #         self.sub_systems_name = [f"{i}" for i in range(1,self.n_ghost_systems)]


@define 
class GhostSystem(BaseClass):
    system_capacity: Optional[float] = field(default = 0.0)
    system_capacity_ac: Optional[float] = field(default = 0.0)
    system_name: Optional[str] = field(default = "ghost_system")
    n_timesteps: float = field(default = 8760)

    #results
    gen: Optional[list[float]] = field(default = None)
    annual_energy: float = field(init = False)
    capacity_factor: float = field(init = False)
    annual_energy_pre_curtailment_ac: float = field(init = False)
    
    # other stuff for multiple systems

    def __attrs_post_init__(self):
        if self.gen is None:
            self.gen = np.zeros(self.n_timesteps)
        self.annual_energy = np.sum(self.gen)
        self.annual_energy_pre_curtailment_ac = np.sum(self.gen)
        if self.system_capacity>0:
            self.capacity_factor = 100*(np.sum(self.gen)/(len(self.gen)*self.system_capacity))
        else:
            self.capacity_factor = 0.0
        if self.system_capacity_ac==0.0 and self.system_capacity>0:
            self.system_capacity_ac = self.system_capacity


    def value(self, name: str, set_value=None):
        """Set or retrieve attribute of `hopp.simulation.technologies.ghost.ghost_plant.GhostSystem`.
            if set_value = None, then retrieve value; otherwise overwrite variable's value.
        
        Args:
            name (str): name of attribute to set or retrieve.
            set_value (Optional): value to set for variable `name`. 
                If `None`, then retrieve value. Defaults to None.
        """
        if set_value is not None:
            self.__setattr__(name, set_value)
        else:
            return self.__getattribute__(name)
    
    def execute(self, project_life):
        return

    def export(self):
        """
        Return all the ghost system configuration in a dictionary for the financial model
        """
        config = {
            'system_capacity': self.system_capacity,
        }
        return config
    
    def update_system_capacity(self,system_capacity_kw:Union[float,int]):
        if system_capacity_kw>0:
            capacity_factor = 100*(np.sum(self.gen)/(len(self.gen)*self.system_capacity))
        else:
            capacity_factor = 0.0
        self.value("system_capacity",system_capacity_kw)
        self.value("capacity_factor",capacity_factor)

    def update_generation_profile(self,generation_profile_kW:Union[list,np.ndarray]):
        if len(generation_profile_kW)==len(self.gen):
            if isinstance(generation_profile_kW,list):
                generation_profile_kW = np.array(generation_profile_kW)
            capacity_factor = 100*(np.sum(generation_profile_kW)/(len(generation_profile_kW)*self.system_capacity))
            self.value("capacity_factor",capacity_factor)
            self.value("annual_energy_pre_curtailment_ac",np.sum(generation_profile_kW))
            self.value("annual_energy",np.sum(generation_profile_kW))
            self.value("gen",list(generation_profile_kW))
            return 
        need_len = len(self.gen)
        is_len = len(generation_profile_kW)
        msg = (
            "Generation profile is not correct length. "
            f"Should be length {need_len} but is length {is_len}")
        raise ValueError(msg)
        

@define
class GhostPlant(PowerSource):
    site: SiteInfo
    config: GhostConfig
    config_name: str = field(init=False, default="CustomGenerationProfileSingleOwner")

    def __attrs_post_init__(self):
        # if self.config.n_ghost_systems==1:
        system_model = GhostSystem(
            self.config.system_capacity_kw,
            self.config.n_timesteps,
            gen=self.config.generation_profile_kw,
            system_capacity_ac = self.config.system_capacity_kwac
            )
        # if self.config.n_ghost_systems>1:
        #     for ii,system_capacity_kw in enumerate(self.config.system_capacity_kw):
        #         subsystem_model = GhostSystem(
        #             system_capacity_kw,
        #             self.config.n_timesteps,
        #             gen=self.config.generation_profile_kw[ii],
        #             system_capacity_ac = self.config.system_capacity_kwac[ii]
        #             )
        financial_model = None
        if isinstance(self.config.fin_model, str):
            if "singleowner" in self.config.fin_model.lower():
                financial_model = Singleowner.default(self.config.fin_model)
            elif isinstance(self.config.fin_model, dict):
                financial_model = CustomFinancialModel(self.config.fin_model, name=self.config.name)
            else:
                financial_model = self.config.fin_model
        if financial_model is None:
            # default
            financial_model = Singleowner.default(self.config_name)
        else:
            financial_model = self.import_financial_model(
                financial_model, system_model, self.config_name
            )

        super().__init__("GhostPlant", self.site, system_model, financial_model)
        self._dispatch = None
        self._layout = None

    @property
    def system_capacity_kw(self):
        return self._system_model.value("system_capacity")

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        self._system_model.update_system_capacity(size_kw)
    
    @property
    def system_capacity_kwac(self):
        return self._system_model.value("system_capacity_ac")

    @system_capacity_kw.setter
    def system_capacity_kwac(self, size_kwac: float):
        self._system_model.value("system_capacity_ac",size_kwac)

    @property
    def generation_profile(self):
        return self._system_model.value("gen")

    @generation_profile.setter
    def generation_profile(self, generation_profile_kW:Union[list,np.ndarray]):
        self._system_model.update_generation_profile(generation_profile_kW)
    

# if __name__ == "__main__":
#     from hopp.simulation.technologies.ghost.ghost_site import make_ghost_site
#     site = make_ghost_site(site_inputs={})
#     config = GhostConfig.from_dict({"system_capacity_kw":5,"generation_profile_kw": [40.0]*8760})
#     plant = GhostPlant(site = site, config = config)
    []