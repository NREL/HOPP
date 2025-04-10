from pathlib import Path
from typing import Optional, Tuple, Union, Sequence

from attrs import define, field
import numpy as np

from hopp.simulation.technologies.power_source import PowerSource
from hopp.simulation.base import BaseClass
from hopp.simulation.technologies.financial import CustomFinancialModel, FinancialModelType
from hopp.simulation.technologies.sites import SiteInfo
import PySAM.Singleowner as Singleowner
from hopp.simulation.technologies.ghost.ghost_multi import GhostMultiSystem

@define
class GhostConfig(BaseClass):
    n_timesteps: float = field(default = 8760)
    system_capacity_kw: float = field(default = 0.0)
    system_capacity_kwac: Optional[float] = field(default = 0.0)
    fin_model: Optional[Union[dict, FinancialModelType]] = field(default=None)
    name: str = field(default="GhostPlant")
    generation_profile_kw: Optional[list[float]] = field(default = None)
    subsystem_name: Optional[str] = field(default="ghost_system")

@define 
class GhostSystem(BaseClass):
    system_capacity: float = field(default = 0.0)
    system_capacity_ac: Optional[float] = field(default = 0.0)
    system_name: Optional[str] = field(default = "ghost_system")
    n_timesteps: Optional[float] = field(default = 8760)
    t_step: Optional[Union[float,int]] = field(default = 1)
    
    #results
    gen: Optional[list[float]] = field(default = None)
    annual_energy: float = field(init = False)
    capacity_factor: float = field(init = False)
    annual_energy_pre_curtailment_ac: float = field(init = False)
    
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

    def update_capacity_factor(self):
        if self.system_capacity>0:
            capacity_factor = 100*(np.sum(self.gen)/(len(self.gen)*self.system_capacity))
        else:
            capacity_factor = 0.0
        self.value("capacity_factor",capacity_factor)

    def update_system_capacity(self,system_capacity_kw:Union[float,int]):
        self.value("system_capacity",system_capacity_kw)
        self.update_capacity_factor()

    def update_generation_profile(self,generation_profile_kW:Union[list,np.ndarray]):
        if len(generation_profile_kW)==len(self.gen):
            if isinstance(generation_profile_kW,list):
                generation_profile_kW = np.array(generation_profile_kW)
            
            self.value("annual_energy_pre_curtailment_ac",np.sum(generation_profile_kW))
            self.value("annual_energy",np.sum(generation_profile_kW))
            self.value("gen",list(generation_profile_kW))
            self.update_capacity_factor()
            return 
        need_len = len(self.gen)
        is_len = len(generation_profile_kW)
        msg = (
            "Generation profile is not correct length. "
            f"Should be length {need_len} but is length {is_len}")
        raise ValueError(msg)
    
    def calc_nominal_capacity(self,interconnect_kw: float):
        W_ac_nom = min(self.system_capacity_ac, interconnect_kw)
        return W_ac_nom
    
    def calc_gen_max_feasible_kwh(self, interconnect_kw: float):
        #t_step = self.site.interval / 60     
        W_ac_nom = self.calc_nominal_capacity(interconnect_kw)
        
        E_net_max_feasible = [min(x,W_ac_nom) * self.t_step for x in self.gen[0:self.n_timesteps]]      # [kWh]
        return E_net_max_feasible

@define
class GhostPlant(PowerSource):
    site: SiteInfo
    config: Union[GhostConfig,list[GhostConfig]]
    config_name: str = field(init=False, default="CustomGenerationProfileSingleOwner")

    def __attrs_post_init__(self):
        # if self.config.n_ghost_systems==1:
        t_step = self.site.interval / 60
        if isinstance(self.config,list):
            subsystems = []
            subsystem_names = []
            for config in self.config:
                sub = GhostSystem(
                    system_capacity = config.system_capacity_kw,
                    n_timesteps = config.n_timesteps,
                    gen = config.generation_profile_kw,
                    system_capacity_ac = config.system_capacity_kwac,
                    system_name = config.subsystem_name,
                    t_step = t_step,
                    )
                subsystems.append(sub)
                subsystem_names.append(config.subsystem_name)
            system_model = GhostMultiSystem(subsystems,subsystem_names=subsystem_names)
            fin_model = self.config[0].fin_model
            fin_model_name = self.config[0].name
        else:
            system_model = GhostSystem(
                system_capacity = self.config.system_capacity_kw,
                n_timesteps = self.config.n_timesteps,
                gen = self.config.generation_profile_kw,
                system_capacity_ac = self.config.system_capacity_kwac,
                system_name = self.config.subsystem_name,
                t_step = t_step,
                )
            fin_model = self.config.fin_model
            fin_model_name = self.config.name
            
        # if self.config.n_ghost_systems>1:
        #     for ii,system_capacity_kw in enumerate(self.config.system_capacity_kw):
        #         subsystem_model = GhostSystem(
        #             system_capacity_kw,
        #             self.config.n_timesteps,
        #             gen=self.config.generation_profile_kw[ii],
        #             system_capacity_ac = self.config.system_capacity_kwac[ii]
        #             )
        
        financial_model = None
        if isinstance(fin_model, str):
            if "singleowner" in fin_model.lower():
                financial_model = Singleowner.default(fin_model)
            elif isinstance(fin_model, dict):
                financial_model = CustomFinancialModel(fin_model, name=fin_model_name)
            else:
                financial_model = fin_model
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
    
    def calc_nominal_capacity(self, interconnect_kw: float):
        W_ac_nom = self._system_model.calc_nominal_capacity(interconnect_kw)
        return W_ac_nom
    
    def calc_gen_max_feasible_kwh(self, interconnect_kw: float):
        E_net_max_feasible = self._system_model.calc_gen_max_feasible_kwh(interconnect_kw)
        return E_net_max_feasible
