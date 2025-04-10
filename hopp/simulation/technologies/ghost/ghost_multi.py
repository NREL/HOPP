from typing import Optional, Union, TYPE_CHECKING

from attrs import define, field
import numpy as np

from hopp.simulation.base import BaseClass

if TYPE_CHECKING:
    from hopp.simulation.technologies.ghost.ghost_plant import GhostSystem

@define 
class GhostMultiSystem(BaseClass):
    subsystems: list["GhostSystem"]
    subsystem_names: Optional[list[str]] = field(default = [])

    # plant-level
    system_capacity: Optional[float] = field(default = 0.0)
    system_capacity_ac: Optional[float] = field(default = 0.0)
    system_name: Optional[str] = field(default = "ghost_multi")
    n_timesteps: float = field(default = 8760)
    
    #results
    gen: Optional[list[float]] = field(default = None)
    annual_energy: float = field(init = False)
    capacity_factor: float = field(init = False)
    annual_energy_pre_curtailment_ac: float = field(init = False)
    def __attrs_post_init__(self):

        if len(self.subsystem_names)==0:
            subsystem_names_original = [sub.system_name for sub in self.subsystem_names]
            for ni,sub_name in enumerate(subsystem_names_original):
                if subsystem_names_original.count(sub_name)>1:
                    subsystem_names_original[ni] = f"{sub_name}_{ni}"
                    self.subsystems[ni].value("system_name", f"{sub_name}_{ni}")
            self.subsystem_names = subsystem_names_original
        
        self.update_generation_profile(None)
        self.update_system_capacity(None)

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
    def update_generation_profile(self,placeholder):
        generation_profile_kW = np.zeros(self.n_timesteps)

        for sub in self.subsystems:
            generation_profile_kW += np.array(sub.value("gen"))
        self.value("gen",generation_profile_kW)
        self.value("annual_energy",np.sum(generation_profile_kW))
        self.value("gen",list(generation_profile_kW))
        self.update_capacity_factor()
    
    def update_system_capacity(self,placeholder):
        system_capacity_kw = 0.0
        system_capacity_kwac = 0.0
        for sub in self.subsystems:
            system_capacity_kw += np.array(sub.value("system_capacity"))
            system_capacity_kwac += np.array(sub.value("system_capacity_ac"))
        self.value("system_capacity",system_capacity_kw)
        self.value("system_capacity_ac",system_capacity_kwac)
        self.update_capacity_factor()

    def update_capacity_factor(self):
        if self.system_capacity>0:
            capacity_factor = 100*(np.sum(self.gen)/(len(self.gen)*self.system_capacity))
        else:
            capacity_factor = 0.0
        self.value("capacity_factor",capacity_factor)

    def get_subsystem_from_name(self,name:str):
        subs = [sub.system_name for sub in self.subsystems if sub.system_name==name]
        if len(subs)==1:
            return subs[0]
        raise UserWarning(f"No subsystems have unique system_name: {name}")

    def update_generation_profile_for_subsystem(self,generation_profile_kW:Union[list,np.ndarray],subsystem_name:str):
        subsystem = self.get_subsystem_from_name(subsystem_name)
        subsystem.update_generation_profile(generation_profile_kW)
    
    def update_system_capacity_for_subsystem(self,system_capacity_kw:Union[float,int],subsystem_name:str):
        subsystem = self.get_subsystem_from_name(subsystem_name)
        subsystem.update_system_capacity(system_capacity_kw)

    
    def update_value_for_system(self,subsystem_name:str,value_name:str, value):
        subsystem = self.get_subsystem_from_name(subsystem_name)
        subsystem.value(value_name,value)
    
    def get_value_for_system(self,subsystem_name:str,value_name:str):
        """_summary_

        Args:
            subsystem_name (str): _description_
            value_name (str): _description_

        Returns:
            _type_: _description_
        """
        subsystem = self.get_subsystem_from_name(subsystem_name)
        return subsystem.value(value_name)
    
    def calc_nominal_capacity(self,interconnect_kw: float):
        """Calculates the nominal AC net system capacity per subsystem.

        Args:
            interconnect_kw (float): grid interconnection limit in kW

        Returns:
            float: sum of subsystem's nominal AC net capacity [kW]
        """
        W_ac_nom = 0.0
        for sub in self.subsystems:
            W_ac = sub.calc_nominal_capacity(interconnect_kw)
            W_ac_nom += W_ac
        return W_ac_nom
    
    def calc_gen_max_feasible_kwh(self, interconnect_kw: float):
        """Calculates the maximum feasible generation profile that could have occurred (year 1)

        Args:
            interconnect_kw (float): grid interconnection limit in kW

        Returns:
            list: sum of subsystem's maximum feasible generation profile [kWh]
        """
        E_net_max_feasible = np.zeros(self.n_timesteps)

        for sub in self.subsystems:
            E_net_max_feasible_sub = sub.calc_gen_max_feasible_kwh(interconnect_kw)
            E_net_max_feasible += np.array(E_net_max_feasible_sub)
        return E_net_max_feasible.tolist()