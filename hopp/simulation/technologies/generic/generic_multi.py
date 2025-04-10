from typing import Optional, Union, TYPE_CHECKING

from attrs import define, field
import numpy as np

from hopp.simulation.base import BaseClass

if TYPE_CHECKING:
    from hopp.simulation.technologies.generic.generic_plant import GenericSystem

@define 
class GenericMultiSystem(BaseClass):
    subsystems: list["GenericSystem"]
    subsystem_names: Optional[list[str]] = field(default = [])

    # plant-level
    system_capacity: Optional[float] = field(default = 0.0)
    system_capacity_ac: Optional[float] = field(default = 0.0)
    system_name: Optional[str] = field(default = "generic_multi")
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

    def value(self, name:str, set_value=None):
        """Set or retrieve attribute of `hopp.simulation.technologies.generic.generic_plant.GenericSystem`.
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
        """Empty execute function since generation is set during initialization.

        Args:
            project_life (int): unused project life in years
        """
        return

    def export(self):
        """Return all the generic system configuration in a dictionary for the financial model
        
        Returns:
            dict: generic system configuration for the financial model.
        """

        config = {
            'system_capacity': self.system_capacity,
        }
        return config

    def update_generation_profile(self, placeholder):
        """Recalculate ``gen`` attribute after subsystem generation profiles may have been updated. Also updates ``annual_energy``, 
        ``annual_energy_pre_curtailment_ac``, and ``capacity_factor``.

        Args:
            placeholder (None): unused placeholder value so this function parallels 
            the function in ``hopp.simulation.technologies.generic.generic_plant.GenericSystem``
        """

        generation_profile_kW = np.zeros(self.n_timesteps)

        for sub in self.subsystems:
            generation_profile_kW += np.array(sub.value("gen"))
        self.value("annual_energy_pre_curtailment_ac",np.sum(generation_profile_kW))
        self.value("annual_energy",np.sum(generation_profile_kW))
        self.value("gen",generation_profile_kW.tolist())
        self.update_capacity_factor()
    
    def update_system_capacity(self, placeholder):
        """Recalculate ``system_capacity`` attribute after subsystem system capacities may have been updated. 
        Also updates ``system_capacity_ac`` and  ``capacity_factor``.

        Args:
            placeholder (None): unused placeholder value so this function parallels 
            the function in ``hopp.simulation.technologies.generic.generic_plant.GenericSystem``
        """

        system_capacity_kw = 0.0
        system_capacity_kwac = 0.0
        for sub in self.subsystems:
            system_capacity_kw += np.array(sub.value("system_capacity"))
            system_capacity_kwac += np.array(sub.value("system_capacity_ac"))
        self.value("system_capacity",system_capacity_kw)
        self.value("system_capacity_ac",system_capacity_kwac)
        self.update_capacity_factor()

    def update_capacity_factor(self):
        """Recalculate and update ``capacity_factor`` as a percent (%)
        """

        if self.system_capacity>0:
            capacity_factor = 100*(np.sum(self.gen)/(len(self.gen)*self.system_capacity))
        else:
            capacity_factor = 0.0
        self.value("capacity_factor",capacity_factor)

    def get_subsystem_from_name(self, subsystem_name:str):
        """Retrieve subsystem object with system_name==subsystem_name.

        Args:
            subsystem_name (str): name of subsystem, `corresponding to GenericSystem.system_name`

        Raises:
            UserWarning: if subsystem_name doesn't match system_name of any subsystems.

        Returns:
            :obj:`hopp.simulation.technologies.generic.generic_plant.GenericSystem`: GenericSystem object
        """
        
        subs = [sub.system_name for sub in self.subsystems if sub.system_name==subsystem_name]
        if len(subs)==1:
            return subs[0]
        raise UserWarning(f"No subsystems have unique system_name: {subsystem_name}")

    def update_generation_profile_for_subsystem(self, generation_profile_kW:Union[list,np.ndarray], subsystem_name:str):
        """Update the generation profile for a single subsystem.

        Args:
            generation_profile_kW (Union[list,np.ndarray]): generation profile of subsystem in kW
            subsystem_name (str): name of subsystem to set the generation profile for.
        """

        subsystem = self.get_subsystem_from_name(subsystem_name)
        subsystem.update_generation_profile(generation_profile_kW)
        self.update_generation_profile(None)
    
    def update_system_capacity_for_subsystem(self, system_capacity_kw:Union[float,int], subsystem_name:str):
        """Update the system capacity for a single subsystem.

        Args:
            system_capacity_kw (Union[float,int]): system capacity of subsystem.
            subsystem_name (str): name of subsystem to set the system capacity for.
        """

        subsystem = self.get_subsystem_from_name(subsystem_name)
        subsystem.update_system_capacity(system_capacity_kw)
        self.update_system_capacity(None)

    def set_subsystem_value(self, subsystem_name:str, variable_name:str, value):
        """Set attribute of `hopp.simulation.technologies.generic.generic_plant.GenericSystem`

        Args:
            subsystem_name (str): name of subsystem, corresponds to `hopp.simulation.technologies.generic.generic_plant.GenericSystem.system_name`
            variable_name (str): name of attribute to retrieve.
            value (Any): value to set for variable `variable_name`. 
        """

        subsystem = self.get_subsystem_from_name(subsystem_name)
        subsystem.value(variable_name,value)
    
    def get_subsystem_value(self, subsystem_name:str, variable_name:str):
        """Retrieve attribute of `hopp.simulation.technologies.generic.generic_plant.GenericSystem`

        Args:
            subsystem_name (str): name of subsystem, corresponds to `hopp.simulation.technologies.generic.generic_plant.GenericSystem.system_name`
            variable_name (str): name of attribute to retrieve.

        Returns:
            any: value for ``variable_name`` attribute of subsystem corresponding to ``subsystem_name``
        """

        subsystem = self.get_subsystem_from_name(subsystem_name)
        return subsystem.value(variable_name)
    
    def calc_nominal_capacity(self, interconnect_kw: float):
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
            list[float]: sum of subsystem's maximum feasible generation profile [kWh]
        """
        E_net_max_feasible = np.zeros(self.n_timesteps)

        for sub in self.subsystems:
            E_net_max_feasible_sub = sub.calc_gen_max_feasible_kwh(interconnect_kw)
            E_net_max_feasible += np.array(E_net_max_feasible_sub)
        return E_net_max_feasible.tolist()