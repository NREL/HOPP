from typing import Optional, Union

from attrs import define, field
import numpy as np

from hopp.simulation.technologies.power_source import PowerSource
from hopp.simulation.base import BaseClass
from hopp.simulation.technologies.financial import CustomFinancialModel, FinancialModelType
from hopp.simulation.technologies.sites import SiteInfo
import PySAM.Singleowner as Singleowner
from hopp.simulation.technologies.generic.generic_multi import GenericMultiSystem

@define
class GenericConfig(BaseClass):
    """Configuration class for GenericPlant

    Args:
        system_capacity_kw (float): system capacity in kW.
        system_capacity_kwac (float, Optional): system capacity in kWac. If not provided then defaults to system_capacity_kw.
        generation_profile_kw (list[float]): generation profile of system in kW.
        subsystem_name (str, Optional): name of subsystem, mostly used if ``GenericMultiSystem`` is the system_model.
        n_timesteps (float | int): number of timesteps in a year, defaults to 8760.
        fin_model (obj | dict | str): Optional financial model. Can be any of the following:

            - a string representing an argument to `Singleowner.default`

            - a dict representing a `CustomFinancialModel`

            - an object representing a `CustomFinancialModel` or `Singleowner.Singleowner` instance

    """

    system_capacity_kw: float = field(default = 0.0)
    system_capacity_kwac: Optional[float] = field(default = 0.0)
    generation_profile_kw: Optional[list[float]] = field(default = None)
    subsystem_name: Optional[str] = field(default="generic_system")

    n_timesteps: Union[float,int] = field(default = 8760)
    fin_model: Optional[Union[dict, FinancialModelType]] = field(default=None)
    name: str = field(default="GenericPlant")
    
    

@define 
class GenericSystem(BaseClass):
    system_capacity: float = field(default = 0.0)
    system_capacity_ac: Optional[float] = field(default = 0.0)
    system_name: Optional[str] = field(default = "generic_system")
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
        
        if self.system_capacity_ac==0.0 and self.system_capacity>0:
            self.system_capacity_ac = self.system_capacity
        
        self.update_capacity_factor()

    def value(self, name: str, set_value=None):
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

    def update_capacity_factor(self):
        """Recalculate and update system capacity_factor as a percent (%)
        """

        if self.system_capacity>0:
            capacity_factor = 100*(np.sum(self.gen)/(len(self.gen)*self.system_capacity))
        else:
            capacity_factor = 0.0
        self.value("capacity_factor",capacity_factor)

    def update_system_capacity(self,system_capacity_kw:Union[float,int]):
        """Update ``system_capacity`` attribute and relcalculate ``capacity_factor``.

        Note:
            If system_capacity_ac is different than system_capacity, please be sure
            to update system_capacity_ac using the `value()` function.

        Args:
            system_capacity_kw (float | int): system capacity in kW
        """

        if self.system_capacity==self.system_capacity_ac:
            self.value("system_capacity_ac",system_capacity_kw)
        self.value("system_capacity",system_capacity_kw)
        self.update_capacity_factor()

    def update_generation_profile(self,generation_profile_kW:Union[list,np.ndarray]):
        """Reset the generation profile and update corresponding attributes 
        (`gen`, `annual_energy_pre_curtailment_ac`, `annual_energy`, and `capacity_factor`).

        Args:
            generation_profile_kW (Union[list,np.ndarray]): generation profile in kW

        Raises:
            ValueError: if input generation_profile_kW is not same length as gen attribute.
        """

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
        """Calculates the nominal AC net system capacity.

        Args:
            interconnect_kw (float): grid interconnection limit in kW

        Returns:
            float: system's nominal AC net capacity [kW]
        """

        W_ac_nom = min(self.system_capacity_ac, interconnect_kw)
        return W_ac_nom
    
    def calc_gen_max_feasible_kwh(self, interconnect_kw: float):
        """Calculates the maximum feasible generation profile that could have occurred (year 1)

        Args:
            interconnect_kw (float): grid interconnection limit in kW

        Returns:
            list[float]: maximum feasible generation [kWh]
        """

        W_ac_nom = self.calc_nominal_capacity(interconnect_kw)
        
        E_net_max_feasible = [min(x,W_ac_nom) * self.t_step for x in self.gen[0:self.n_timesteps]]      # [kWh]
        return E_net_max_feasible

@define
class GenericPlant(PowerSource):
    site: SiteInfo
    config: Union[GenericConfig,list[GenericConfig]]
    config_name: str = field(init=False, default="CustomGenerationProfileSingleOwner")

    def __attrs_post_init__(self):
        t_step = self.site.interval / 60
        
        if isinstance(self.config,list):
            # requires GenericMultiSystem as system_model
            subsystems = []
            subsystem_names = []
            for config in self.config:
                sub = GenericSystem(
                    system_capacity = config.system_capacity_kw,
                    n_timesteps = config.n_timesteps,
                    gen = config.generation_profile_kw,
                    system_capacity_ac = config.system_capacity_kwac,
                    system_name = config.subsystem_name,
                    t_step = t_step,
                    )
                subsystems.append(sub)
                subsystem_names.append(config.subsystem_name)
            system_model = GenericMultiSystem(subsystems,subsystem_names=subsystem_names)
            fin_model = self.config[0].fin_model
            fin_model_name = self.config[0].name
        else:
            # requires GenericSystem as system_model
            system_model = GenericSystem(
                system_capacity = self.config.system_capacity_kw,
                n_timesteps = self.config.n_timesteps,
                gen = self.config.generation_profile_kw,
                system_capacity_ac = self.config.system_capacity_kwac,
                system_name = self.config.subsystem_name,
                t_step = t_step,
                )
            fin_model = self.config.fin_model
            fin_model_name = self.config.name
            
        
        financial_model = None
        if isinstance(fin_model, str):
            if "singleowner" in fin_model.lower():
                financial_model = Singleowner.default(fin_model)
            elif isinstance(fin_model, dict):
                financial_model = CustomFinancialModel(fin_model, name=fin_model_name)
            else:
                financial_model = fin_model
        if financial_model is None:
            financial_model = Singleowner.default(self.config_name)
        else:
            financial_model = self.import_financial_model(
                financial_model, system_model, self.config_name
            )

        super().__init__("GenericPlant", self.site, system_model, financial_model)
        self._dispatch = None
        self._layout = None

    @property
    def system_capacity_kw(self):
        """float: System capacity in kW.
        """
        return self._system_model.value("system_capacity")

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        self._system_model.update_system_capacity(size_kw)
    
    @property
    def system_capacity_kwac(self):
        """float: AC system capacity in kW-AC.
        """
        return self._system_model.value("system_capacity_ac")

    @system_capacity_kw.setter
    def system_capacity_kwac(self, size_kwac: float):
        self._system_model.value("system_capacity_ac",size_kwac)

    @property
    def generation_profile(self):
        """list[float]: generation profile in kW.
        """
        return self._system_model.value("gen")

    @generation_profile.setter
    def generation_profile(self, generation_profile_kW:Union[list,np.ndarray]):
        self._system_model.update_generation_profile(generation_profile_kW)
    
    def calc_nominal_capacity(self, interconnect_kw: float):
        """Calculates the nominal AC net system capacity.

        Args:
            interconnect_kw (float): grid interconnection limit in kW

        Returns:
            float: sum of subsystem's nominal AC net capacity [kW]
        """

        W_ac_nom = self._system_model.calc_nominal_capacity(interconnect_kw)
        return W_ac_nom
    
    def calc_gen_max_feasible_kwh(self, interconnect_kw: float):
        """Calculates the maximum feasible generation profile that could have occurred (year 1).

        Args:
            interconnect_kw (float): grid interconnection limit in kW

        Returns:
            list[float]: maximum feasible generation timeseries [kWh]
        """

        E_net_max_feasible = self._system_model.calc_gen_max_feasible_kwh(interconnect_kw)
        return E_net_max_feasible
