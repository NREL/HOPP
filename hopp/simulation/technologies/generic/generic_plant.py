from typing import Optional, Union

from attrs import define, field
import numpy as np

from hopp.simulation.technologies.power_source import PowerSource
from hopp.simulation.base import BaseClass
from hopp.simulation.technologies.financial import CustomFinancialModel, FinancialModelType
from hopp.simulation.technologies.sites import SiteInfo
import PySAM.Singleowner as Singleowner
from hopp.simulation.technologies.generic.generic_multi import GenericMultiSystem
from hopp.utilities.validators import gt_zero
import warnings

@define
class GenericConfig(BaseClass):
    """Configuration class for GenericPlant

    Args:
        system_capacity_kw (float): system capacity in kW.
        system_capacity_kwac (float, Optional): system capacity in kWac. If not provided then defaults to system_capacity_kw.
        generation_profile_kw (list[float]): generation profile of system in kW.
        subsystem_name (str, Optional): name of subsystem, only used if ``GenericMultiSystem`` is the system_model.
        n_timesteps (float | int): number of timesteps in a year, defaults to 8760.
        fin_model (obj | dict | str): Optional financial model. Can be any of the following:

            - a string representing an argument to `Singleowner.default`

            - a dict representing a `CustomFinancialModel`

            - an object representing a `CustomFinancialModel` or `Singleowner.Singleowner` instance

    """

    system_capacity_kw: float = field(default = 0.0)
    system_capacity_kwac: Optional[float] = field(default = None)
    generation_profile_kw: Optional[list[float]] = field(default = None)
    subsystem_name: Optional[str] = field(default="generic_system")

    n_timesteps: Union[float,int] = field(default = 8760)
    fin_model: Optional[Union[dict, FinancialModelType]] = field(default=None)
    name: str = field(default="GenericPlant")
    
    

@define 
class GenericSystem(BaseClass):
    """Represents a single generic system defined by its system capacity and generation profile.

    Args:
        system_capacity (float): system capacity in kW. For DC-systems, 
            this is likely equal to the system capacity in kW-DC.
        gen (list[float]): generation profile in kW. 
        system_capacity_ac (float, Optional): system capacity in kW-AC. Defaults to system_capacity if not specified.
            Should be specified if `system_capacity_ac` is different than `system_capacity`. 
            This input is helpful when representing DC-systems such as ``PVPlant``.
        system_name (str, Optional): name of system, primarily used if using `GenericMultiSystem`.
            Defaults to "generic_system".
        n_timesteps (float | int, Optional): number of timesteps in a simulation. Defaults to 8760.
        t_step (float | int, Optional): time step in hours. Defaults to 1.

    Attributes:
        annual_energy (float): annual energy production in kWh/year
        capacity_factor (float): capacity factor of the system based on system_capacity as a percent
        annual_energy_pre_curtailment_ac (float): annual energy production in kWh/year
    """

    system_capacity: float = field(validator=gt_zero)
    gen: list[float]
    
    system_capacity_ac: Optional[float] = field(default = None)
    system_name: Optional[str] = field(default = "generic_system")
    n_timesteps: Optional[Union[float,int]] = field(default = 8760)
    t_step: Optional[Union[float,int]] = field(default = 1)
    
    # Calculated values
    annual_energy: float = field(init = False)
    capacity_factor: float = field(init = False)
    annual_energy_pre_curtailment_ac: float = field(init = False)
    
    def __attrs_post_init__(self):
        """Initialize some attributes and set defaults if needed. This method does the following:

        1) calculate attributes:
            - `annual_energy`
            - `annual_energy_pre_curtailment_ac`
            - `capacity_factor`

        2) set `system_capacity_ac` to `system_capacity` if `system_capacity_ac` was not input.

        Raises:
            ValueError: if length of self.gen is not equal to self.n_timesteps
        """

        self.annual_energy = np.sum(self.gen)
        self.annual_energy_pre_curtailment_ac = np.sum(self.gen)
        
        if self.system_capacity_ac is None:
            self.system_capacity_ac = self.system_capacity
        
        if len(self.gen)!=self.n_timesteps:
            msg = (
                f"Generation profile expected to have {self.n_timesteps} values but "
                f"has {len(self.gen)} values."
            )

            raise ValueError(msg)

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
            project_life (int): currently unused project life in years. 
                May be used in financial calculation in the future.
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
        """Update ``system_capacity`` attribute and recalculate ``capacity_factor``.
        Also updates ``system_capacity_ac`` if it was previously equal to ``system_capacity``.

        Note:
            If system_capacity_ac is different than system_capacity, please be sure
            to update system_capacity_ac using the `value()` function.

        Args:
            system_capacity_kw (float | int): system capacity in kW
        """

        if self.system_capacity!=self.system_capacity_ac:
            msg = (
                f"Resetting system_capacity for {self.system_name} but system_capacity_ac ({self.system_capacity_ac}) "
                f"is different than system_capacity ({self.system_capacity}). Remember to update system_capacity_ac too."
            )
            warnings.warn(msg,UserWarning)
        else:
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

        if len(generation_profile_kW)==self.n_timesteps:
            if isinstance(generation_profile_kW,list):
                generation_profile_kW = np.array(generation_profile_kW)
            
            self.value("annual_energy_pre_curtailment_ac",np.sum(generation_profile_kW))
            self.value("annual_energy",np.sum(generation_profile_kW))
            self.value("gen",list(generation_profile_kW))
            self.update_capacity_factor()
            return 
        msg = (
            "Generation profile is not correct length. "
            f"Should be length {self.n_timesteps} but is length {len(generation_profile_kW)}")
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
                    gen = config.generation_profile_kw,
                    n_timesteps = config.n_timesteps,
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
                gen = self.config.generation_profile_kw,
                n_timesteps = self.config.n_timesteps,
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
