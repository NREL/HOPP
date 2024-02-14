from typing import Sequence, Optional, Union, List

from attrs import define, field
import PySAM.Pvsamv1 as Pvsam
import PySAM.Singleowner as Singleowner

from hopp.simulation.technologies.financial import FinancialModelType, CustomFinancialModel
from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.technologies.power_source import PowerSource
from hopp.simulation.technologies.layout.pv_layout import PVLayout, PVGridParameters
from hopp.simulation.technologies.layout.pv_design_utils import (
    align_from_capacity, get_inverter_power, verify_capacity_from_electrical_parameters
)
from hopp.simulation.technologies.layout.pv_module import (
    get_module_attribs, set_module_attribs
)
from hopp.simulation.technologies.layout.pv_inverter import (
    set_inverter_attribs, get_inverter_attribs
)
from hopp.simulation.base import BaseClass

from hopp.tools.utils import flatten_dict


@define
class DetailedPVConfig(BaseClass):
    """
    Configuration class for `DetailedPVPlant`.

    Converts nested dicts into relevant instances for layout and
    financial configurations.

    Args:
        system_capacity_kw: Design system capacity
        use_pvwatts: Whether to use PVWatts (defaults to True). If False, this
            config should be used in a `DetailedPVPlant`.
        layout_params: Optional layout parameters
        layout_model: Optional layout model instance
        fin_model: Optional financial model. Can be any of the following:

            - a string representing an argument to `Singleowner.default`

            - a dict representing a `CustomFinancialModel`

            - an object representing a `CustomFinancialModel` or `Singleowner.Singleowner` instance

        tech_config: Optional dict with more detailed system configuration
        dc_degradation: Annual DC degradation for lifetime simulations [%/year]
    """
    system_capacity_kw: Optional[float] = field(default=None)
    tech_config: Optional[dict] = field(default=None)

    use_pvwatts: bool = field(default=True)
    layout_params: Optional[Union[dict, PVGridParameters]] = field(default=None)
    layout_model: Optional[Union[dict, PVLayout]] = field(default=None)
    fin_model: Optional[Union[str, dict, FinancialModelType]] = field(default=None)
    dc_degradation: Optional[List[float]] = field(default=None)

@define
class DetailedPVPlant(PowerSource):
    """
    A detailed PV Plant, typically using `Pvsam`.

    Args:
        site: The site information.
        config: Configuration dictionary representing a `DetailedPVConfig`.

    """
    site: SiteInfo
    config: DetailedPVConfig

    config_name: str = field(init=False, default="FlatPlatePVSingleOwner")

    def __attrs_post_init__(self):
        system_model = Pvsam.default(self.config_name)

        if isinstance(self.config.fin_model, str):
            financial_model = Singleowner.default(self.config.fin_model)
        elif isinstance(self.config.fin_model, dict):
            financial_model = CustomFinancialModel(self.config.fin_model)
        else:
            financial_model = self.config.fin_model

        if isinstance(self.config.layout_params, dict):
            layout_params = PVGridParameters(**self.config.layout_params)
        else:
            layout_params = self.config.layout_params

        if isinstance(self.config.layout_model, dict):
            layout_model = PVLayout(**self.config.layout_model)
        else:
            layout_model = self.config.layout_model

        if financial_model is None:
            # default
            financial_model = Singleowner.from_existing(system_model, self.config_name)
        else:
            financial_model = self.import_financial_model(financial_model, system_model, self.config_name)

        super().__init__("PVPlant", self.site, system_model, financial_model)

        if self.site.solar_resource is not None:
            self._system_model.SolarResource.solar_resource_data = self.site.solar_resource.data

        if self.config.dc_degradation is not None:
            self.dc_degradation = self.config.dc_degradation
        else:
            self.dc_degradation = [0]

        if layout_model is not None:
            self.layout = layout_model
            self.layout._system_model = self._system_model
        else:
            self.layout = PVLayout(
                self.site, 
                self._system_model, 
                layout_params
            )

        self.processed_assign()

    def processed_assign(self):
        """
        Assign attributes from dictionaries with additional processing
        to enforce coherence between attributes.
        """
        if self.config.system_capacity_kw is not None:       # aggregate into tech_config
            if self.config.tech_config is None:
                self.config.tech_config = {}
            self.config.tech_config['system_capacity'] = self.config.system_capacity_kw

        if self.config.tech_config is not None:
            config = self.config.tech_config
            
            if 'subarray2_enable' in config.keys() and config['subarray2_enable'] == 1 \
              or 'subarray3_enable' in config.keys() and config['subarray3_enable'] == 1 \
              or 'subarray4_enable' in config.keys() and config['subarray4_enable'] == 1:
                raise Exception('Detailed PV plant currently only supports one subarray.')

            # Get PV module attributes
            system_params = flatten_dict(self._system_model.export())
            system_params.update(config)
            module_attribs = get_module_attribs(system_params)

            # Verify system capacity is cohesive with interdependent parameters if all are specified
            nstrings_keys = [f'subarray{i}_nstrings' for i in range(1, 5)]
            if 'system_capacity' in config.keys() and any(nstrings in config.keys() for nstrings in nstrings_keys):               
                # Build subarray electrical configuration input lists
                n_strings = []
                modules_per_string = []
                for i in range(1, 5):
                    if i == 1:
                        subarray_enabled = True
                    else:
                        subarray_enabled = config[f'subarray{i}_enable'] \
                                           if f'subarray{i}_enable' in config.keys() \
                                           else self.value(f'subarray{i}_enable')

                    if not subarray_enabled:
                        n_strings.append(0)
                    elif f'subarray{i}_nstrings' in config.keys():
                        n_strings.append(config[f'subarray{i}_nstrings'])
                    else:
                        try:
                            n_strings.append(self.value(f'subarray{i}_nstrings'))
                        except:
                            n_strings.append(0)

                    if f'subarray{i}_modules_per_string' in config.keys():
                        modules_per_string.append(config[f'subarray{i}_modules_per_string'])
                    else:
                        try:
                            modules_per_string.append(self.value(f'subarray{i}_modules_per_string'))
                        except:
                            modules_per_string.append(0)

                config['system_capacity'] = verify_capacity_from_electrical_parameters(
                    system_capacity_target=config['system_capacity'],
                    n_strings=n_strings,
                    modules_per_string=modules_per_string,
                    module_power=module_attribs['P_mp_ref']
                )
            
                # Set all interdependent parameters directly and at once to avoid interdependent changes with existing values via properties
                if 'system_capacity' in config.keys():
                    self._system_model.value('system_capacity', config['system_capacity'])
                for i in range(1, 5):
                    if f'subarray{i}_nstrings' in config.keys():
                        self._system_model.value(f'subarray{i}_nstrings', config[f'subarray{i}_nstrings'])
                    if f'subarray{i}_modules_per_string' in config.keys():
                        self._system_model.value(f'subarray{i}_modules_per_string', config[f'subarray{i}_modules_per_string'])
                if 'module_model' in config.keys():
                    self._system_model.value('module_model', config['module_model'])
                if 'module_aspect_ratio' in config.keys():
                    self._system_model.value('module_aspect_ratio', config['module_aspect_ratio'])
                for key in config.keys():
                    # set module parameters:
                    if key.startswith('spe_') \
                      or key.startswith('cec_') \
                      or key.startswith('sixpar_') \
                      or key.startswith('snl_') \
                      or key.startswith('sd11par_') \
                      or key.startswith('mlm_'):
                        self._system_model.value(key, config[key])

            # Set all parameters
            self.assign(config)

        if self.layout.parameters is not None:
            self.layout.set_layout_params(self.system_capacity, self.layout.parameters)

    def get_pv_module(self, only_ref_vals: bool = True) -> dict:
        """
        Returns the PV module attributes for either the PVsamv1 or PVWattsv8 models.

        Args:
            only_ref_vals: returns only the reference values (e.g., I_sc_ref) if True
                or model params if False

        Returns:
            dict: PV module attributes
        """
        return get_module_attribs(self._system_model, only_ref_vals)

    def set_pv_module(self, params: dict):
        """
        Sets the PV module model parameters for either the PVsamv1 or PVWattsv8 models.

        Args:
            params: dictionary of parameters

        """
        set_module_attribs(self._system_model, params)
        # update system capacity directly to not recalculate the number of inverters, consistent with the SAM UI
        self._system_model.value('system_capacity', self.module_power * self.modules_per_string * self.n_strings)

    def get_inverter(self, only_ref_vals: bool = True) -> dict:
        """
        Returns the inverter attributes for either the PVsamv1 or PVWattsv8 models.

        Args:
            only_ref_vals: optional, returns only the reference values (e.g., V_dc_max) if True or model params if False

        Returns:
            dict: inverter attributes
        """
        return get_inverter_attribs(self._system_model, only_ref_vals)

    def set_inverter(self, params: dict):
        """
        Sets the inverter model parameters for either the PVsamv1 or PVWattsv8 models.

        Args:
            params: dictionary of parameters

        """
        set_inverter_attribs(self._system_model, params)

    @property
    def system_capacity(self) -> float:
        """Pass through to established name property."""
        return self.system_capacity_kw

    @system_capacity.setter
    def system_capacity(self, size_kw: float):
        """Pass through to established name setter."""
        self.system_capacity_kw = size_kw

    @property
    def system_capacity_kw(self) -> float:
        return self._system_model.value('system_capacity')      # [kW] DC

    @system_capacity_kw.setter
    def system_capacity_kw(self, system_capacity_kw_: float):
        """
        Sets the system capacity.

        Args:
            system_capacity_kw_: DC system size in kW

        """
        n_strings, system_capacity, n_inverters = align_from_capacity(
            system_capacity_target=system_capacity_kw_,
            dc_ac_ratio=self.dc_ac_ratio,
            modules_per_string=self.modules_per_string,
            module_power=self.module_power,
            inverter_power=self.inverter_power,
        )
        self._system_model.value('system_capacity', system_capacity)
        self._system_model.value('subarray1_nstrings', n_strings)
        self._system_model.value('subarray2_nstrings', 0)
        self._system_model.value('subarray3_nstrings', 0)
        self._system_model.value('subarray4_nstrings', 0)
        self._system_model.value('subarray2_enable', 0)
        self._system_model.value('subarray3_enable', 0)
        self._system_model.value('subarray4_enable', 0)
        self._system_model.value('inverter_count', n_inverters)

    @property
    def dc_degradation(self) -> float:
        """Annual DC degradation for lifetime simulations [%/year]"""
        return self._system_model.Lifetime.dc_degradation

    @dc_degradation.setter
    def dc_degradation(self, dc_deg_per_year: Sequence):
        self._system_model.Lifetime.dc_degradation = dc_deg_per_year

    @property
    def dc_ac_ratio(self) -> float:
        return self.system_capacity / (self.n_inverters * self.inverter_power)
    
    @dc_ac_ratio.setter
    def dc_ac_ratio(self, target_dc_ac_ratio: float):
        """
        Sets the dc to ac ratio while keeping the existing system capacity, by
        adjusting the modules per string and number of inverters.
        """
        n_strings, system_capacity, n_inverters = align_from_capacity(
            system_capacity_target=self.system_capacity_kw,
            dc_ac_ratio=target_dc_ac_ratio,
            modules_per_string=self.modules_per_string,
            module_power=self.module_power,
            inverter_power=self.inverter_power,
        )
        self._system_model.value('system_capacity', system_capacity)
        self._system_model.value('subarray1_nstrings', n_strings)
        self._system_model.value('subarray2_nstrings', 0)
        self._system_model.value('subarray3_nstrings', 0)
        self._system_model.value('subarray4_nstrings', 0)
        self._system_model.value('subarray2_enable', 0)
        self._system_model.value('subarray3_enable', 0)
        self._system_model.value('subarray4_enable', 0)
        self._system_model.value('inverter_count', n_inverters)

    @property
    def module_power(self) -> float:
        """Module power in kW."""
        module_attribs = get_module_attribs(self._system_model)
        return module_attribs['P_mp_ref']

    @property
    def module_width(self) -> float:
        """Module width in meters."""
        module_attribs = get_module_attribs(self._system_model)
        return module_attribs['width']

    @property
    def module_length(self) -> float:
        """Module length in meters."""
        module_attribs = get_module_attribs(self._system_model)
        return module_attribs['length']

    @property
    def module_height(self) -> float:
        """Module height in meters."""
        return self.module_length

    @property
    def inverter_power(self) -> float:
        """Inverter power in kW."""
        return get_inverter_power(self._system_model)

    @property
    def modules_per_string(self) -> float:
        """Modules per string."""
        return self._system_model.SystemDesign.subarray1_modules_per_string

    @modules_per_string.setter
    def modules_per_string(self, _modules_per_string: float):
        """Sets the modules per string and updates the system capacity."""
        self._system_model.SystemDesign.subarray1_modules_per_string = _modules_per_string
        self._system_model.SystemDesign.subarray2_modules_per_string = 0 
        self._system_model.SystemDesign.subarray3_modules_per_string = 0
        self._system_model.SystemDesign.subarray4_modules_per_string = 0
        # update system capacity directly to not recalculate the number of inverters, consistent with the SAM UI
        self._system_model.value('system_capacity', self.module_power * _modules_per_string * self.n_strings)

    @property
    def subarray1_modules_per_string(self) -> float:
        """Number of modules per string in subarray 1."""
        return self._system_model.value('subarray1_modules_per_string')

    @subarray1_modules_per_string.setter
    def subarray1_modules_per_string(self, subarray1_modules_per_string_: float):
        """
        Sets the number of modules per string in subarray 1, which is for now
        the same in all subarrays.
        """
        self.modules_per_string = subarray1_modules_per_string_

    @property
    def n_strings(self) -> float:
        """Total number of strings."""
        return self._system_model.SystemDesign.subarray1_nstrings \
               + self._system_model.SystemDesign.subarray2_nstrings \
               + self._system_model.SystemDesign.subarray3_nstrings \
               + self._system_model.SystemDesign.subarray4_nstrings

    @n_strings.setter
    def n_strings(self, _n_strings: float):
        """Sets the total number of strings and updates the system capacity."""
        self._system_model.SystemDesign.subarray1_nstrings = _n_strings
        self._system_model.SystemDesign.subarray2_nstrings = 0 
        self._system_model.SystemDesign.subarray3_nstrings = 0
        self._system_model.SystemDesign.subarray4_nstrings = 0
        # update system capacity directly to not recalculate the number of inverters, consistent with the SAM UI
        self._system_model.value('system_capacity', self.module_power * self.modules_per_string * _n_strings)

    @property
    def subarray1_nstrings(self) -> float:
        """Number of strings in subarray 1."""
        return self._system_model.value('subarray1_nstrings')

    @subarray1_nstrings.setter
    def subarray1_nstrings(self, subarray1_nstrings_: float):
        """
        Sets the number of strings in subarray 1, which is for now the total
        number of strings.
        """
        self.n_strings = subarray1_nstrings_

    @property
    def n_inverters(self) -> float:
        """Total number of inverters."""
        return self._system_model.SystemDesign.inverter_count

    @n_inverters.setter
    def n_inverters(self, _n_inverters: float):
        """Sets the total number of inverters."""
        self._system_model.SystemDesign.inverter_count = _n_inverters
