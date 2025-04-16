from pathlib import Path
from typing import Optional, Tuple, Union, Sequence

from attrs import define, field
import numpy as np

import PySAM.Singleowner as Singleowner
import PySAM.Windpower as Windpower
from hopp.simulation.base import BaseClass
from hopp.simulation.technologies.financial import CustomFinancialModel, FinancialModelType
from hopp.simulation.technologies.layout.wind_layout import (
    WindLayout, 
    WindBoundaryGridParameters, 
    WindBasicGridParameters,
    WindCustomParameters,
    WindGridParameters,
)
import hopp.tools.design.wind.turbine_library_interface_tools as turb_lib_interface
from hopp.tools.design.wind.turbine_library_tools import check_turbine_library_for_turbine, print_turbine_name_list
from hopp.simulation.technologies.power_source import PowerSource
from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.technologies.wind.floris import Floris
from hopp.tools.resource.wind_tools import calculate_air_density_losses
from hopp.type_dec import resource_file_converter
from hopp.utilities import load_yaml
from hopp.utilities.log import hybrid_logger as logger
from hopp.utilities.validators import gt_zero, contains, range_val


@define
class WindConfig(BaseClass):
    """
    Configuration class for WindPlant.

    Args:
        num_turbines (int): number of turbines in the farm
        turbine_rating_kw (float): turbine rating in kW
        rotor_diameter (float | int, Optional): turbine rotor diameter in meters
        hub_height (float, Optional): turbine hub height in meters
        turbine_name (str, Optional): unused currently. Defaults to None.
        layout_mode (str):
            - 'boundarygrid': regular grid with boundary turbines, requires 
                WindBoundaryGridParameters as 'layout_params'
            - 'grid': regular grid with dx, dy distance, 0 angle; does not require 'layout_params'
            - 'basicgrid': most-square grid layout, requires WindBasicGridParameters 
                as 'layout_params'
            - 'custom': use a user-provided layout
            - 'floris_layout': use layout provided in `floris_config`.
        model_name (str): which model to use. Options are 'floris' and 'pysam'
        model_input_file (str): file specifying a full PySAM input
        layout_params (obj | dict, Optional): layout configuration object corresponding to 
            `layout_mode` or dictionary.
        rating_range_kw (Tuple[int]): allowable kw range of turbines, default is 1000 - 3000 kW
        floris_config (dict | str | Path): Floris configuration, only used if 
            `model_name` == 'floris'
        adjust_air_density_for_elevation (bool): whether to adjust air density for elevation. 
            Defaults to False. Only used if True and ``site.elev`` is not None. 
        resource_parse_method (str): method to parse wind resource data if using floris and 
            downloaded resource data for 2 heights. Can either be "weighted_average" or "average". 
            Defaults to "average".
        operational_losses (float, Optional): total percentage losses in addition to wake losses, 
            defaults based on PySAM (only used for Floris model)
        timestep (Tuple[int]): Timestep (required for floris runs, otherwise optional). 
            Defaults to (0,8760)
        fin_model (obj | dict | str): Optional financial model. Can be any of the following:

            - a string representing an argument to `Singleowner.default`

            - a dict representing a `CustomFinancialModel`

            - an object representing a `CustomFinancialModel` or `Singleowner.Singleowner` instance
        verbose (bool): if True, print simulation progress statements. Defaults to True. 
        store_turbine_performance_results (bool): If running FLORIS, whether to save speed and power timeseries
            for each turbine in the farm. Defaults to False. 
        store_floris_config_dict (bool): If running FLORIS, whether to store the input dictionary as an attribute. 
            Defaults to True.
        override_wind_resource_height (bool): Whether to ignore a possible discrepancy in wind resource height 
            and the turbine hub-height. Defaults to False.
        recalculate_pysam_powercurve (bool): If True, recalculates the turbine power-curve for the rotor diameter and turbine rating. 
            If False, only scales turbine power-curve for turbine rated power. Defaults to False. Only used if ``model_name = 'pysam'``
    """
    # TODO: put `resource_parse_method`, `store_turbine_performance_results`, and `verbose` in "floris_kwargs" dictionary
    num_turbines: int = field(validator=gt_zero)
    turbine_rating_kw: Optional[float] = field(default = None)
    rotor_diameter: Optional[float] = field(default=None)
    layout_params: Optional[
        Union[
            dict, WindBoundaryGridParameters, WindBasicGridParameters, WindCustomParameters, WindGridParameters
        ]
    ] = field(default=None)
    hub_height: Optional[float] = field(default=None)
    turbine_name: Optional[str] = field(default=None)
    turbine_group: str = field(
        default="none",
        validator=contains(["offshore", "onshore", "distributed", "none"]),
        converter=(str.strip, str.lower)
    )
    layout_mode: str = field(
        default="grid",
        validator=contains(["boundarygrid", "grid", "basicgrid", "custom", "floris_layout"]),
        converter=(str.strip, str.lower)
    )
    model_name: str = field(
        default="pysam",
        validator=contains(["pysam", "floris"]),
        converter=(str.strip, str.lower)
    )
    model_input_file: Optional[str] = field(default=None)
    rating_range_kw: Tuple[int, int] = field(default=(1000, 3000))
    floris_config: Optional[Union[dict, str, Path]] = field(default=None)
    adjust_air_density_for_elevation: Optional[bool] = field(default=False)
    resource_parse_method: str = field(
        default="average",
        validator=contains(["weighted_average", "average"]),
        converter=(str.strip, str.lower)
    )
    operational_losses: float = field(default=12.83, validator=range_val(0, 100))
    timestep: Optional[Tuple[int, int]] = field(default=(0,8760))
    fin_model: Optional[Union[dict, FinancialModelType]] = field(default=None)
    name: str = field(default="WindPlant")
    verbose: bool = field(default = True)
    store_turbine_performance_results: bool = field(default = False)
    store_floris_config_dict: bool = field(default = True)
    override_wind_resource_height: bool = field(default = False)
    recalculate_pysam_powercurve: bool = field(default = False)

    def __attrs_post_init__(self):
        if self.model_name == 'floris' and self.timestep is None:
            raise ValueError("Timestep (Tuple[int, int]) required for floris")

        if self.turbine_rating_kw is None and self.turbine_name is None:
            if self.model_name == "pysam" and self.model_input_file is None:
                raise ValueError("Parameters of turbine_rating_kw or turbine_name are required")


@define
class WindPlant(PowerSource):
    site: SiteInfo
    config: WindConfig

    config_name: str = field(init=False, default="WindPowerSingleOwner")
    _rating_range_kw: Tuple[int, int] = field(init=False)

    def __attrs_post_init__(self):
        """
        WindPlant

        Args:
            site: Site information
            config: Wind plant configuration
        """
        self._rating_range_kw = self.config.rating_range_kw
        layout_params = self.config.layout_params
        layout_mode = self.config.layout_mode
        # Parse input for a financial model
        if isinstance(self.config.fin_model, str):
            financial_model = Singleowner.default(self.config_name)
        elif isinstance(self.config.fin_model, dict):
            financial_model = CustomFinancialModel(self.config.fin_model, name=self.config.name)
        else:
            financial_model = self.config.fin_model

        if self.config.model_name == 'floris':
            if self.config.verbose:
                print('FLORIS is the system model...')
            system_model = Floris(self.site, self.config)
            if (
                self.config.num_turbines == len(system_model.wind_farm_xCoordinates)
                and self.config.layout_mode == "floris_layout"
            ):
                # use layout in floris config by using "floris_layout" layout params
                x_coords,y_coords = system_model.wind_farm_layout
                layout_params = WindCustomParameters(layout_x=x_coords, layout_y=y_coords)
                # modify to custom for WindLayout
                layout_mode = "custom"

            if financial_model is None:
                # default
                financial_model = Singleowner.default(self.config_name)
            else:
                financial_model = self.import_financial_model(
                    financial_model, system_model, self.config_name
                )
        else:
            if self.config.model_input_file is None:
                system_model = Windpower.default(self.config_name)
            else:
                # initialize system using pysam input file
                input_file_path = resource_file_converter(self.config.model_input_file)
                input_dict = load_yaml(input_file_path)

                system_model = Windpower.new()
                system_model.assign(input_dict)

                wind_farm_xCoordinates = input_dict['Farm']['wind_farm_xCoordinates']
                nTurbs = len(wind_farm_xCoordinates)
                system_model.value("wind_resource_data", self.site.wind_resource.data)

                # turbine power curve (array of kW power outputs)
                self.wind_turbine_powercurve_powerout = [1] * nTurbs            

            if financial_model is None:
                # default
                financial_model = Singleowner.from_existing(system_model, self.config_name)
            else:
                financial_model = self.import_financial_model(
                    financial_model, system_model, self.config_name
                )

        super().__init__("WindPlant", self.site, system_model, financial_model)
        self._system_model.value("wind_resource_data", self.site.wind_resource.data)

        self._layout = WindLayout(self.site.polygon, system_model, layout_mode, layout_params)

        self._dispatch = None

        if self.config.turbine_rating_kw is not None:
            self.turb_rating = self.config.turbine_rating_kw
        self.num_turbines = self.config.num_turbines
            
        if self.config.model_name=="pysam":
            self.initialize_pysam_wind_turbine()
    
    def initalize_pysam_turbine_from_turbine_library(self, turbine_name):
        """Initialize PySAM wind turbine from a turbine available in the turbine-models library.

        Args:
            turbine_name (str): name of turbine in turbine-models library.

        Raises:
            ValueError: rotor diameter from turbine library specs does not match hub-height in WindConfig.
            ValueError: hub-height from turbine library specs does not match hub-height in WindConfig.
        """
        valid_name = check_turbine_library_for_turbine(turbine_name,turbine_group=self.config.turbine_group)
        if not valid_name:
            print_turbine_name_list()
            msg = (
                f"Turbine name {turbine_name} was not found the turbine-models library. "
                "Please try an available name."
            )
            ValueError(msg)
        
        turbine_dict = turb_lib_interface.get_pysam_turbine_specs(turbine_name,self)
        self._system_model.Turbine.assign(turbine_dict)
        self.rotor_diameter = turbine_dict["wind_turbine_rotor_diameter"]
        self.turb_rating = np.round(max(turbine_dict["wind_turbine_powercurve_powerout"]), decimals = 3)

        if self.config.rotor_diameter is not None:
            if self.config.rotor_diameter != self._system_model.Turbine.wind_turbine_rotor_diameter:
                msg = (
                    f"Input rotor diameter ({self.config.rotor_diameter}) does not match does not match rotor diameter "
                    f"for turbine ({self._system_model.Turbine.wind_turbine_rotor_diameter})."
                    f"Please correct the value for rotor_diameter in the hopp config input "
                    f"to {self._system_model.Turbine.wind_turbine_rotor_diameter}."
                )
                raise ValueError(msg)
        if self.config.hub_height is not None:
            if self.config.hub_height != self._system_model.Turbine.wind_turbine_hub_ht:
                msg = (
                    f"Input hub-height ({self.config.hub_height}) does not match hub-height "
                    f"for turbine ({self._system_model.Turbine.wind_turbine_hub_ht}). "
                    f"Please correct the value for hub_height in the hopp config input "
                    f"to {self._system_model.Turbine.wind_turbine_hub_ht}."
                )

                raise ValueError(msg)

    def initialize_pysam_wind_turbine(self):
        """Initialize wind turbine parameters for PySAM simulation.

        """

        
        if self.config.turbine_name is not None:
            self.initalize_pysam_turbine_from_turbine_library(self.config.turbine_name)
        else:
            if self.config.rotor_diameter is not None:
                self.rotor_diameter = self.config.rotor_diameter # this will update the layout
            if self.config.hub_height is not None:
                self._system_model.value("wind_turbine_hub_ht", self.config.hub_height)
            if self.config.turbine_rating_kw is not None:
                self.turb_rating = self.config.turbine_rating_kw
            if self.config.recalculate_pysam_powercurve:
                self.modify_powercurve(self.rotor_diameter, self.turb_rating)
                msg = (
                    f"updating wind turbine power-curve for rotor diameter {self.rotor_diameter}m "
                    f"and rating {self.turb_rating} kW"
                )
                logger.info(msg)

        # check wind resource height against turbine hub-height
        hub_height = self._system_model.Turbine.wind_turbine_hub_ht
        if hub_height != self.site.wind_resource.hub_height_meters:
            if hub_height >= min(self.site.wind_resource.data["heights"]) and hub_height<=max(self.site.wind_resource.data["heights"]):
                self.site.wind_resource.hub_height_meters = float(hub_height)
                self.site.hub_height = float(hub_height)
                logger.info(f"updating wind resource hub-height to {hub_height}m")
            else:  
                if not self.config.override_wind_resource_height:
                    logger.warning(f"updating wind resource hub-height to {hub_height}m and redownloading wind resource data")
                    self.site.hub_height = hub_height
                    data = {
                        "lat": self.site.wind_resource.latitude,
                        "lon": self.site.wind_resource.longitude,
                        "year": self.site.wind_resource.year,
                    }
                    wind_resource = self.site.initialize_wind_resource(data)
                    self.site.wind_resource = wind_resource
                    self._system_model.value("wind_resource_data", self.site.wind_resource.data)
        
        # add losses for air density if specified and site elevation is input
        if self.config.adjust_air_density_for_elevation and self.site.elev is not None:
            air_dens_losses = calculate_air_density_losses(self.site.elev)
            self._system_model.Losses.assign({"turb_specific_loss":air_dens_losses})

    @property
    def wake_model(self) -> str:
        try:
            model_type = self._system_model.value("wind_farm_wake_model")
            if model_type == 0:
                return "0 [Simple]"
            elif model_type == 1:
                return "1 [Park (WAsP)]"
            elif model_type == 2:
                return "2 [Eddy Viscosity]"
            elif model_type == 3:
                return "3 [Constant %]"
            else:
                raise ValueError("wake model type unrecognized")
        except:
            raise NotImplementedError

    @wake_model.setter
    def wake_model(self, model_type: int):
        if 0 <= model_type < 4:
            try:
                self._system_model.value("wind_farm_wake_model", model_type)
            except:
                raise NotImplementedError

    @property
    def num_turbines(self):
        return len(self._system_model.value("wind_farm_xCoordinates"))

    @num_turbines.setter
    def num_turbines(self, n_turbines: int):
        
        if self._layout.layout_mode == "custom":
            if n_turbines == len(self._layout.parameters.layout_x):
                self._layout.set_num_turbines(n_turbines)
            else:
                if n_turbines != len(self._system_model.value("wind_farm_xCoordinates")):
                    n_turbs_layout = len(self._system_model.value("wind_farm_xCoordinates"))
                    msg = (
                        f"Using custom wind farm layout and input number of turbines ({n_turbines}) "
                        f"does not equal length of layout ({n_turbs_layout}). "
                        f"Please either update num_turbines in the hopp config to {n_turbs_layout} "
                        f"Or change the layout to include {n_turbines} unique turbine positions."
                    )
                    raise ValueError(msg)
        self._layout.set_num_turbines(n_turbines)

    @property
    def rotor_diameter(self):
        return self._system_model.value("wind_turbine_rotor_diameter")

    @rotor_diameter.setter
    def rotor_diameter(self, d):
        self._system_model.value("wind_turbine_rotor_diameter", d)
        # recalculate layout spacing in case min spacing is violated
        self.num_turbines = self.num_turbines

    @property
    def turb_rating(self):
        """

        :return: kw rating of turbine
        """
        return max(self._system_model.value("wind_turbine_powercurve_powerout"))

    @turb_rating.setter
    def turb_rating(self, rating_kw):
        """
        Set the turbine rating. System capacity gets modified as a result.
        Turbine powercurve will be recalculated according to one of the following methods:

        :param rating_kw: float
        """
        scaling = rating_kw / self.turb_rating
        self._system_model.value(
            "wind_turbine_powercurve_powerout",
            [i * scaling for i in self._system_model.value("wind_turbine_powercurve_powerout")],
        )
        self._system_model.value(
            "system_capacity",
            self.turb_rating * len(self._system_model.value("wind_farm_xCoordinates")),
        )

    def modify_powercurve(self, rotor_diam, rating_kw):
        """
        Recalculate the turbine power curve

        :param rotor_diam: meters
        :param rating_kw: kw

        :return:
        """
        elevation = 0
        wind_default_max_cp = 0.45
        wind_default_max_tip_speed = 60
        wind_default_max_tip_speed_ratio = 8
        wind_default_cut_in_speed = 4
        wind_default_cut_out_speed = 25
        wind_default_drive_train = 0
        try:
            # could fail if current rotor diameter is too big or small for rating
            self._system_model.Turbine.calculate_powercurve(
                rating_kw,
                int(self._system_model.value("wind_turbine_rotor_diameter")),
                elevation,
                wind_default_max_cp,
                wind_default_max_tip_speed,
                wind_default_max_tip_speed_ratio,
                wind_default_cut_in_speed,
                wind_default_cut_out_speed,
                wind_default_drive_train,
            )
            logger.info("WindPlant recalculated powercurve")
        except:
            raise RuntimeError(
                "WindPlant.turb_rating could not calculate turbine powercurve with diameter={}"
                ", rating={}. Check diameter or turn off 'recalculate_powercurve'".
                format(rotor_diam, rating_kw)
            )
        self._system_model.value("wind_turbine_rotor_diameter", rotor_diam)
        self._system_model.value("system_capacity", rating_kw * self.num_turbines)
        logger.info("WindPlant set system_capacity to {} kW".format(self.system_capacity_kw))

    def modify_coordinates(self, xcoords: Sequence, ycoords: Sequence):
        """
        Change the location of the turbines
        """
        if len(xcoords) != len(ycoords):
            raise ValueError("WindPlant turbine coordinate arrays must have same length")
        if self.config.model_name=="floris":
            self._system_model.wind_farm_layout(xcoords, ycoords)
        else:
            self._system_model.value("wind_farm_xCoordinates", xcoords)
            self._system_model.value("wind_farm_yCoordinates", ycoords)
            self._system_model.value("system_capacity", self.turb_rating * len(xcoords))
        logger.debug("WindPlant set xcoords to {}".format(xcoords))
        logger.debug("WindPlant set ycoords to {}".format(ycoords))
        logger.info("WindPlant set system_capacity to {} kW".format(self.system_capacity_kw))

    @property
    def system_capacity_kw(self):
        return self._system_model.value("system_capacity")

    def system_capacity_by_rating(self, wind_size_kw: float):
        """
        Sets the system capacity by adjusting the rating of the turbines within the 
        provided boundaries.

        :param wind_size_kw: desired system capacity in kW
        """
        turb_rating_kw = wind_size_kw / self.num_turbines
        if self._rating_range_kw[0] <= turb_rating_kw <= self._rating_range_kw[1]:
            self.turb_rating = turb_rating_kw
        else:
            logger.error("WindPlant could not meet target system_capacity by adjusting rating")
            raise RuntimeError("WindPlant could not meet target system_capacity")

    def system_capacity_by_num_turbines(self, wind_size_kw):
        """
        Sets the system capacity by adjusting the number of turbines

        :param wind_size_kw: desired system capacity in kW
        """
        new_num_turbines = round(wind_size_kw / self.turb_rating)
        if self.num_turbines != new_num_turbines:
            self.num_turbines = new_num_turbines

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        """
        Sets the system capacity by updates the number of turbines placed according to layout_mode
        :param size_kw:
        :return:
        """
        self.system_capacity_by_num_turbines(size_kw)

    def modify_layout_params(
        self,
        wind_capacity_kW: float,
        layout_params: Union[dict, WindBoundaryGridParameters, WindBasicGridParameters, WindCustomParameters, WindGridParameters],
        layout_mode: Optional[str] = None):
        
        if isinstance(layout_params, dict):
            if layout_mode == "custom":
                layout_params = WindCustomParameters(**layout_params)
            elif layout_mode == "grid":
                layout_params = WindGridParameters(**layout_params)
            elif layout_mode == "basicgrid":
                layout_params = WindBasicGridParameters(**layout_params)
            elif layout_mode == "boundarygrid":
                layout_params = WindBoundaryGridParameters(**layout_params)
            elif layout_mode is None:
                msg = (
                    "If providing layout_params as a dictionary, please specify layout_mode."
                )
                raise ValueError(msg)
        
        self._layout.set_layout_params(wind_capacity_kW, params = layout_params)