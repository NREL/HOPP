from pathlib import Path
from typing import Optional, Tuple, Union, Sequence

import PySAM.Windpower as Windpower
import PySAM.Singleowner as Singleowner
from attrs import define, field

from hopp.simulation.base import BaseClass
from hopp.type_dec import resource_file_converter
from hopp.utilities import load_yaml
from hopp.utilities.validators import gt_zero, contains, range_val
from hopp.simulation.technologies.wind.floris import Floris
from hopp.simulation.technologies.power_source import PowerSource
from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.technologies.layout.wind_layout import WindLayout, WindBoundaryGridParameters
from hopp.simulation.technologies.financial import CustomFinancialModel, FinancialModelType
from hopp.utilities.log import hybrid_logger as logger


@define
class WindConfig(BaseClass):
    """
    Configuration class for WindPlant.

    Args:
        num_turbines: number of turbines in the farm
        turbine_rating_kw: turbine rating
        rotor_diameter: turbine rotor diameter
        hub_height: turbine hub height
        layout_mode:
            - 'boundarygrid': regular grid with boundary turbines, requires WindBoundaryGridParameters as 'params'
            - 'grid': regular grid with dx, dy distance, 0 angle; does not require 'params'
        model_name: which model to use. Options are 'floris' and 'pysam'
        model_input_file: file specifying a full PySAM input
        layout_params: layout configuration
        rating_range_kw: allowable kw range of turbines, default is 1000 - 3000 kW
        floris_config: Floris configuration, only used if `model_name` == 'floris'
        operational_losses: total percentage losses in addition to wake losses, defaults based on PySAM (only used for Floris model)
        timestep: Timestep (required for floris runs, otherwise optional)
        fin_model: Optional financial model. Can be any of the following:

            - a string representing an argument to `Singleowner.default`

            - a dict representing a `CustomFinancialModel`

            - an object representing a `CustomFinancialModel` or `Singleowner.Singleowner` instance
    """
    num_turbines: int = field(validator=gt_zero)
    turbine_rating_kw: float = field(validator=gt_zero)
    rotor_diameter: Optional[float] = field(default=None)
    layout_params: Optional[Union[dict, WindBoundaryGridParameters]] = field(default=None)
    hub_height: Optional[float] = field(default=None)
    layout_mode: str = field(default="grid", validator=contains(["boundarygrid", "grid"]))
    model_name: str = field(default="pysam", validator=contains(["pysam", "floris"]))
    model_input_file: Optional[str] = field(default=None)
    rating_range_kw: Tuple[int, int] = field(default=(1000, 3000))
    floris_config: Optional[Union[dict, str, Path]] = field(default=None)
    operational_losses: float = field(default = 12.83, validator=range_val(0, 100))
    timestep: Optional[Tuple[int, int]] = field(default=None)
    fin_model: Optional[Union[dict, FinancialModelType]] = field(default=None)

    def __attrs_post_init__(self):
        if self.model_name == 'floris' and self.timestep is None:
            raise ValueError("Timestep (Tuple[int, int]) required for floris")

        if self.layout_mode == 'boundarygrid' and self.layout_params is None:
            raise ValueError("Parameters of WindBoundaryGridParameters required for boundarygrid layout mode")


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

        if self.config.model_name == 'floris':
            print('FLORIS is the system model...')
            system_model = Floris(self.site, self.config)
            financial_model = Singleowner.default(self.config_name)
        else:
            if self.config.model_input_file is None:
                system_model = Windpower.default(self.config_name)
                financial_model = Singleowner.from_existing(system_model, self.config_name)
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

                financial_model = Singleowner.from_existing(system_model, self.config_name)

        # Parse user input for financial model
        if isinstance(self.config.fin_model, str):
            financial_model = Singleowner.default(self.config.fin_model)
        elif isinstance(self.config.fin_model, dict):
            financial_model = CustomFinancialModel(self.config.fin_model)

        if isinstance(self.config.layout_params, dict):
            layout_params = WindBoundaryGridParameters(**self.config.layout_params)
        else:
            layout_params = self.config.layout_params

        super().__init__("WindPlant", self.site, system_model, financial_model)
        self._system_model.value("wind_resource_data", self.site.wind_resource.data)

        self._layout = WindLayout(self.site, system_model, self.config.layout_mode, layout_params)

        self._dispatch = None

        self.turb_rating = self.config.turbine_rating_kw
        self.num_turbines = self.config.num_turbines

        if self.config.hub_height is not None:
            self._system_model.Turbine.wind_turbine_hub_ht = self.config.hub_height
        if self.config.rotor_diameter is not None:
            self.rotor_diameter = self.config.rotor_diameter
            
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
        self._system_model.value("wind_turbine_powercurve_powerout",
            [i * scaling for i in self._system_model.value("wind_turbine_powercurve_powerout")])
        self._system_model.value("system_capacity", self.turb_rating * len(self._system_model.value("wind_farm_xCoordinates")))

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
            self._system_model.Turbine.calculate_powercurve(rating_kw,
                                                            int(self._system_model.value("wind_turbine_rotor_diameter")),
                                                            elevation,
                                                            wind_default_max_cp,
                                                            wind_default_max_tip_speed,
                                                            wind_default_max_tip_speed_ratio,
                                                            wind_default_cut_in_speed,
                                                            wind_default_cut_out_speed,
                                                            wind_default_drive_train)
            logger.info("WindPlant recalculated powercurve")
        except:
            raise RuntimeError("WindPlant.turb_rating could not calculate turbine powercurve with diameter={}"
                               ", rating={}. Check diameter or turn off 'recalculate_powercurve'".
                               format(rotor_diam, rating_kw))
        self._system_model.value("wind_turbine_rotor_diameter", rotor_diam)
        self._system_model.value("system_capacity", rating_kw * self.num_turbines)
        logger.info("WindPlant set system_capacity to {} kW".format(self.system_capacity_kw))

    def modify_coordinates(self, xcoords: Sequence, ycoords: Sequence):
        """
        Change the location of the turbines
        """
        if len(xcoords) != len(ycoords):
            raise ValueError("WindPlant turbine coordinate arrays must have same length")
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
        Sets the system capacity by adjusting the rating of the turbines within the provided boundaries

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
