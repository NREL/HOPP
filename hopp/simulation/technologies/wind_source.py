from attrs import define, field
from typing import Optional, Union, Sequence

import PySAM.Windpower as Windpower
import PySAM.Singleowner as Singleowner

from hopp.simulation.base import BaseClass, BaseModel
from hopp.simulation.technologies.wind.floris import Floris
from hopp.simulation.technologies.power_source import PowerSource
from hopp.simulation.technologies.wind.pysam_wind import PySAMWind
from hopp.simulation.technologies.wind.pysam_financial import PySAMFinancial
from hopp.simulation.technologies.financial.custom_financial_model import CustomFinancialModel
from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.technologies.layout.wind_layout import WindLayout, WindBoundaryGridParameters
from hopp.simulation.technologies.dispatch.power_sources.wind_dispatch import WindDispatch
from hopp.utilities.log import hybrid_logger as logger


MODEL_MAP = {
    "wind_simulation_model": {
        "pysam": PySAMWind,
        "floris": Floris,
    },
    "wind_financial_model": {
        "pysam": PySAMFinancial,
    }
}

@define
class WindPlant(PowerSource):
    site: SiteInfo = field()
    farm_config: dict = field(converter=dict)
    rating_range_kw: tuple = field(default=(1000, 3000))

    _system_model: Union[PySAMWind, Floris] = field(init=False)
    _financial_model: Singleowner.Singleowner = field(init=False)
    _layout: WindLayout = field(init=False)
    _dispatch: WindDispatch = field(init=False)

    def __attrs_post_init__(self) -> None:
        """
        Set up a WindPlant

        :param farm_config: dict, with keys ('num_turbines', 'turbine_rating_kw', 'rotor_diameter',
        'hub_height', 'layout_mode', 'layout_params')
            where layout_mode can be selected from the following:
            - 'boundarygrid': regular grid with boundary turbines, requires
            WindBoundaryGridParameters as 'params'
            - 'grid': regular grid with dx, dy distance, 0 angle; does not require 'params'

        :param rating_range_kw:
            allowable kw range of turbines, default is 1000 - 3000 kW
        """
        wind_simulation_model_string = self.farm_config["simulation_model"].lower()
        model: BaseClass = MODEL_MAP["wind_simulation_model"][wind_simulation_model_string]
        system_model = model(self.farm_config, self.site, timestep=self.farm_config['timestep'])

        # wind_financial_model_string = self.farm_config["financial_model"]
        # model: BaseClass = MODEL_MAP["wind_financial_model"][wind_financial_model_string]
        # financial_model = model(self.farm_config).financial_model
        self.config_name = "WindPowerSingleOwner"
        financial_model = Singleowner.from_existing(system_model.system_model, self.config_name)

        if 'fin_model' in self.farm_config.keys():
            financial_model = self.import_financial_model(CustomFinancialModel(self.farm_config['fin_model']), system_model, self.config_name)

        super().__init__("WindPlant", self.site, system_model, financial_model)

        if 'layout_mode' not in self.farm_config.keys():
            layout_mode = 'grid'
        else:
            layout_mode = self.farm_config['layout_mode']

        params: Optional[Union[WindBoundaryGridParameters, dict]] = None
        if layout_mode == 'boundarygrid':
            if 'layout_params' not in self.farm_config.keys():
                raise ValueError(
                    "Parameters of WindBoundaryGridParameters required for boundarygrid "
                    "layout mode"
                )
            elif isinstance(self.farm_config['layout_params'], dict):
                params = WindBoundaryGridParameters(**self.farm_config['layout_params'])
            elif isinstance(self.farm_config['layout_params'], WindBoundaryGridParameters):
                params = self.farm_config['layout_params']
            else:
                raise TypeError("farm_config['layout_params'] non-supported type")

        self._layout = WindLayout(self.site, system_model, layout_mode, params)

        self._dispatch: WindDispatch = None

        if 'turbine_rating_kw' not in self.farm_config.keys():
            raise ValueError("Turbine rating required for WindPlant")

        if 'num_turbines' not in self.farm_config.keys():
            raise ValueError("Num Turbines required for WindPlant")

        self.turb_rating = self.farm_config['turbine_rating_kw']
        self.num_turbines = self.farm_config['num_turbines']
        if 'hub_height' in self.farm_config.keys():
            self._system_model.Turbine.wind_turbine_hub_ht = self.farm_config['hub_height']
        if 'rotor_diameter' in self.farm_config.keys():
            self.rotor_diameter = self.farm_config['rotor_diameter']

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
        self._system_model.value(
            "system_capacity",
            self.turb_rating * len(self._system_model.value("wind_farm_xCoordinates"))
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
                wind_default_drive_train
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
        Sets the system capacity by adjusting the rating of the turbines within the provided
        boundaries

        :param wind_size_kw: desired system capacity in kW
        """
        turb_rating_kw = wind_size_kw / self.num_turbines
        if self.rating_range_kw[0] <= turb_rating_kw <= self.rating_range_kw[1]:
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
