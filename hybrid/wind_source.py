from typing import Optional, Union, Sequence
import PySAM.Windpower as Windpower
import PySAM.Singleowner as Singleowner

try:
    from hybrid.add_custom_modules.custom_wind_floris import Floris
except:
    Floris = None

from hybrid.power_source import *
from hybrid.layout.wind_layout import WindLayout, WindBoundaryGridParameters
from hybrid.dispatch.power_sources.wind_dispatch import WindDispatch


class WindPlant(PowerSource):
    _system_model: Union[Windpower.Windpower, Floris]
    _financial_model: Singleowner.Singleowner
    _layout: WindLayout
    _dispatch: WindDispatch

    def __init__(self,
                 site: SiteInfo,
                 farm_config: dict,
                 rating_range_kw: tuple = (1000, 3000),
                 ):
        """
        Set up a WindPlant

        :param farm_config: dict, with keys ('num_turbines', 'turbine_rating_kw', 'rotor_diameter', 'hub_height', 'layout_mode', 'layout_params')
            where layout_mode can be selected from the following:
            - 'boundarygrid': regular grid with boundary turbines, requires WindBoundaryGridParameters as 'params'
            - 'grid': regular grid with dx, dy distance, 0 angle; does not require 'params'

        :param rating_range_kw:
            allowable kw range of turbines, default is 1000 - 3000 kW
        """
        self.config_name = "WindPowerSingleOwner"
        self._rating_range_kw = rating_range_kw

        if 'model_name' in farm_config.keys():
            if farm_config['model_name'] == 'floris':
                print('FLORIS is the system model...')
                system_model = Floris(farm_config, site, timestep=farm_config['timestep'])
                financial_model = Singleowner.default(self.config_name)
            else:
                raise NotImplementedError
        else:
            system_model = Windpower.default(self.config_name)
            financial_model = Singleowner.from_existing(system_model, self.config_name)

        if 'fin_model' in farm_config.keys():
            financial_model = self.import_financial_model(farm_config['fin_model'], system_model, self.config_name)

        super().__init__("WindPlant", site, system_model, financial_model)
        self._system_model.value("wind_resource_data", self.site.wind_resource.data)

        if 'layout_mode' not in farm_config.keys():
            layout_mode = 'grid'
        else:
            layout_mode = farm_config['layout_mode']

        params: Optional[WindBoundaryGridParameters] = None
        if layout_mode == 'boundarygrid':
            if 'layout_params' not in farm_config.keys():
                raise ValueError("Parameters of WindBoundaryGridParameters required for boundarygrid layout mode")
            else:
                params: WindBoundaryGridParameters = farm_config['layout_params']

        self._layout = WindLayout(site, system_model, layout_mode, params)

        self._dispatch: WindDispatch = None

        if 'turbine_rating_kw' not in farm_config.keys():
            raise ValueError("Turbine rating required for WindPlant")

        if 'num_turbines' not in farm_config.keys():
            raise ValueError("Num Turbines required for WindPlant")

        self.turb_rating = farm_config['turbine_rating_kw']
        self.num_turbines = farm_config['num_turbines']
        if 'hub_height' in farm_config.keys():
            self._system_model.Turbine.wind_turbine_hub_ht = farm_config['hub_height']
        if 'rotor_diameter' in farm_config.keys():
            self.rotor_diameter = farm_config['rotor_diameter']

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
