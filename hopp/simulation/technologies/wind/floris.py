from pathlib import Path
from typing import TYPE_CHECKING, Tuple

from attrs import define, field
import numpy as np

from floris import FlorisModel, TimeSeries
from floris.turbine_library.turbine_previewer import INTERNAL_LIBRARY
from hopp.simulation.base import BaseClass
from hopp.simulation.technologies.sites import SiteInfo
# avoid circular dep
if TYPE_CHECKING:
    from hopp.simulation.technologies.wind.wind_plant import WindConfig
from hopp.tools.resource.wind_tools import (
    calculate_air_density, 
    parse_resource_data,
    weighted_parse_resource_data
)
from hopp.utilities import load_yaml
from hopp.utilities.log import hybrid_logger as logger


@define
class Floris(BaseClass):
    site: SiteInfo = field()
    config: "WindConfig" = field()
    verbose: bool = field(default = True)

    _operational_losses: float = field(init=False)
    _timestep: Tuple[int, int] = field(init=False)
    fi: FlorisModel = field(init=False)
    
    # turbine parameters
    turbine_name: str = field(init = False)
    wind_turbine_rotor_diameter: float = field(init = False)
    turb_rating: float = field(init = False)
    # turbine power curve (array of kW power outputs)
    wind_turbine_powercurve_powerout: list[float] = field(init = False)
    wind_farm_xCoordinates: list[float] = field(init = False)
    wind_farm_yCoordinates: list[float] = field(init = False)
    system_capacity: float = field(init = False)
    
    #results
    gen: list[float] = field(init = False)
    annual_energy: float = field(init = False)
    capacity_factor: float = field(init = False)
    annual_energy_pre_curtailment_ac: float = field(init = False)
    
    #TODO: add option to store turbine-powers and velocities or not
    turb_velocities: np.ndarray = field(init = False)
    turb_powers: np.ndarray = field(init = False)

    def __attrs_post_init__(self):
        """Set-up and initialize floris_config and floris model. This method does the following:

        1) check that floris config is provided
        2) load floris config if needed
        3) modify air density in floris config if needed
        4) initialize attributes from floris config and update floris config as needed
        5) initialize floris model

        Raises:
            ValueError: "A floris configuration must be provided"
            ValueError: "A timestep is required."
        """
        
        if self.config.floris_config is None:
            raise ValueError("A floris configuration must be provided")
        if self.config.timestep is None:
            raise ValueError("A timestep is required.")

        if isinstance(self.config.floris_config,(str, Path)):
            floris_config = load_yaml(self.config.floris_config)
        else:
            floris_config = self.config.floris_config

        if self.config.adjust_air_density_for_elevation and self.site.elev is not None:
            rho = calculate_air_density(self.site.elev)
            floris_config["flow_field"].update({"air_density":rho})
        
        floris_config = self.initialize_from_floris(floris_config)
        
        self.fi = FlorisModel(floris_config)
        self._timestep = self.config.timestep
        self._operational_losses = self.config.operational_losses
        
        if self.config.resource_parse_method == "average":
            self.speeds, self.wind_dirs = parse_resource_data(self.site.wind_resource)
        elif self.config.resource_parse_method == "weighted_average":
            self.speeds, self.wind_dirs = weighted_parse_resource_data(self.site.wind_resource)

        self.system_capacity = self.nTurbs * self.turb_rating

        # time to simulate
        if len(self.config.timestep) > 0:
            self.start_idx = self.config.timestep[0]
            self.end_idx = self.config.timestep[1]
        else:
            self.start_idx = 0
            self.end_idx = 8759
        

    def initialize_from_floris(self,floris_config):
        """
        Please populate all the wind farm parameters
        """
        
        if self.config.turbine_name is None:
            # NOTE: eventually the turbine name provided in the config will be used 
            # to load a turbine from the turbine-models library.
            if isinstance(floris_config["farm"]["turbine_type"][0],dict):
                self.turbine_name = floris_config["farm"]["turbine_type"][0]["turbine_type"]

            # load file from internal floris library
            if isinstance(floris_config["farm"]["turbine_type"][0],str):
                self.turbine_name = floris_config["farm"]["turbine_type"][0]
                turb_dict = load_yaml(
                    INTERNAL_LIBRARY / "{}.yaml".format(floris_config["farm"]["turbine_type"][0])
                )
                floris_config["farm"]["turbine_type"][0] = turb_dict

        # see if rotor diameter was input in config but not set in floris config
        if self.config.rotor_diameter is not None:
            floris_config["farm"]["turbine_type"][0].setdefault(
                "rotor_diameter",self.config.rotor_diameter
            )
        # see if hub-height was input in config but not set in floris config
        if self.config.hub_height is not None:
            floris_config["farm"]["turbine_type"][0].setdefault(
                "hub_height", self.config.hub_height
            )
        # NOTE: hub-height should also be checked against wind resource hub-height

        # set attributes:
        self.wind_turbine_rotor_diameter = floris_config["farm"]["turbine_type"][0]["rotor_diameter"]
        self.wind_turbine_powercurve_powerout = floris_config["farm"]["turbine_type"][0]["power_thrust_table"]["power"]
        self.wind_farm_xCoordinates = floris_config["farm"]["layout_x"]
        self.wind_farm_yCoordinates = floris_config["farm"]["layout_y"]
        self.nTurbs = len(self.wind_farm_xCoordinates)  

        self.turb_rating = max(self.wind_turbine_powercurve_powerout)
        if self.config.turbine_rating_kw is not None:
            if self.config.turbine_rating_kw != self.turb_rating:
                raise UserWarning(
                    f"Input turbine rating ({self.config.turbine_rating_kw} kW) does not match "
                    f"rating from floris power-curve ({self.turb_rating} kW)"
                )
        
        # check if user-input num_turbines equals number of turbines in layout
        if self.nTurbs != self.config.num_turbines:
            logger.warning(
                f"num_turbines in WindConfig ({self.config.num_turbines}) does not equal "
                f"number of turbines in floris config layout ({self.nTurbs})"
            )
        return floris_config

    def value(self, name: str, set_value=None):
        """Set or retrieve attribute of `hopp.simulation.technologies.wind.floris.Floris`.
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

    def set_floris_value(self, name, value):
        if value is not None:
            self.fi.set(**{name:value})

    def set_floris_param(self, param, value):
        if value is not None:
            self.fi.set_param(param, value)

    def get_floris_param(self, param):
        return self.fi.get_param(param)

    def execute(self, project_life):
        """Simulate wind farm performance using floris.

        Args:
            project_life (int): unused project life in years
        """
        
        if self.verbose:
            print('Simulating wind farm output in FLORIS...')

        # check if user-input num_turbines equals number of turbines in layout
        if self.nTurbs != self.config.num_turbines:
            # Log warning if discrepancy in number of turbines.
            # Not raising a warning since wind farm capacity can be modified 
            # before simulation begins.
            logger.warning(
                f"num_turbines input in WindConfig ({self.config.num_turbines}) does not equal "
                f"number of turbines in floris model ({self.nTurbs})"
            )
        logger.info(f"simulating {self.nTurbs} turbines using FLORIS")

        # find generation of wind farm
        power_turbines = np.zeros((self.nTurbs, 8760))
        power_farm = np.zeros(8760)

        time_series = TimeSeries(
            wind_directions=self.wind_dirs[self.start_idx:self.end_idx],
            wind_speeds=self.speeds[self.start_idx:self.end_idx],
            turbulence_intensities=self.fi.core.flow_field.turbulence_intensities[0]
        )

        self.fi.set(wind_data=time_series)
        self.fi.run()

        power_turbines[:, self.start_idx:self.end_idx] = self.fi.get_turbine_powers().reshape(
            (self.nTurbs, self.end_idx - self.start_idx)
        )
        power_farm[self.start_idx:self.end_idx] = self.fi.get_farm_power().reshape(
            (self.end_idx - self.start_idx)
        )

        operational_efficiency = ((100 - self._operational_losses)/100)
        # Adding losses from PySAM defaults (excluding turbine and wake losses)
        self.gen = power_farm * operational_efficiency / 1000 # kW

        self.annual_energy = np.sum(self.gen) # kWh
        self.capacity_factor = np.sum(self.gen) / (8760 * self.system_capacity) * 100
        self.turb_powers = power_turbines * operational_efficiency / 1000 # kW
        self.turb_velocities = self.fi.turbine_average_velocities
        self.annual_energy_pre_curtailment_ac = np.sum(self.gen) # kWh

    def export(self):
        """
        Return all the floris system configuration in a dictionary for the financial model
        """
        import pdb; pdb.set_trace()
        config = {
            'system_capacity': self.system_capacity,
            'annual_energy': self.annual_energy,
        }
        return config

    @property
    def wind_farm_layout(self):
        xcoords, ycoords = self.fi.get_turbine_layout()
        return xcoords, ycoords

    def set_wind_farm_layout(self, xcoords, ycoords):
        """
        Sets the wind farm layout and updates relevant parameters.

        Args:
            xcoords (list[float]): A list of x-coordinates for turbine locations.
            ycoords (list[float]): A list of y-coordinates for turbine locations.

        Raises:
            ValueError: If x- and y-coordinates are not the same length, an error is raised.
        """
        if len(xcoords) != len(ycoords):
            raise ValueError("WindPlant turbine coordinate arrays must have same length")
        self.fi.set(layout_x=xcoords, layout_y=ycoords)
        self.nTurbs = len(xcoords)
        self.system_capacity = len(xcoords) * self.turb_rating
        self.value("wind_farm_xCoordinates", xcoords)
        self.value("wind_farm_yCoordinates", ycoords)     
