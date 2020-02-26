import logging
import math
from typing import Sequence

import numpy as np
from shapely.affinity import scale
from shapely.geometry import Point, Polygon

import PySAM.Windpower as Windpower

from hybrid.site_info import SiteInfo
from hybrid.power_source import *


class WindPlant(PowerSource):
    system_model: Windpower.Windpower
    financial_model: Singleowner.Singleowner

    def __init__(self,
                 site: SiteInfo,
                 system_capacity_kw : float,
                 rating_range_kw: tuple = (1000, 3000),
                 grid_not_row_layout: bool = False):
        """

        :param system_capacity_kw:
        :param grid_not_row_layout:
            make a regular grid instead of a row whose layout is irrespective of site boundaries
        :param size_adjustment:
            'n_turb': adjust system capacity size by adding or removing turbines
            'rating': adjust system capacity size by rating within range, then change n_turbs
        :param rating_range_kw:
            allowable kw range of turbines, default is 1000 - 3000 kW
        """
        super().__init__(site)

        self._rating_range_kw = rating_range_kw

        self.system_model = Windpower.default("WindPowerSingleOwner")
        self.financial_model = Singleowner.from_existing(self.system_model, "WindPowerSingleOwner")

        self.system_model.Resource.wind_resource_data = self.site.wind_resource.data

        self.total_installed_cost_dollars = 0
        self._construction_financing_cost_per_kw = self.financial_model.FinancialParameters.construction_financing_cost\
                                                   / self.financial_model.FinancialParameters.system_capacity
        self.financial_model.Revenue.ppa_soln_mode = 1

        self._grid_not_row_layout = grid_not_row_layout
        self.row_spacing = 5 * self.system_model.Turbine.wind_turbine_rotor_diameter
        self.grid_spacing = None

        self.system_capacity_closest_fit(system_capacity_kw)

    @property
    def wake_model(self) -> str:
        model_type = self.system_model.Farm.wind_farm_wake_model
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

    @wake_model.setter
    def wake_model(self, model_type: int):
        if 0 <= model_type < 4:
            self.system_model.Farm.wind_farm_wake_model = model_type

    @property
    def num_turbines(self):
        return len(self.system_model.Farm.wind_farm_xCoordinates)

    @num_turbines.setter
    def num_turbines(self, n_turbines: int):
        if self._grid_not_row_layout:
            self.set_num_turbines_in_grid(n_turbines)
        else:
            self.set_num_turbines_in_row(n_turbines)

    def set_num_turbines_in_grid(self, n_turbines: int):
        """
        Set the number of turbines. System capacity gets modified as a result.
        Wind turbines will be placed in a row

        :param n_turbines: int
        """
        xcoords = []
        ycoords = []
        if not self.site.polygon:
            raise ValueError("WindPlant set_num_turbines_in_grid requires site polygon")
        spacing = math.sqrt(self.site.polygon.area / n_turbines) * self.site.polygon.envelope.area / self.site.polygon.area
        spacing = max(spacing, self.rotor_diameter * 3)
        coords = []
        while len(coords) < n_turbines:

            envelope = Polygon(self.site.polygon.envelope)
            while len(coords) < n_turbines and envelope.area > spacing * spacing:
                d = 0
                sub_boundary = envelope.boundary
                while d <= sub_boundary.length and len(coords) < n_turbines:
                    coord = sub_boundary.interpolate(d)
                    if self.site.polygon.buffer(1e3).contains(coord):
                        coords.append(coord)
                    d += spacing
                if len(coords) < n_turbines:
                    envelope = scale(envelope, (envelope.bounds[2] - spacing)/envelope.bounds[2],
                                     (envelope.bounds[3] - spacing)/envelope.bounds[3])
            if len(coords) < n_turbines:
                spacing *= .95
                coords = []
        for _, p in enumerate(coords):
            xcoords.append(p.x)
            ycoords.append(p.y)
        self.system_model.Farm.wind_farm_xCoordinates = xcoords
        self.system_model.Farm.wind_farm_yCoordinates = ycoords
        self._grid_not_row_layout = True
        self.system_model.Farm.system_capacity_kw = n_turbines * self.turb_rating
        logger.info("WindPlant set num turbines to {} in grid". format(n_turbines))
        logger.debug("WindPlant set xcoords to {}".format(xcoords))
        logger.debug("WindPlant set ycoords to {}".format(ycoords))
        logger.info("WindPlant set system_capacity to {} kW".format(self.system_capacity_kw))

    def set_num_turbines_in_row(self, n_turbines: int, spacing: float = None, angle_deg: float = 0):
        """
        Set the number of turbines by placing wind turbines will be placed in a row with given angle
        If spacing is not provided, original spacing is used.
        If angle is not provided, 0 is used (horizontal row)
        System capacity gets modified as a result.
        """
        xcoords = []
        ycoords = []
        if spacing:
            self.row_spacing = max(spacing, self.rotor_diameter * 3)
            logger.info("WindPlant set row spacing to {}".format(self.row_spacing))
        dx = self.row_spacing * np.cos(np.radians(angle_deg))
        dy = self.row_spacing * np.sin(np.radians(angle_deg))
        x0 = 0
        y0 = 0
        if self.site.polygon:
            x0 = self.site.polygon.bounds[0]
            y0 = self.site.polygon.bounds[1]
        for i in range(n_turbines):
            turb = Point((x0 + i * dx, y0 + i * dy))
            if self.site.polygon:
                if not self.site.polygon.contains(turb):
                    logger.warning("WindPlant turbine at {} outside of site boundary".format(turb))
            xcoords.append(turb.x)
            ycoords.append(turb.y)

        self.system_model.Farm.wind_farm_xCoordinates = xcoords
        self.system_model.Farm.wind_farm_yCoordinates = ycoords
        self._grid_not_row_layout = False
        self.system_model.Farm.system_capacity = n_turbines * self.turb_rating
        logger.info("WindPlant set num turbines to {} in row". format(n_turbines))
        logger.debug("WindPlant set xcoords to {}".format(xcoords))
        logger.debug("WindPlant set ycoords to {}".format(ycoords))
        logger.info("WindPlant set system_capacity to {} kW".format(self.system_capacity_kw))

    @property
    def rotor_diameter(self):
        return self.system_model.Turbine.wind_turbine_rotor_diameter

    @rotor_diameter.setter
    def rotor_diameter(self, d):
        self.system_model.Turbine.wind_turbine_rotor_diameter = d
        logger.info("WindPlant set rotor diameter to {} m".format(d))
        # recalculate layout spacing
        self.num_turbines = self.num_turbines

    @property
    def turb_rating(self):
        """
        :return: kw rating of turbine
        """
        return max(self.system_model.Turbine.wind_turbine_powercurve_powerout)

    @turb_rating.setter
    def turb_rating(self, rating_kw):
        """
        Set the turbine rating. System capacity gets modified as a result.
        Turbine powercurve will be recalculated according to one of the following methods:

        :param rating_kw: float
        """
        scaling = rating_kw / self.turb_rating
        self.system_model.Turbine.wind_turbine_powercurve_powerout = \
            [i * scaling for i in self.system_model.Turbine.wind_turbine_powercurve_powerout]
        logger.info("WindPlant set turb_rating to {} kW".format(rating_kw))
        self.system_model.Farm.system_capacity = self.turb_rating * self.num_turbines
        logger.info("WindPlant set system_capacity to {} kW".format(self.system_capacity_kw))

    def modify_powercurve(self, rotor_diam, rating_kw):
        """
        Recalculate the turbine power curve
        :param rotor_diam: meters
        :param rating_kw: kw
        :return:
        """
        wind_default_max_tip_speed = 80
        wind_default_max_tip_speed_ratio = 8
        wind_default_cut_in_speed = 4
        wind_default_cut_out_speed = 25
        wind_default_drive_train = 0
        self.system_model.Turbine.wind_turbine_rotor_diameter = rotor_diam
        try:
            # could fail if current rotor diameter is too big or small for rating
            self.system_model.Turbine.calculate_powercurve(rating_kw,
                                                           self.system_model.Turbine.wind_turbine_rotor_diameter,
                                                           wind_default_max_tip_speed,
                                                           wind_default_max_tip_speed_ratio,
                                                           wind_default_cut_in_speed,
                                                           wind_default_cut_out_speed,
                                                           wind_default_drive_train)
            logger.info("WindPlant recalculated powercurve")
        except:
            raise RuntimeError("WindPlant.turb_rating could not calculate turbine powercurve with diameter=" + str(rotor_diam)
                               + ", rating=" + str(rating_kw) + ". Check diameter or turn off 'recalculate_powercurve'")
        self.system_model.Farm.system_capacity = rating_kw * self.num_turbines
        logger.info("WindPlant set system_capacity to {} kW".format(self.system_capacity_kw))

    def modify_coordinates(self, xcoords: Sequence, ycoords: Sequence):
        """
        Change the location of the turbines
        """
        if len(xcoords) != len(ycoords):
            raise ValueError("WindPlant turbine coordinate arrays must have same length")
        self.system_model.Farm.wind_farm_xCoordinates = xcoords
        self.system_model.Farm.wind_farm_yCoordinates = ycoords
        self.system_model.Farm.system_capacity = self.turb_rating * len(xcoords)
        logger.debug("WindPlant set xcoords to {}".format(xcoords))
        logger.debug("WindPlant set ycoords to {}".format(ycoords))
        logger.info("WindPlant set system_capacity to {} kW".format(self.system_capacity_kw))

    @property
    def system_capacity_kw(self):
        return self.system_model.Farm.system_capacity

    def system_capacity_closest_fit(self, wind_size_kw: float):
        if wind_size_kw == 0:
            self.num_turbines = 0
            return
        new_rating = round(wind_size_kw / self.num_turbines)
        if self._rating_range_kw[0] < new_rating < self._rating_range_kw[1]:
            self.turb_rating = new_rating
        else:
            new_rating = math.ceil(new_rating / 100) * 100
            new_n_turbs = self.num_turbines
            n_its = 0
            while new_n_turbs != wind_size_kw / new_rating and n_its < 100:
                if new_rating < self._rating_range_kw[0]:
                    new_rating += 100
                else:
                    new_rating -= 100

                new_n_turbs = round(wind_size_kw / new_rating)
                n_its += 1
            if n_its == 100:
                raise RuntimeError("Could not solve for n_turbs and rating pair for size of " + str(wind_size_kw))
            self.turb_rating = new_rating
            self.num_turbines = new_n_turbs

    def system_capacity_by_rating(self, wind_size_kw: float, turb_rating_kw: float):
        """
        Sets the system capacity by adjusting the rating of the turbines
        """
        if self._rating_range_kw[0] < turb_rating_kw < self._rating_range_kw[1]:
            self.turb_rating = turb_rating_kw
        else:
            logger.error("WindPlant could not meet target system_capacity by adjusting rating")
            raise RuntimeError("WindPlant could not meet target system_capacity")

    def system_capacity_by_num_turbines(self, wind_size_kw):
        """
        Sets the system capacity by adjusting the number of turbines

        :param wind_size_kw
        :return:
        """
        new_num_turbines = round(wind_size_kw / self.turb_rating)
        if self.num_turbines != new_num_turbines:
            self.num_turbines = new_num_turbines

    @property
    def total_installed_cost_dollars(self) -> float:
        return self.financial_model.SystemCosts.total_installed_cost

    @total_installed_cost_dollars.setter
    def total_installed_cost_dollars(self, total_installed_cost_dollars: float):
        self.financial_model.SystemCosts.total_installed_cost = total_installed_cost_dollars
        logger.info("WindPlant set total_installed_cost to ${}".format(self.total_installed_cost_dollars))

    @property
    def construction_financing_cost_per_kw(self):
        return self._construction_financing_cost_per_kw

    def simulate(self):
        self.system_model.execute(0)
        if self.system_capacity_kw > 0:
            self.financial_model.execute(0)
        logger.info("WindPlant simulation executed")

    def annual_energy_kw(self):
        if self.system_capacity_kw > 0:
            return self.system_model.Outputs.annual_energy
        else:
            return 0

    def generation_profile(self):
        if self.system_capacity_kw > 0:
            return self.system_model.Outputs.gen
        else:
            return [0] * self.site.n_timesteps

    def copy(self):
        raise NotImplementedError
