import numpy as np
import openmdao.api as om

import electrolyzer.inputs.validation as val
from electrolyzer import run_lcoh

from shapely.geometry import Polygon, Point

class HOPPComponent(om.ExplicitComponent):

    def initialize(self):
        self.options.declare("hi", recordable=False)
        self.options.declare("turbine_x_init")
        self.options.declare("turbine_y_init")
        self.options.declare("verbose")

    def setup(self):

        self.add_input("turbine_x", val=self.options["turbine_x_init"], units="m")
        self.add_input("turbine_y", val=self.options["turbine_y_init"], units="m")
        # self.add_input("battery_capacity_kw", val=15000, units="kW")
        # self.add_input("battery_capacity_kwh", val=15000, units="kW*h")
        # self.add_input("electrolyzer_rating_kw", val=15000, units="kW*h")
        # self.add_input("pv_rating_kw", val=15000, units="kW")
        # self.add_input("wind_rating_kw", val=150000, units="kW")
        self.add_output("aep", units="kW*h")
        self.add_output("lcoe_real", units="USD/(MW*h)")
        self.add_output("p_wind", units="kW", val=np.zeros(8760))

    def compute(self, inputs, outputs):

        if self.options["verbose"]:
            print("reinitialize")
            print("x: ", inputs["turbine_x"])
            print("y: ", inputs["turbine_y"])

        hi = self.options["hi"]

        hi.system.wind._system_model.fi.reinitialize(layout_x=inputs["turbine_x"], layout_y=inputs["turbine_y"], time_series=True)

        # run simulation
        hi.simulate(25)

        if self.options["verbose"]:
            print(f"obj: {hi.system.annual_energies.hybrid}")

        # get result
        outputs["aep"] = hi.system.annual_energies.hybrid
        outputs["lcoe_real"] = hi.system.lcoe_real.hybrid/100.0 # convert from cents/kWh to USD/kWh
        outputs["p_wind"] = hi.system.grid.generation_profile[0:8760]

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd', form="forward")

class TurbineDistanceComponent(om.ExplicitComponent):

    def initialize(self):
        self.options.declare("turbine_x_init")
        self.options.declare("turbine_y_init")

    def setup(self):
        n_turbines = len(self.options["turbine_x_init"])
        n_distances = int(n_turbines*(n_turbines-1)/2)
        self.add_input("turbine_x", val=self.options["turbine_x_init"], units="m")
        self.add_input("turbine_y", val=self.options["turbine_y_init"], units="m")
        self.add_output("spacing_vec", val=np.zeros(n_distances), units="m")

    def compute(self, inputs, outputs):

        n_turbines = len(inputs["turbine_x"])
        spacing_vec = np.zeros(int(n_turbines*(n_turbines-1)/2))
        k = 0
        for i in range(len(inputs["turbine_x"])):
            for j in range(i+1, len(inputs["turbine_y"])):  # only look at pairs once
                spacing_vec[k] = np.linalg.norm([(inputs["turbine_x"][i]-inputs["turbine_x"][j]), (inputs["turbine_y"][i]-inputs["turbine_y"][j])])
                k += 1
        outputs["spacing_vec"] = spacing_vec

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd', form='forward')

class BoundaryDistanceComponent(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare("hopp_interface", recordable=False)
        self.options.declare("turbine_x_init")
        self.options.declare("turbine_y_init")

    def setup(self):
        self.n_turbines = len(self.options["turbine_x_init"])
        self.n_distances = self.n_turbines
        self.add_input("turbine_x", val=self.options["turbine_x_init"], units="m")
        self.add_input("turbine_y", val=self.options["turbine_y_init"], units="m")
        self.add_output("boundary_distance_vec", val=np.zeros(self.n_distances), units="m")

    def compute(self, inputs, outputs):

        # get hopp interface
        hi = self.options["hopp_interface"]

        # get polygon for boundary
        boundary_polygon = Polygon(hi.system.site.vertices)     
        # check if turbines are inside polygon and get distance
        for i in range(0, self.n_distances):
            point = Point(inputs["turbine_x"][i], inputs["turbine_y"][i])
            outputs["boundary_distance_vec"][i] = boundary_polygon.exterior.distance(point)
            if not boundary_polygon.contains(point):
                outputs["boundary_distance_vec"][i] *= -1

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd', form='forward')

class ElectrolyzerComponent(om.ExplicitComponent):
    """
    This is an OpenMDAO wrapper to the generic electrolyzer model.

    It makes some assumptions about the number of electrolyzers, stack size, and
    how to distribute electricity across the different electrolyzers. These
    could be later made into modeling options to allow for more user configuration.
    """
    def initialize(self):
        self.options.declare("h2_modeling_options")
        self.options.declare("h2_opt_options")
        self.options.declare("modeling_options")

    def setup(self):
        self.add_input("p_wind", val=np.zeros(8760), units="W")
        self.add_input("lcoe_real", units="USD/kW/h")
        if self.options["h2_opt_options"]["control"]["system_rating_MW"]["flag"] \
            or self.options["modeling_options"]["rating_equals_turbine_rating"]:
            self.add_input("system_rating_MW", units="MW", val=self.options["h2_modeling_options"]["electrolyzer"]["control"]["system_rating_MW"])
        self.add_output("h2_produced", units="kg")
        self.add_output("max_curr_density", units="A/cm**2")
        self.add_output("capex", units="USD")
        self.add_output("opex", units="USD")
        self.add_output("lcoh", units="USD/kg")

    def compute(self, inputs, outputs):
        # Set electrolyzer parameters from model inputs
        power_signal = inputs["p_wind"]
        lcoe_real = inputs["lcoe_real"][0]

        if self.options["h2_opt_options"]["control"]["system_rating_MW"]["flag"] \
            or self.options["modeling_options"]["rating_equals_turbine_rating"]:
            self.options["h2_modeling_options"]["electrolyzer"]["control"]["system_rating_MW"] = inputs["system_rating_MW"][0]

        elif "electrolyzer_rating_MW" in self.options["modeling_options"]["overridden_values"]:
            electrolyzer_rating_MW = self.options["modeling_options"]["overridden_values"]["electrolyzer_rating_MW"]
            self.options["h2_modeling_options"]["electrolyzer"]["control"]["system_rating_MW"] = electrolyzer_rating_MW

        h2_prod, max_curr_density, lcoh, lcoh_dict, _  = run_lcoh(
            self.options["h2_modeling_options"],
            power_signal,
            lcoe_real,
            optimize=True
        )

        lt = lcoh_dict["LCOH Breakdown"]["Life Totals [$]"]
        capex = lt["CapEx"]
        opex = lt["OM"]

        # msg = (
        #     f"\n====== Electrolyzer ======\n"
        #     f"  - h2 produced (kg): {h2_prod}\n"
        #     f"  - max current density (A/cm^2): {max_curr_density}\n"
        #     f"  - LCOH ($/kg): {lcoh}\n"
        # )

        # logger.info(msg)

        outputs["h2_produced"] = h2_prod
        outputs["max_curr_density"] = max_curr_density
        outputs["capex"] = capex
        outputs["opex"] = opex
        outputs["lcoh"] = lcoh

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd', form='forward')
