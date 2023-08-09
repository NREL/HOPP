import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from hopp.layout.flicker_mismatch_grid import FlickerMismatch, FlickerMismatchGrid
from hopp.layout.flicker_data.plot_flicker import plot_contour
from hopp.keys import set_developer_nrel_gov_key


# Set API key
load_dotenv()
NREL_API_KEY = os.getenv("NREL_API_KEY")
set_developer_nrel_gov_key(NREL_API_KEY)  # Set this key manually here if you are not setting it using the .env


def plot_flicker(flicker_model, maps, titles):
    for m, t in zip(maps, titles):
        axs = flicker_model.plot_on_site(False, False)
        c = plot_contour(m, flicker_model, axs)
        plt.colorbar(c)
        plt.title(t)
        plt.show()


if __name__ == '__main__':
    lat = 39.7555
    lon = -105.2211

    #
    # Part 1: Time-based Flicker map for a Single Turbine
    #

    # switch to full-size mesh and 15-min timesteps
    FlickerMismatch.diam_mult_nwe = 8
    FlickerMismatch.diam_mult_s = 4
    FlickerMismatch.steps_per_hour = 1

    # run flicker calculation of just the blades
    FlickerMismatch.turbine_tower_shadow = False

    # using ellipse as swept area
    flicker_no_tower = FlickerMismatch(lat, lon, blade_length=45, angles_per_step=None,
                                    gridcell_height=90, gridcell_width=90, gridcells_per_string=1)

    # to run for whole year, do not provide any intervals-- they will be automatically calculated
    (flicker_hours_annual, ) = flicker_no_tower.run_parallel(1, ("time", ))

    plot_flicker(flicker_no_tower, (flicker_hours_annual, ), ("Flicker weighted by Time",))


    #
    # Part 2: POA- and Power-based Flicker maps for a Single Turbine
    #

    # single turbine on a mini-mesh that is only 3 turbine diameters wide to the north, west and east, and 1 to the south
    FlickerMismatch.diam_mult_nwe = 3
    FlickerMismatch.diam_mult_s = 1
    FlickerMismatch.turbine_tower_shadow = True

    # run one blade angles per timestep and create heatmaps of plane-of-array and pv power losses
    flicker_single = FlickerMismatch(lat, lon, angles_per_step=1)

    # only run a few hours of the year; normalization will be based on these timesteps only
    hours_to_run = range(3186, 3208)
    shadow, loss = flicker_single.create_heat_maps(hours_to_run, ("poa", "power"))
    plot_flicker(flicker_single, (shadow, loss), ("Flicker weighted by POA Loss", "Flicker weighted by Power Loss"))


    #
    # Part 3: POA- and Power-based Flicker maps for a Grid of Turbines with Multiprocessing
    #

    # multiple turbines in a 45 degree grid spaced out 3 turbine diameters apart both dimensions
    grid_dx_diams = 3
    grid_dy_diams = 3
    grid_degrees = 45
    flicker_grid = FlickerMismatchGrid(lat, lon, grid_dx_diams, grid_dy_diams, grid_degrees)

    # split up the hours into two, one per processor
    hours_to_run = [range(3186, 3197), range(3197, 3208)]
    shadow_grid, loss_grid = flicker_grid.run_parallel(2, ("poa", "power"), hours_to_run)
    plot_flicker(flicker_single, (shadow, loss), ("Flicker weighted by POA Loss", "Flicker weighted by Power Loss"))
