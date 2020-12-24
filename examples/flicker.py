import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from hybrid.flicker.flicker_mismatch_grid import FlickerMismatch, FlickerMismatchGrid
from hybrid.flicker.data.plot_flicker import plot_contour
from hybrid.keys import set_developer_nrel_gov_key

# Set API key
load_dotenv()
NREL_API_KEY = os.getenv("NREL_API_KEY")
set_developer_nrel_gov_key(NREL_API_KEY)  # Set this key manually here if you are not setting it using the .env


def plot_flicker(flicker_model, shadow_map, loss_map):
    axs = flicker_model.plot_on_site(False, False)
    plot_contour(shadow_map, flicker_model, axs)
    plt.title("Shadow Weighted by POA")
    plt.show()

    axs = flicker_model.plot_on_site(False, False)
    plot_contour(loss_map, flicker_model, axs)
    plt.title("Flicker")
    plt.show()


lat = 39.7555
lon = -105.2211
hours_to_run = range(3186, 3208)

# plot single turbine
FlickerMismatch.diam_mult_nwe = 3
FlickerMismatch.diam_mult_s = 1
flicker_single = FlickerMismatch(lat, lon, angles_per_step=1)
shadow, loss = flicker_single.create_heat_maps(hours_to_run, ("poa", "power"))
plot_flicker(flicker_single, shadow, loss)

# plot grid of turbines
grid_dx_diams = 3
grid_dy_diams = 3
grid_degrees = 45
hours_to_run = [range(3186, 3190), range(3190, 3208)]
flicker_grid = FlickerMismatchGrid(lat, lon, grid_dx_diams, grid_dy_diams, grid_degrees)
shadow_grid, loss_grid = flicker_grid.run_parallel(2, hours_to_run, ("poa", "power"))
plot_flicker(flicker_grid, shadow_grid, loss_grid)
