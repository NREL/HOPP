import matplotlib.pyplot as plt
from hybrid.solar_wind.flicker_mismatch_grid import FlickerMismatch
from hybrid.solar_wind.data.plot_flicker import plot_contour


lat = 39.7555
lon = -105.2211

hours_to_run = range(3186, 3208)

FlickerMismatch.diam_mult_nwe = 3
FlickerMismatch.diam_mult_s = 1
flicker = FlickerMismatch(lat, lon, angles_per_step=1)
shadow, loss = flicker.create_heat_maps_irradiance(hours_to_run)

axs = flicker.plot_on_site(False, False)
plot_contour(shadow, flicker, axs)
plt.title("Shadow Weighted by POA")
plt.show()

axs = flicker.plot_on_site(False, False)
plot_contour(loss, flicker, axs)
plt.title("Flicker")
plt.show()
