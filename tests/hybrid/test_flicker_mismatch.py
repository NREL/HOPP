import sys
sys.path.append('..')
from pytest import approx
import time
import numpy as np
import matplotlib.pyplot as plt
from hybrid.solar_wind.flicker_mismatch_grid import FlickerMismatchGrid, FlickerMismatch, create_heat_map_irradiance
from hybrid.solar_wind.data.plot_flicker import plot_contour, plot_heat_map

lat = 39.7555
lon = -105.2211


def test_single_turbine():
    FlickerMismatch.diam_mult_nwe = 3
    FlickerMismatch.diam_mult_s = 1
    flicker = FlickerMismatch(lat, lon, angles_per_step=1)
    shadow, loss = flicker.create_heat_maps_irradiance(range(3186, 3200))

    assert(np.max(shadow) == approx(0.00065403, 1e-4))
    assert(np.average(shadow) * 1e6 == approx(2.85111, 1e-4))
    assert(np.count_nonzero(shadow) == 917)
    assert(np.max(loss) == approx(0.207013, 1e-4))
    assert(np.average(loss) * 1e5 == approx(9.1324, 1e-4))
    assert(np.count_nonzero(loss) == 60)

    axs = flicker.plot_on_site(False, False)
    plot_contour(loss, flicker, axs)
    plt.title("Flicker Loss\n{}mod/str, periodic {}".
              format(FlickerMismatchGrid.modules_per_string, FlickerMismatchGrid.periodic))
    # plt.xlim((-30, 30))
    # plt.ylim((40, 100))
    # plt.show()


dx = 1
dy = 2
angle = 0


def test_grid():
    flicker = FlickerMismatchGrid(lat, lon, dx, dy, angle, angles_per_step=3)
    shadow, loss = flicker.create_heat_maps_irradiance(range(3186, 3200))
    axs = flicker.plot_on_site()

    assert(np.max(shadow) == approx(0.0019620, 1e-4))
    assert(np.average(shadow) == approx(7.3780e-05, 1e-4))
    assert(np.count_nonzero(shadow) == 606)
    assert(np.max(loss) == approx(2.48706, 1e-4))
    assert(np.average(loss) == approx(0.30068, 1e-4))
    assert(np.count_nonzero(loss) == 2166)

    plot_contour(shadow, flicker, axs)
    plt.title("Flicker Loss\n{}dx, {}dy, {}deg, {}mod/str, periodic {}".
              format(dx, dx * dy, angle, FlickerMismatchGrid.modules_per_string, FlickerMismatchGrid.periodic))
    # plt.xlim((-30, 30))
    # plt.ylim((40, 100))
    # plt.show()


def test_parallel_grid():
    flicker = FlickerMismatchGrid(lat, lon, dx, dy, angle)

    start = time.time()
    shadow_s, flicker_map_s = flicker.create_heat_maps_irradiance(range(500, 560))
    print("serial time:", time.time() - start)

    start = time.time()
    intervals = (range(500, 530), range(530, 560))
    shadow_p, flicker_map_p = flicker.run_parallel(2, intervals=intervals)
    print("2 proc time:", time.time() - start)

    start = time.time()
    intervals = (range(500, 510), range(510, 520), range(520, 530), range(530, 540), range(540, 550), range(550, 560))
    shadow_p, flicker_map_p = flicker.run_parallel(6, intervals=intervals)
    print("6 proc time:", time.time() - start)

    start = time.time()
    intervals = (range(500, 505), ) * 12
    shadow_p, flicker_map_p = flicker.run_parallel(12, intervals=intervals)
    print("12 proc time:", time.time() - start)

    diff_shadow = shadow_p - shadow_s
    diff_flicker = flicker_map_p - flicker_map_s
