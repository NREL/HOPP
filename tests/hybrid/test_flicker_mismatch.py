from pytest import approx
import time
from hybrid.flicker.data.plot_flicker import *

lat = 39.7555
lon = -105.2211


def test_single_turbine():
    FlickerMismatch.diam_mult_nwe = 3
    FlickerMismatch.diam_mult_s = 1
    flicker = FlickerMismatch(lat, lon, angles_per_step=1)
    shadow, loss = flicker.create_heat_maps(range(3186, 3200), ("poa", "power"))

    assert(np.max(shadow) == approx(0.00067, 1e-1))
    assert(np.average(shadow) * 1e6 == approx(3.1, 1e-1))
    assert(np.count_nonzero(shadow) > 900)
    assert(np.max(loss) > 0.2)
    assert(np.average(loss) > 9e-6)
    assert(np.count_nonzero(loss) == 60)

    axs = flicker.plot_on_site(False, False)
    plot_contour(loss, flicker, axs)
    # plt.xlim((-30, 30))
    # plt.ylim((40, 100))
    # plt.show()


def test_parallel_single():
    FlickerMismatch.diam_mult_nwe = 3
    FlickerMismatch.diam_mult_s = 1
    flicker = FlickerMismatch(lat, lon, angles_per_step=1)

    shadow, loss = flicker.run_parallel(2, (range(3186, 3188), range(3188, 3200)), ("poa", "power"))

    assert(np.max(shadow) == approx(0.00067, 1e-1))
    assert(np.average(shadow) * 1e6 == approx(3.1, 1e-1))
    assert(np.count_nonzero(shadow) > 900)
    assert(np.max(loss) > 0.2)
    assert(np.average(loss) > 9e-6)
    assert(np.count_nonzero(loss) == 60)


def test_single_turbine_time_weighted():
    FlickerMismatch.diam_mult_nwe = 3
    FlickerMismatch.diam_mult_s = 1
    flicker = FlickerMismatch(lat, lon, angles_per_step=None)
    (hours_shaded, ) = flicker.create_heat_maps(range(3186, 3208), ("time",))

    axs = flicker.plot_on_site(False, False)
    c = plot_contour(hours_shaded, flicker, axs, vmin=np.amin(hours_shaded), vmax=np.amax(hours_shaded))
    # plt.colorbar(c)
    # plt.show()
    assert(np.max(hours_shaded) == approx(0.000913242))
    assert(np.average(hours_shaded) == approx(1.206235e-05))
    assert(np.count_nonzero(hours_shaded) > 6560)


dx = 1
dy = 2
angle = 0


def test_grid():
    flicker = FlickerMismatchGrid(lat, lon, dx, dy, angle, angles_per_step=3)
    shadow, loss = flicker.create_heat_maps(range(3186, 3200), ("poa", "power"))
    axs = flicker.plot_on_site()

    assert(np.max(shadow) == approx(0.0021, 1e-1))
    assert(np.average(shadow) == approx(7.5e-05, 1e-1))
    assert(np.count_nonzero(shadow) == approx(750, 1e-1))
    assert(np.max(loss) == approx(3.4, 1e-1))
    assert(np.count_nonzero(loss) == approx(2170, 1e-2))

    plot_contour(shadow, flicker, axs)
    # plt.show()


def test_parallel_grid():
    flicker = FlickerMismatchGrid(lat, lon, dx, dy, angle)

    start = time.time()
    shadow_s, flicker_map_s = flicker.create_heat_maps(range(500, 560), ("poa", "power"))
    print("serial time:", time.time() - start)

    start = time.time()
    intervals = (range(500, 530), range(530, 560))
    shadow_p, flicker_map_p = flicker.run_parallel(2, intervals, ("poa", "power"))
    print("2 proc time:", time.time() - start)

    start = time.time()
    intervals = (range(500, 510), range(510, 520), range(520, 530), range(530, 540), range(540, 550), range(550, 560))
    shadow_p, flicker_map_p = flicker.run_parallel(6, intervals, ("poa", "power"))
    print("6 proc time:", time.time() - start)

    start = time.time()
    intervals = (range(500, 505), ) * 12
    shadow_p, flicker_map_p = flicker.run_parallel(12, intervals, ("poa", "power"))
    print("12 proc time:", time.time() - start)

    diff_shadow = shadow_p - shadow_s
    diff_flicker = flicker_map_p - flicker_map_s


def test_plot():
    data_path = Path(__file__).parent.parent.parent / "hybrid" / "flicker" / "data"
    flicker_path = data_path / "{}_{}_{}_{}_shadow.txt".format(lat,
                                                               lon,
                                                               4, 12)
    try:
        flicker_heatmap = np.loadtxt(flicker_path)
    except OSError:
        raise NotImplementedError("Flicker look up table for project's lat and lon does not exist.")

    flicker = FlickerMismatch(lat, lon, angles_per_step=12)
    axs = flicker.plot_on_site(False, False)
    plot_contour(flicker_heatmap, flicker, axs)
    plot_tiled(flicker_heatmap, flicker, axs)
