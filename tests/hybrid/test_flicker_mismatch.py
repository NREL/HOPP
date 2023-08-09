import platform
from pytest import approx
from hopp.layout.flicker_data.plot_flicker import *
from hopp.keys import set_nrel_key_dot_env


set_nrel_key_dot_env()

lat = 39.7555
lon = -105.2211


def plot_maps(maps, flicker):
    for m in maps:
        axs = flicker.plot_on_site(False, False)
        c = plot_contour(m, flicker, axs, vmin=np.amin(m), vmax=np.amax(m))
        plt.colorbar(c)
        plt.show()


def test_single_turbine():
    FlickerMismatch.diam_mult_nwe = 3
    FlickerMismatch.diam_mult_s = 1
    flicker = FlickerMismatch(lat, lon, angles_per_step=1)
    shadow, loss = flicker.create_heat_maps(range(3185, 3187), ("poa", "power"))

    assert(np.max(shadow) == approx(1.0, 1e-4))
    assert(np.average(shadow) == approx(0.0041092, 1e-4))
    assert(np.count_nonzero(shadow) == approx(636, 1e-4))
    assert(np.max(loss) == approx(0.314133, 1e-4))
    assert(np.average(loss) == approx(0.0042872, 1e-4))
    assert(np.count_nonzero(loss) == approx(2940, 1e-4))


def test_single_turbine_multiple_angles():
    FlickerMismatch.diam_mult_nwe = 3
    FlickerMismatch.diam_mult_s = 1
    flicker = FlickerMismatch(lat, lon, angles_per_step=3)
    shadow, loss = flicker.create_heat_maps(range(3185, 3187), ("poa", "power"))

    assert(np.max(shadow) == approx(1.0, 1e-4))
    assert(np.average(shadow) == approx(0.0042229, 1e-4))
    assert(np.count_nonzero(shadow) == 698)
    assert(np.max(loss) == approx(0.314133, 1e-4))
    assert(np.average(loss) == approx(0.0043571, 1e-4))
    assert(np.count_nonzero(loss) == 3010)
    # plot_maps((shadow, loss), flicker)

    # run parallel
    if platform.system() != "Darwin":
        return
    shadow_p, loss_p = flicker.run_parallel(2, ("poa", "power",), (range(3185, 3186), range(3186, 3187)))

    assert(np.max(shadow_p) == approx(1.0, 1e-4))
    assert(np.average(shadow_p) == approx(0.0042229, 1e-4))
    assert(np.count_nonzero(shadow_p) == 698)
    assert(np.max(loss_p) == approx(0.314133, 1e-4))
    assert(np.average(loss_p) == approx(0.0043571, 1e-4))
    assert(np.count_nonzero(loss_p) == 3010)


def test_single_turbine_time_weighted():
    # two time steps: one with shading, one without
    FlickerMismatch.diam_mult_nwe = 3
    FlickerMismatch.diam_mult_s = 1
    flicker = FlickerMismatch(lat, lon, angles_per_step=None)
    (hours_shaded, ) = flicker.create_heat_maps(range(3187, 3189), ("time",))

    assert(np.max(hours_shaded) == approx(0.5))
    assert(np.average(hours_shaded) == approx(0.0016010, 1e-4))
    assert(np.count_nonzero(hours_shaded) == 435)

    if platform.system() != "Darwin":
        return
    intervals = (range(3187, 3188), range(3188, 3189))
    (hours_shaded_p, ) = flicker.run_parallel(2, ("time",), intervals)
    # plot_maps((hours_shaded, hours_shaded_p), flicker)

    assert(np.max(hours_shaded_p) == approx(0.5))
    assert(np.average(hours_shaded_p) == approx(0.0016010, 1e-4))
    assert(np.count_nonzero(hours_shaded_p) == 435)


def test_single_turbine_time_weighted_no_tower():
    FlickerMismatch.turbine_tower_shadow = False
    FlickerMismatch.diam_mult_nwe = 3
    FlickerMismatch.diam_mult_s = 1
    flicker = FlickerMismatch(lat, lon, angles_per_step=None)
    (hours_shaded, ) = flicker.create_heat_maps(range(3183, 3185), ("time",))
    # plot_maps((hours_shaded, ), flicker)

    assert(np.max(hours_shaded) == approx(0.5))
    assert(np.average(hours_shaded) == approx(0.0057390, 1e-4))
    assert(np.count_nonzero(hours_shaded) == 1066)


def test_single_turbine_time_weighted_no_tower_subhourly():
    FlickerMismatch.turbine_tower_shadow = False
    FlickerMismatch.diam_mult_nwe = 3
    FlickerMismatch.diam_mult_s = 1
    FlickerMismatch.steps_per_hour = 4
    flicker = FlickerMismatch(lat, lon, angles_per_step=None)

    # need to multiply index by number of steps in an hour
    (hours_shaded,) = flicker.create_heat_maps(range(3183 * FlickerMismatch.steps_per_hour,
                                                     3185 * FlickerMismatch.steps_per_hour), ("time",))

    # average is similar to hourly run
    assert(np.max(hours_shaded) == approx(0.375))
    assert(np.average(hours_shaded) == approx(0.0043852, 1e-4))
    assert(np.count_nonzero(hours_shaded) == 2353)

    FlickerMismatch.steps_per_hour = 16
    flicker = FlickerMismatch(lat, lon, angles_per_step=None)

    (hours_shaded,) = flicker.create_heat_maps(range(3183 * FlickerMismatch.steps_per_hour,
                                                     3185 * FlickerMismatch.steps_per_hour), ("time",))

    # plot_maps((hours_shaded, ), flicker)

    # average is very similar to 4-steps run
    assert(np.max(hours_shaded) == approx(0.362146, 1e-4))
    assert(np.average(hours_shaded) == approx(0.0041415, 1e-4))
    assert(np.count_nonzero(hours_shaded) == 3508)


def test_single_turbine_wind_dir():
    FlickerMismatch.turbine_tower_shadow = False
    FlickerMismatch.diam_mult_nwe = 3
    FlickerMismatch.diam_mult_s = 1
    FlickerMismatch.steps_per_hour = 1
    wind_dir = [90 if i % 2 else 0 for i in range(8760)]

    flicker = FlickerMismatch(lat, lon, angles_per_step=None, wind_dir=wind_dir)

    (hours_shaded,) = flicker.create_heat_maps(range(3183, 3185), ("time",))
    # plot_maps((hours_shaded, ), flicker)

    assert(np.max(hours_shaded) == approx(1.0))
    assert(np.average(hours_shaded) == approx(0.017173, 1e-4))
    assert(np.count_nonzero(hours_shaded) == 2819)


def test_grid():
    dx = 1
    dy = 2
    angle = 0
    FlickerMismatch.turbine_tower_shadow = True
    flicker = FlickerMismatchGrid(lat, lon, dx, dy, angle, angles_per_step=1)
    shadow, loss = flicker.create_heat_maps(range(3185, 3187), ("poa", "power"))

    assert(np.max(shadow) == approx(1.0, 1e-4))
    assert(np.average(shadow) == approx(0.031547, 1e-4))
    assert(np.count_nonzero(shadow) == approx(390, 1e-4))
    assert(np.max(loss) == approx(0.418338, 1e-4))
    assert(np.average(loss) == approx(0.033167, 1e-4))
    assert(np.count_nonzero(loss) == approx(1364, 1e-4))

    # run parallel with  multiple angles
    if platform.system() != "Darwin":
        return
    flicker = FlickerMismatchGrid(lat, lon, dx, dy, angle, angles_per_step=3)
    intervals = (range(3185, 3186), range(3186, 3187))
    shadow_p, loss_p = flicker.run_parallel(2, ("poa", "power"), intervals)

    assert(np.max(shadow_p) == approx(1.0, 1e-4))
    assert(np.average(shadow_p) == approx(0.031462, 1e-4))
    assert(np.count_nonzero(shadow_p) == approx(390, 1e-4))
    assert(np.max(loss_p) == approx(0.41833, 1e-4))
    assert(np.average(loss_p) == approx(0.0331158, 1e-4))
    assert(np.count_nonzero(loss_p) == approx(1364, 1e-4))


def test_plot():
    data_path = Path(__file__).parent.parent.parent / "hybrid" / "layout" / "flicker_data"
    flicker_path = data_path / "{}_{}_{}_{}_shadow.txt".format(lat,
                                                               lon,
                                                               4, 12)
    try:
        flicker_heatmap = np.loadtxt(flicker_path)
    except OSError:
        raise NotImplementedError("Flicker look up table for project's lat and lon does not exist.")

    flicker = FlickerMismatch(lat, lon, angles_per_step=12)
    axs = flicker.plot_on_site(False, False)
    plot_tiled(flicker_heatmap, flicker, axs)