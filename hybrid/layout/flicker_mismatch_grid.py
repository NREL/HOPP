from hybrid.layout.flicker_mismatch import *
from hybrid.layout.shadow_flicker import create_turbines_in_grid, get_turbine_grid_shadow

from hybrid.log import flicker_logger as logger
sys.path.append('.')

lat = 39.7555
lon = -105.2211
dx_multiples = range(3, 12, 2)
dy_multiples = range(3, 12, 2)
deg_multiples = range(0, 90, 18)
func_space = product(dx_multiples, dy_multiples, deg_multiples)


class FlickerMismatchGrid(FlickerMismatch):
    n_turbines_per_side = 4
    """
    Simulates wind turbines placed in a grid. The grid layout is determined by dx, dy and grid angle. The shadow cast
    by all turbines are calculated for the # of simulation steps: number of blade angles (evenly spaced) per step of the
    hour.

    The shadow heat map is produced as a loss ratio relative to unshaded areas (0 - 1). This loss ratio is with
    respect to plane-of-array irradiance, as calculated for a single-axis tracking system.

    The flicker heat map is another loss ratio (0 - 1), but with respect to power production of an unshaded
    string of panels as modeled by PVMismatch. This is calculated by modeling panels at each grid location, grouped into
    strings, and simulating the power of each string. If FlickerMismatch.periodic is true, then a string that continues
    past the bottom edge of the heat map emerges from the top edge and the loss value for the overlapping cells is the
    average of the modules located there.

    The losses are aggregated over the year.
    """
    def __init__(self,
                 lat: float,
                 lon: float,
                 turbine_nx: float,
                 turbine_ny: float,
                 angle: float = 0,
                 blade_length: int = 35,
                 angles_per_step: int = 1):
        """

        :param lat: latitude
        :param lon: longitude
        :param turbine_nx: number of turbine diameters for horizontal spacing in grid
        :param turbine_ny: number of turbine diameters for vertical spacing in grid
        :param angle: degree of rotation for turbine grid
        :param blade_length: meters
        :param angles_per_step: number of blade angles per step of the hour
        """
        FlickerMismatch.periodic = True
        self.center_grid = None
        self.turbine_dx = turbine_nx * blade_length * 2
        self.turbine_dy = turbine_ny * blade_length * 2
        self.grid_angle = int(angle) % 90
        self.n_rows_modules = int(turbine_nx / (0.124 * 12))
        super().__init__(lat, lon, blade_length=blade_length, angles_per_step=angles_per_step)

        self.filename_full = "{}_{}_{}_{}_{}_{}_{}".format(self.lat, self.lon,
                                                           self.steps_per_hour, self.angles_per_step,
                                                           self.turbine_dx, self.turbine_dy, self.grid_angle)
        self.grid_turbine_shadow_file = Path(__file__).parent / "data" / str(self.filename_full + "_shd.pkl")
        logger.info("Creating FlickerMismatchModel with filename_full {}".format(self.filename_full))

    def _setup_array(self
                     ) -> None:
        """
        Setup the solar panel array as a Point per panel
        """
        self.turb_pos = []
        theta = np.radians(self.grid_angle)

        # where the turbines are
        self.turb_pos, self.site = create_turbines_in_grid(self.turbine_dx,
                                                           self.turbine_dy,
                                                           theta,
                                                           FlickerMismatchGrid.n_turbines_per_side)

        # find the center grid which is symmetrical to all inner grids
        center_grid_coordinates = [(self.turb_pos[t][0], self.turb_pos[t][1]) for t in (5, 6, 10, 9)]
        self.center_grid = Polygon(center_grid_coordinates)
        self.site_points, self.heat_map_template = self._setup_heatmap_template(self.center_grid.bounds,
                                                                                self.gridcell_width,
                                                                                self.gridcell_height)

        # where solar strings are
        string_width = module_width
        string_height = self.center_grid.bounds[3] - self.center_grid.bounds[1]
        y_pos = self.center_grid.bounds[1]
        biggest_string_coordinates = ((0, y_pos),
                                      (string_width, y_pos),
                                      (string_width, y_pos + string_height),
                                      (0, y_pos + string_height))
        biggest_string = Polygon(biggest_string_coordinates)

        self.array = []
        x_pos = self.center_grid.bounds[0]
        while x_pos < self.center_grid.bounds[2]:
            string = translate(biggest_string, x_pos, 0)
            string = string.intersection(self.center_grid)
            if string.area > 0:
                self.array.append(string)
            x_pos += string_width
        logger.info("setup_turbines_and_arrays success")

        # Create points centered on each module
        self.array_string_points = []
        for array in self.array:
            array_points = self.site_points.intersection(array.buffer(module_height / 4))

            if array_points.is_empty:
                continue

            string_points = self._setup_string_points(array_points)

            self.array_string_points.append(string_points)
        logger.info("setup_point_maps success")

    def _calculate_turbine_shadow(self,
                                  ind: int):
        return get_turbine_grid_shadow(self.turbine_shadow[ind], self.turb_pos)

    def plot_on_site(self,
                     plot_points=True,
                     plot_array=True):
        fig, axs = plt.subplots()
        axs.set_aspect('equal')
        xs, ys = self.site.exterior.xy
        plt.plot(xs, ys)
        for t in self.turb_pos:
            plt.plot(t[0], t[1], 'bo')
        if plot_array:
            for p in self.array:
                x, y = p.exterior.xy
                plt.plot(x, y)
        if plot_points:
            for pt in self.site_points:
                plt.plot(pt.x, pt.y, 'k.')
            for array in self.array_string_points:
                if not array:
                    continue
                for string in array:
                    for pt in string:
                        plt.plot(pt.x, pt.y, 'w.')
        x, y = self.center_grid.exterior.xy
        plt.plot(x, y)
        return axs


def create_heat_map_irradiance(grid_dx_diams: float,
                               grid_dy_diams: float,
                               grid_degrees: float,
                               angles: int = 1,
                               steps: Union[range, None] = None,
                               procs: int = mp.cpu_count()
                               ) -> tuple:
    """
    Runs FlickerMismatchGrid to produce heat maps for shading and flicker for the lat/lon at the top of this script

    :param grid_dx_diams: spacing of turbine grid in diameters
    :param grid_dy_diams: spacing of turbine grid in diameters
    :param grid_degrees: grid rotation angle
    :param angles: number of blade angles per simulation step
    :param steps: list of ranges for each processor to simulate
    :param procs: number processors
    :return: tuple of nd.arrays for shadow and flicker heat maps
    """
    flicker_shading = FlickerMismatchGrid(lat, lon, grid_dx_diams, grid_dy_diams, grid_degrees, angles_per_step=angles)
    return flicker_shading.run_parallel(procs, ("poa", "power"), steps)
