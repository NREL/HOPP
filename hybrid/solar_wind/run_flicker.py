import sys
sys.path.append('.')
from hybrid.solar_wind.flicker_mismatch import *
import argparse
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

"""
Script to run a FlickerMismatch simulation using MPI for a single lat/lon

flicker_mismatch.py contains a range of lat and lon: "func_space"

Set to run 4 steps per hour, with 12 blade angles per step

:argument skip: 
            if -1 use fixed lat/lon;
            if > 0, index within "func_space" to simulation
"""

if rank == 0:
    parser = argparse.ArgumentParser()
    parser.add_argument('skip', type=int, nargs='?', default=-1)
    args = parser.parse_args()
    skip = args.skip

    n_procs = mp.cpu_count()
    func_count = len(lat_range) * len(lon_range)
    # logger.info("Set up %d runs over %d processes", func_count, n_procs)

    if skip > 0:
        for i in range(skip):
            func_space.__next__()
        lat, lon = func_space.__next__()
    else:
        lat = 36.334
        lon = -119.769
    logger.info("Run #: lat {}, lon {}".format(lat, lon))
else:
    lat, lon = None, None

lat = comm.bcast(lat, root=0)
lon = comm.bcast(lon, root=0)
angles = 12
FlickerMismatch.steps_per_hour = 4
heat_map_shadow, heat_map_flicker = create_heat_map_irradiance(lat, lon, angles=angles,
                                                               procs=n_procs, total_parts=size, n_part=rank)

comm.Barrier()
if comm.rank == 0:
    total_heat_map_shadow = np.zeros(heat_map_shadow.shape)
    total_heat_map_flicker = np.zeros(heat_map_shadow.shape)
else:
    total_heat_map_shadow = None
    total_heat_map_flicker = None

comm.Reduce(
    [heat_map_shadow, MPI.DOUBLE],
    [total_heat_map_shadow, MPI.DOUBLE],
    op=MPI.SUM,
    root=0
)

comm.Reduce(
    [heat_map_flicker, MPI.DOUBLE],
    [total_heat_map_flicker, MPI.DOUBLE],
    op=MPI.SUM,
    root=0
)

if comm.rank == 0:
    with np.printoptions(threshold=np.inf):
        logger.debug(total_heat_map_shadow)
        logger.debug(total_heat_map_flicker)
        shadow_path = "{}_{}_{}_{}_shadow.txt".format(lat, lon, FlickerMismatch.steps_per_hour, angles)
        flicker_path = "{}_{}_{}_{}_flicker.txt".format(lat, lon, FlickerMismatch.steps_per_hour, angles)
        logger.debug(shadow_path + "\n" + flicker_path)
        np.savetxt(shadow_path, total_heat_map_shadow)
        np.savetxt(flicker_path, total_heat_map_flicker)

        shadow_path = "hybrid/solar_wind/data/heatmaps/{}_{}_1_{}_shadow.txt".format(lat, lon, angles)
        flicker_path = "hybrid/solar_wind/data/heatmaps/{}_{}_1_{}_flicker.txt".format(lat, lon, angles)
        logger.debug(shadow_path + "\n" + flicker_path)
        # np.savetxt(shadow_path, total_heat_map_shadow)
        # np.savetxt(flicker_path, total_heat_map_flicker)


