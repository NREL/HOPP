import sys
sys.path.append('.')
from pathlib import Path
import argparse
import random
import multiprocessing as mp
import numpy as np

from hybrid.log import flicker_logger as logger
from hybrid.solar_wind.flicker_mismatch_grid import create_heat_map_irradiance, n_procs, \
    dx_multiples, dy_multiples, deg_multiples, func_space

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

"""
Script to run a FlickerMismatchGrid simulation using MPI for a single layout

A layout is defined by a dx (number of diameters between turbines horizontally), dy (vertically), and angle degree.

flicker_mismatch_grid.py contains a range of layouts: "func_space"

Set to run 1 steps per hour, with 1 blade angles per step

:argument skip: 
            index within "func_space" to simulate, if not random
:argument random:
            random dx, dy amd deg
"""

if rank == 0:
    parser = argparse.ArgumentParser()
    parser.add_argument('skip', type=int, nargs='?', default=0)
    parser.add_argument('random', type=int, nargs='?', default=0)
    args = parser.parse_args()
    skip = args.skip
    use_random = args.random

    n_procs = mp.cpu_count()
    func_count = len(dx_multiples) * len(dy_multiples) * len(deg_multiples)
    # logger.info("Set up %d runs over %d processes", func_count, n_procs)

    if not use_random:
        for i in range(skip):
            func_space.__next__()

        dx, dy, degrees = func_space.__next__()
        logger.info("Run #: dx {}, dy {}, angle {}".format(dx, dy, degrees))

        # for k, it in enumerate(func_space):
        #     logger.info("Run #{}: {}".format(k, it))
        #     create_heat_map_irradiance(*it)
    else:
        random.seed()
        dx = round(random.uniform(3, 11), 2)
        dy = round(random.uniform(3, 11), 2)
        degrees = round(random.uniform(0, 90), 2)
        logger.info("Run random: {}, {}, {}".format(dx, dy, degrees))
else:
    dx, dy, degrees = None, None, None

dx = comm.bcast(dx, root=0)
dy = comm.bcast(dy, root=0)
degrees = comm.bcast(degrees, root=0)

heat_map_shadow, heat_map_flicker = create_heat_map_irradiance(dx, dy, degrees, total_parts=size, n_part=rank)

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
    dx *= 70
    dy *= 70
    with np.printoptions(threshold=np.inf):
        shadow_path = Path(__file__).parent / "data" / "heatmaps" / "{}_{}_{}_shadow.txt".format(dx, dy, degrees)
        flicker_path = Path(__file__).parent / "data" / "heatmaps" / "{}_{}_{}_flicker.txt".format(dx, dy, degrees)
        np.savetxt(shadow_path, total_heat_map_shadow)
        np.savetxt(flicker_path, total_heat_map_flicker)
        logger.debug(shadow_path + "\n" + flicker_path)
