import sys
from typing import Union
sys.path.append('.')
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from hybrid.flicker.flicker_mismatch import FlickerMismatch
from hybrid.flicker.flicker_mismatch_grid import FlickerMismatchGrid, xs, ys, func_space, lat, lon


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 1)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def plot_contour(map, flicker: Union[FlickerMismatch, FlickerMismatchGrid], axs, levels=500, cmap="viridis"):
    if hasattr(flicker, 'center_grid'):
        sx, sy = flicker.center_grid.exterior.xy
        axs.plot(sx, sy)
    coords = flicker.heat_map_template
    c = axs.contourf(coords[1], coords[2], map, levels)
    return c


def plot_tiled(map, flicker: Union[FlickerMismatch, FlickerMismatchGrid], axs, levels=500):
    # if hasattr(flicker, 'center_grid'):
    #     sx, sy = flicker.center_grid.exterior.xy
    #     axs.plot(sx, sy)
    coords = flicker.heat_map_template
    map_tile = np.tile(map, (3, 3))
    # x = np.tile(coords[1], 3)
    # y = np.tile(coords[2], 3)
    h = axs.imshow(map_tile)
    axs.invert_yaxis()
    return h

