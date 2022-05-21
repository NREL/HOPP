import sys
from typing import Union
sys.path.append('.')
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from hybrid.layout.flicker_mismatch import FlickerMismatch
from hybrid.layout.flicker_mismatch_grid import FlickerMismatchGrid, func_space, lat, lon


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 1)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def plot_contour(map, flicker: Union[FlickerMismatch, FlickerMismatchGrid], axs, levels=500, **kwargs):
    if hasattr(flicker, 'center_grid'):
        sx, sy = flicker.center_grid.exterior.xy
        axs.plot(sx, sy)
    coords = flicker.heat_map_template
    axs.set_aspect("equal")
    print(coords[1][0])
    c = axs.contourf(coords[1], coords[2], map, levels, **kwargs)
    return c


def plot_tiled(map, flicker: Union[FlickerMismatch, FlickerMismatchGrid], axs, **kwargs):
    coords = flicker.heat_map_template
    map_tile = np.tile(map, (3, 3))
    h = axs.imshow(map_tile, **kwargs)
    axs.set_aspect("equal")
    axs.invert_yaxis()
    return h

