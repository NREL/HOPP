import sys
sys.path.append('.')
from pathlib import Path
import glob
import re
import logging
import argparse
import copy
from ast import literal_eval
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hybrid.flicker.flicker_mismatch import FlickerMismatch
from hybrid.flicker.flicker_mismatch_grid import FlickerMismatchGrid, xs, ys, func_space, lat, lon


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 1)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def plot_heat_map(map, flicker, axs, vmin, cmap='viridis'):
    # z = np.pad(map, 10, pad_with)
    h = axs.imshow(z, vmin=vmin, vmax=1, cmap=cmap, interpolation="bilinear")
    # plt.colorbar(orientation='horizontal')
    axs.invert_yaxis()
    return h


def plot_contour(map, flicker, axs, levels=500, cmap="viridis"):
    if hasattr(flicker, 'center_grid'):
        sx, sy = flicker.center_grid.exterior.xy
        axs.plot(sx, sy)
    coords = flicker.heat_map_template
    c = axs.contourf(coords[1], coords[2], map, levels)
    return c


def plot_tiled(map, flicker, axs, levels=500):
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


def plot_gridded():
    dx = 1
    dy = 2
    ang = 0
    diam = 70
    steps_per_hour = 1
    shadow_file = "/Users/dguittet/Projects/HybridSystems/hybrid_systems/tests/1_2_0_3_3_shadow.txt"
    flicker_file = "/Users/dguittet/Projects/HybridSystems/hybrid_systems/tests/1_2_0_3_3_flicker.txt"

    heat_map_shadow = np.loadtxt(shadow_file)
    heat_map_flicker = np.loadtxt(flicker_file)

    # heat_map_shadow = 1 - heat_map_shadow
    # heat_map_flicker /= (8760 * steps_per_hour)
    # heat_map_flicker = 1 - heat_map_flicker

    model = FlickerMismatchGrid(lat, lon, dx, dy, ang)
    ratio = heat_map_flicker / heat_map_shadow
    vmin = np.amin(heat_map_flicker)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1, aspect='equal')
    ax1.set_title("Shadow")
    ax2 = fig.add_subplot(1, 3, 2, aspect='equal')
    ax2.set_title("Flicker")
    ax3 = fig.add_subplot(1, 3, 3, aspect='equal')
    ax3.set_title("Ratio")

    # plot_contour(heat_map_shadow, model, ax1)
    heat_map_shadow = 1 - heat_map_shadow
    plot_tiled(heat_map_shadow, model, ax1)
    plt.show()

    c = plot_heat_map(heat_map_shadow, model, ax1, vmin)
    fig.colorbar(c)
    plot_heat_map(heat_map_flicker, model, ax2, vmin)
    c = plot_heat_map(ratio, model, ax3, np.amin(ratio), cmap="gray")
    fig.colorbar(c)

    plt.suptitle("Shading vs Flicker Loss for {} diams dx, {} diams dy, {} deg".format(dx, dy, ang))
    plt.show()


def plot_single():
    angles = 1
    shadow_file = Path(__file__).parent / "heatmaps" / "{}_{}_1_{}_shadow.txt".format(lat, lon, angles)
    flicker_file = Path(__file__).parent / "heatmaps" / "{}_{}_1_{}_flicker.txt".format(lat, lon, angles)

    heat_map_shadow = np.loadtxt(shadow_file)
    heat_map_flicker = np.loadtxt(flicker_file)

    heat_map_shadow = 1 - heat_map_shadow
    heat_map_flicker /= 8760
    heat_map_flicker = 1 - heat_map_flicker

    model = FlickerMismatch(lat, lon, angles_per_step=angles)
    ratio = heat_map_flicker / heat_map_shadow
    vmin = np.amin(heat_map_flicker)

    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1, aspect='equal')
    ax1.set_title("Shadow")
    ax2 = fig.add_subplot(3, 1, 2, aspect='equal')
    ax2.set_title("Flicker")
    ax3 = fig.add_subplot(3, 1, 3, aspect='equal')
    ax3.set_title("Ratio")

    c = plot_heat_map(heat_map_shadow, model, ax1, vmin)
    fig.colorbar(c)
    plot_heat_map(heat_map_flicker, model, ax2, vmin)
    c = plot_heat_map(ratio, model, ax3, np.amin(ratio), cmap="gray")
    fig.colorbar(c)

    # plt.suptitle("Shading vs Flicker Loss for {} diams dx, {} diams dy, {} deg".format(dx, dy, ang))
    plt.show()


# plot_gridded()