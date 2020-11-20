import pytest
import numpy as np

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.turbine_layout_tools import create_grid


def test_create_grid():
    site_info = SiteInfo(flatirons_site)
    bounding_shape = site_info.polygon.buffer(-200)
    site_info.plot()
    turbine_positions, solar_positions = create_grid(bounding_shape,
                                                     site_info.polygon.centroid,
                                                     np.pi / 4,
                                                     200,
                                                     200,
                                                     .5)
    turb_x = [454.4841526950461, 595.9055089323556, 737.3268651696651, 878.7482214069747, 325.19483081370095,
              466.61618705101046, 608.0375432883199, 749.4588995256292, 337.3268651696651, 478.7482214069747,
              208.03754328831985, 349.45889952562925]
    turb_y = [385.8597920298026, 527.281148267112, 668.7025045044215, 810.123860741731, 456.57047014845733,
              597.9918263857669, 739.4131826230762, 880.8345388603857, 668.7025045044215, 810.123860741731,
              739.4131826230762, 880.8345388603857]
    for i in range(len(turbine_positions)):
        assert(turbine_positions[i].x == pytest.approx(turb_x[i], 1e-3))
        assert(turbine_positions[i].y == pytest.approx(turb_y[i], 1e-3))

    solar_rows_lengths = [768.0984021863013, 729.573732924646, 636.7181859910035, 354.926462175975, 73.13473836094663]
    for i in range(len(solar_positions[0])):
        assert(solar_positions[0][i].length == pytest.approx(solar_rows_lengths[i]))
    assert(solar_positions[1] == 200)

