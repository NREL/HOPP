import pytest
import numpy as np

from hybrid.sites import SiteInfo, flatirons_site
from hybrid.turbine_layout_tools import create_grid


def test_create_grid():
    site_info = SiteInfo(flatirons_site)
    bounding_shape = site_info.polygon.buffer(-200)
    site_info.plot()
    turbine_positions = create_grid(bounding_shape,
                                    site_info.polygon.centroid,
                                    np.pi / 4,
                                    200,
                                    200,
                                    .5)
    expected_positions = [
        [242., 497.],
        [383., 355.],
        [312., 709.],
        [454., 568.],
        [595., 426.],
        [525., 780.],
        [666., 638.],
        [737., 850.],
        [878., 709.]
    ]
    for n, t in enumerate(turbine_positions):
        assert(t.x == pytest.approx(expected_positions[n][0], 1e-1))
        assert(t.y == pytest.approx(expected_positions[n][1], 1e-1))

