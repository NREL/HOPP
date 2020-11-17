import numpy as np

from .locations import locations


def make_circular_site(
        radius: float = 3000,
        num_segments: int = 64,
        lat: float = locations[1][0],
        lon: float = locations[1][1],
        elev: float = locations[1][2],
        year: int = 2012,
        tz: int = -6,
        ) -> dict:
    verts = [(radius * np.sin(theta), radius * np.cos(theta))
             for theta in np.arange(0, 2 * np.pi, 2 * np.pi / num_segments)]

    return {
        "lat":             lat,
        "lon":             lon,
        "elev":            elev,
        "year":            year,
        "tz":              tz,
        'site_boundaries': {
            'verts':        verts,
            'verts_simple': verts,
            }
        }


circular_site = make_circular_site()
