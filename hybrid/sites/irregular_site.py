import numpy as np

from .locations import locations


# case 2
def make_irregular_site(
        year: int = 2012,
        tz: int = -6,
        lat: float = locations[1][0],
        lon: float = locations[1][1],
        elev: float = locations[1][2],
        ) -> dict:
    x_turb = [10363.7833, 9894.9437, 8450.2895, 9008.9311, 9567.5726, 10126.2142, 7862.2807, 8537.7355, 9213.1903,
              9888.6451, 7274.2718, 8066.5399, 8858.8079, 9651.076, 6686.263, 7371.5049, 8056.7467, 8741.9886,
              9427.2305, 6098.2541, 6750.8566, 7403.4592, 8056.0622, 8708.67, 9361.2778]
    y_turb = [6490.2719, 6316.918, 6455.3421, 6043.4997, 5631.6572, 5219.8148, 5665.8933, 5093.7148, 4521.5362, 3949.3577,
              4876.4446,
              4143.9299, 3411.4153, 2678.9006, 4086.9958, 3416.8405, 2746.6851, 2076.5297, 1406.3743, 3297.5471, 2665.4498,
              2033.3525,
              1401.2557, 769.1637, 137.0718]

    x_boundary = np.array(
        [10363.8, 9449.7, 9387.0, 9365.1, 9360.8, 9361.5, 9361.3, 7997.6, 6098.3, 8450.3, 8505.4, 9133.0, 9332.8, 9544.2,
         9739.0,
         9894.9, 10071.8, 10106.9, 10363.8])
    y_boundary = np.array(
        [6490.3, 1602.2, 1056.6, 625.5, 360.2, 126.9, 137.1, 1457.9, 3297.5, 6455.3, 6422.3, 6127.4, 6072.6, 6087.1, 6171.2,
         6316.9, 6552.5, 6611.1, 6490.3])

    shift_x = np.mean(x_boundary)
    shift_y = np.mean(y_boundary)
    x_turb = np.array(x_turb) - shift_x
    y_turb = np.array(y_turb) - shift_y
    verts = [(x_boundary[i] - shift_x, y_boundary[i] - shift_y) for i in range(len(x_boundary))]

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


irregular_site = make_irregular_site()
