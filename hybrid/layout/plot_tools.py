import matplotlib.pyplot as plt
from shapely.geometry import (
    LineString,
    MultiPolygon,
    )


def plot_turbines(turb_pos_x: list,
                  turb_pos_y: list,
                  color='g',
                  alpha=.5
                  ) -> None:
    for n in range(len(turb_pos_y)):
        plt.plot(turb_pos_x[n], turb_pos_y[n], 'o', color=color, alpha=alpha)


def plot_solar_strands(
        figure,
        axes,
        areas: (int, float, LineString),
        *args,
        **kwargs
        ) -> None:
    if type(areas[0]) is int:
        areas = (areas,)
    for a in areas:
        for s in a.strands:
            segment: LineString = s[2]
            x, y = segment.xy
            axes.plot(x, y, *args, **kwargs)


def plot_shape(
        figure,
        axes,
        shape,
        *args,
        **kwargs):
    if isinstance(shape, MultiPolygon):
        for poly in shape:
            x, y = poly.exterior.xy
            axes.plot(x, y, *args, **kwargs)
    elif isinstance(shape, LineString):
        points = list(shape.coords)
        axes.plot([point[0] for point in points], [point[1] for point in points], *args, **kwargs)
    else:
        try:
            x, y = shape.exterior.xy
            axes.plot(x, y, *args, **kwargs)
        except:
            pass