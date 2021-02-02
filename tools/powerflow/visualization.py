import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def grid_layout(grid_model, figure_args=None, scatter_args=None):
    """ 
    Show 2D or 3D mapping of plant layout.
    """
    default_args = {'marker':'o',
                    'c':'black',
                    'depthshade':False}
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    X = grid_model.nodes[:,0]
    Y = grid_model.nodes[:,1]
    Z = grid_model.nodes[:,2]
    
    # Plot nodes
    if scatter_args is not None:
        ax.scatter(X, Y, Z, **scatter_args)
    else:
        ax.scatter(X[1:], Y[1:], Z[1:], **default_args)
        del default_args['marker']
        ax.scatter(X[0], Y[0], Z[0], marker='s', **default_args)
    
    # Add lines (does not plot line method correctly (square, direct, manual))
    A_b = grid_model.admittance_matrix != 0
    for i in range(0, grid_model.N+1):
        for j in range(0, i): # only search lower triangle
            if A_b[i,j]: # Edge to plot
                ax.plot([X[i], X[j]], [Y[i], Y[j]], [Z[i], Z[j]], 
                        c='black')
    
    # Put grid point labels on (not very robust, but works for now)
    for n in range(len(X)):
        ax.text(X[n]+5000, Y[n]+10000, Z[n], grid_model.node_labels[n])

    # Adjust plot spacing correctly
    largest_range = np.array([X.max()-X.min(), 
                              Y.max()-Y.min(), 
                              Z.max()-Z.min()]).max()
    ax.set_xlim(X.min(), X.min()+largest_range)
    ax.set_ylim(Y.min(), Y.min()+largest_range)
    #ax.set_zlim(Z.min, Z.min+largest_range)
    ax.set_xlabel('X-position [m]')
    ax.set_ylabel('Y-position [m]')


    return ax, plt

def overlay_quantity(grid_model, z, ax, label=None, title=None):
    """
    Plot bar graph of a quantity (e.g. power, voltage) on top of layout.
    """

    X = grid_model.nodes[:,0]
    Y = grid_model.nodes[:,1]
    z_b = grid_model.nodes[:,2].max()
    
    # Create color scale
    max_color = [1, 0, 0]
    zero_color = [1, 1, 1]
    c = [0, 0, 0]
    for i in range(len(z)):
        a = z[i]/z.max()
        c[0] = a*max_color[0] + (1-a)*zero_color[0]
        c[1] = a*max_color[1] + (1-a)*zero_color[1]
        c[2] = a*max_color[2] + (1-a)*zero_color[2]

        ax.bar3d(X[i], Y[i], z_b, 1900, 1900, z[i], color=c)

    
    ax.set_zlabel(label)
    ax.set_title(title)

    return ax

def add_wind_direction(ax, bearing, coordinates=[0,0,0], length=100):
    """
    Place arrow showing the wind direction on the plot.

    TODO: Currently not working as desired
    """

    arrow_width = 0.5
    kwargs = {'width':arrow_width}

    dx = length*np.sin(bearing*np.pi/180)
    dy = -length*np.cos(bearing*np.pi/180)

    x = np.linspace(coordinates[0], dx, 11)
    y = np.linspace(coordinates[1], dy, 11)
    z = np.linspace(coordinates[2], coordinates[2], 11)

    u = np.zeros((11,11))
    v = np.zeros((11,11))
    w = np.zeros((11,11))
    u[5,5] = dx
    v[5,5] = dy
    w[5,5] = 0

    #ax.arrow(coordinates[0],coordinates[1],coordinates[2], dx,dy,0, **kwargs)
    ax.quiver(x, y, z, u, v, w, scale=1, units='xy')
    return ax
