import matplotlib.pyplot as plt 
import numpy as np 

import matplotlib.colors

from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

# Create our own custom color maps to transition from black to the color of interest
cmap_C0 = LinearSegmentedColormap.from_list("blue", ["k", "C0"])
cmap_C1 = LinearSegmentedColormap.from_list("orange", ["k", "C1"])

def plot_cline(ax, x, y, dates, lw=1.0, primary=True):
    '''
    Plot the orbit shaded by phase.

    Args:
        ax: matplotlib axes to plot onto 
        x: the matplotlib x values (i.e., the Y astro coordinates)
        y: the matplotlib y values (i.e., the X astro coordinates)
        dates: the dates corresponding to each (x,y) value 
        lw: the linewidth of the orbit
        primary: if True, plot in blue, else plot orange.

    Returns:
        None

    Example

            plot_cline(ax_orbit, Y_A, X_A, dates, primary=True)
            plot_cline(ax_orbit, Y_B, X_B, dates, primary=False)

    '''

    cmap = cmap_C0 if primary else cmap_C1
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap,
                        norm=plt.Normalize(np.min(dates), np.max(dates)))
    lc.set_array(dates)
    lc.set_linewidth(lw)
    ax.add_collection(lc)

def plot_nodes(ax, Omega, radius):
    '''
    Plot the line of nodes on the sky plot. The ascending node half is red, while the 
    descending node half is blue. 

    Args:
        ax: the matplotlib axes to plot onto 
        Omega: the position angle of the ascending node (in radians)
        radius: how far away from the center to draw the line

    Returns:
        None
    '''

    # calculate the slope of the line from Omega

    x_a = radius * np.sin(Omega)
    y_a = radius * np.cos(Omega)

    x_d = radius * np.sin(Omega + np.pi)
    y_d = radius * np.cos(Omega + np.pi)
    
    # plot in two segments
    ax.plot([0, x_a], [0, y_a], color="r")
    ax.plot([0, x_d], [0, y_d], color="b")
