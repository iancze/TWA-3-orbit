import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import collections

import matplotlib.colors

from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

# Create our own custom color maps to transition from black to the color of interest
cmap_C0 = LinearSegmentedColormap.from_list("blue", ["k", "C0"])
cmap_C1 = LinearSegmentedColormap.from_list("orange", ["k", "C1"])


def plot_cline(ax, x, y, dates, lw=1.0, primary=True):
    """
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

    """

    cmap = cmap_C0 if primary else cmap_C1
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(
        segments, cmap=cmap, norm=plt.Normalize(np.min(dates), np.max(dates))
    )
    lc.set_array(dates)
    lc.set_linewidth(lw)
    ax.add_collection(lc)


def plot_nodes(ax, Omega, radius):
    """
    Plot the line of nodes on the sky plot. The ascending node half is red, while the 
    descending node half is blue. 

    Args:
        ax: the matplotlib axes to plot onto 
        Omega: the position angle of the ascending node (in radians)
        radius: how far away from the center to draw the line

    Returns:
        None
    """

    # calculate the slope of the line from Omega

    x_a = radius * np.sin(Omega)
    y_a = radius * np.cos(Omega)

    x_d = radius * np.sin(Omega + np.pi)
    y_d = radius * np.cos(Omega + np.pi)

    # plot in two segments
    ax.plot([0, x_a], [0, y_a], color="r")
    ax.plot([0, x_d], [0, y_d], color="b")


def plot_errorbar(ax, thetas, rhos, theta_errs, rho_errs, **kwargs):
    """
    Custom routine for plotting polar error bars onto cartesian plane
    All kwargs sent to plot. Thetas are in radians.
    """

    thetas = np.atleast_1d(thetas)
    rhos = np.atleast_1d(rhos)
    theta_errs = np.atleast_1d(theta_errs)
    rho_errs = np.atleast_1d(rho_errs)

    assert len(thetas) == len(rhos) == len(theta_errs) == len(rho_errs)

    for theta, rho, theta_err, rho_err in zip(thetas, rhos, theta_errs, rho_errs):

        # In this notation, x has been flipped.
        # x = rho * np.sin(theta)
        # y = rho * np.cos(theta)

        # Calculate two lines, one for theta_err, one for rho_err

        # theta error
        # theta error needs a small correction in the rho direction in order to go right through the data point
        delta = rho * (1 / np.cos(theta_err) - 1)
        xs = (rho + delta) * np.sin(np.array([theta - theta_err, theta + theta_err]))
        ys = (rho + delta) * np.cos(np.array([theta - theta_err, theta + theta_err]))
        ax.plot(xs, ys, **kwargs)

        # rho error
        xs = np.sin(theta) * np.array([(rho - rho_err), (rho + rho_err)])
        ys = np.cos(theta) * np.array([(rho - rho_err), (rho + rho_err)])
        ax.plot(xs, ys, **kwargs)


def efficient_trace(trace, var_names, figstem, max_panel=6):
    """
    Wrap the arviz call for trace to speed things up by splitting into smaller figures bundled as a scrollable PDF.
    """

    N = len(var_names)
    n = 0
    i = 0
    while n <= N:
        plot_vars = var_names[n:n+max_panel]

        az.plot_trace(trace, var_names=plot_vars)
        plt.savefig(figstem.format(i))
        plt.close("all")
        n += max_panel 
        i += 1


def efficient_autocorr(trace, var_names, figstem, max_panel=6):
    """
    Wrap the arviz call for correlation to speed things up by splitting into smaller figures bundled as a scrollable PDF.
    """

    N = len(var_names)
    n = 0
    i = 0
    while n <= N:
        plot_vars = var_names[n:n+max_panel]

        az.plot_autocorr(trace, var_names=plot_vars)
        plt.savefig(figstem.format(i))
        plt.close("all")
        n += max_panel 
        i += 1
