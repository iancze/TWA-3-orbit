import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap

from astropy.table import Table, Column
from astropy.io import ascii

import numpy as np

# Create our own custom color maps to transition from black to the color of interest
cmap_C0 = LinearSegmentedColormap.from_list("blue", ["k", "C0"])
cmap_C1 = LinearSegmentedColormap.from_list("orange", ["k", "C1"])
cmap_C2 = LinearSegmentedColormap.from_list("green", ["k", "C2"])
cmap_C2_alt = LinearSegmentedColormap.from_list("green", ["0.5", "C2"])


# for the points
pkwargs = {"marker":"o", "ms":3, "color":"C0", "ls":""}
# For the errorbars
ekwargs = {"lw_fine":0.8, "ebar_color":"0.2", "ebar_lw":0.4, "fmt":'none', "capsize":0.0}


# How do we create segments from X and Y?
# Create a set of line segments so that we can color them individually
# This creates the points as a N x 1 x 2 array so that we can stack points
# together easily to get the segments. The segments array for line collection
# needs to be numlines x points per line x 2 (x and y)
def plot_cline(ax, x, y, dates, cmap, lw):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap,
                        norm=plt.Normalize(np.min(dates), np.max(dates)))
    lc.set_array(dates)
    lc.set_linewidth(lw)
    ax.add_collection(lc)


# Custom routine for plotting polar error bars onto cartesian plane
def plot_errorbar(ax, thetas, rhos, theta_errs, rho_errs, **kwargs):
    '''
    All kwargs sent to plot. Thetas are in radians.
    '''

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
        delta = rho * (1/np.cos(theta_err) - 1)
        xs = (rho + delta) * np.sin(np.array([theta - theta_err, theta + theta_err]))
        ys = (rho + delta) * np.cos(np.array([theta - theta_err, theta + theta_err]))
        ax.plot(xs, ys, **kwargs)

        # rho error
        xs = np.sin(theta) * np.array([(rho - rho_err), (rho + rho_err)])
        ys = np.cos(theta) * np.array([(rho - rho_err), (rho + rho_err)])
        ax.plot(xs, ys, **kwargs)


# X_ABs = rho_ABs * np.cos(theta_ABs * np.pi/180)
# Y_ABs = rho_ABs * np.sin(theta_ABs * np.pi/180)

# plot_cline(ax[0,0], Y_ABs, X_ABs, dates_inner_good, cmap_C1, lw=lw)



data = ascii.read("data/visual_data_besselian.csv", format="csv")

ax_width = 2.7
hmargin = 0.2
ax_height = 2.7
sax_height = (ax_height - hmargin)/2
tmargin = 0.1
bmargin = 0.5
lmargin = 0.5
rmargin = 0.5
mmargin = 0.6

xx = 2 * ax_width + lmargin + rmargin + mmargin
yy = ax_height + tmargin + bmargin
print("Fig width", xx, "in")

fig = plt.figure(figsize=(xx,yy))

ax_sky = fig.add_axes([lmargin/xx, bmargin/yy, ax_width/xx, ax_height/yy])
ax_sky.set_xlabel(r'$\Delta \alpha \cos \delta\; [{}^{\prime\prime}]$')
ax_sky.set_ylabel(r'$\Delta \delta\; [{}^{\prime\prime}]$')
ax_sky.invert_xaxis()

# plot the star
ax_sky.plot(0,0, "*", ms=5, color="k", mew=0.1)

# plot the sky positions
X_ABs = data["sep"] * np.cos(data["pa"] * np.pi/180) # north
Y_ABs = data["sep"] * np.sin(data["pa"] * np.pi/180) # east
ax_sky.plot(Y_ABs, X_ABs, **pkwargs)

# plot the sky uncertainties

# set the frame to look fine
rad = 2.5
ax_sky.set_xlim(rad,-rad)
ax_sky.set_ylim(-rad,rad)

yr_lim = (1990, 2020)

ax_sep = fig.add_axes([(lmargin + ax_width + mmargin)/xx, 1 - (tmargin + sax_height)/yy, ax_width/xx, sax_height/yy])
ax_pa = fig.add_axes([(lmargin + ax_width + mmargin)/xx, bmargin/yy, ax_width/xx, sax_height/yy])

ax_sep.plot(data["epoch"], data["sep"], **pkwargs)
# ax_sep.errorbar(data["epoch"], data["sep"], yerr=data["sep_err"], ls="", color="C0")
ax_sep.set_ylabel(r'$\rho\;[{}^{\prime\prime}]$')
ax_sep.set_xlim(*yr_lim)
ax_sep.xaxis.set_ticklabels([])

ax_pa.plot(data["epoch"], data["pa"], **pkwargs)
# ax_pa.errorbar(data["epoch"], data["pa"], yerr=data["pa_err"], ls="", color="C0")
ax_pa.set_ylabel(r'$\theta\ [{}^\circ]$')
ax_pa.set_xlim(*yr_lim)

ax_pa.set_xlabel("Epoch [yr]")

fig.savefig("plots/sep_pa.pdf")
