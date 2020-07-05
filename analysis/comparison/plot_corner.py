import corner
import matplotlib.pyplot as plt
import pandas as pd
import os

from collections import OrderedDict

import twa.data as d
from twa.constants import *
from twa import joint
from pathlib import Path


import matplotlib

matplotlib.rcParams["axes.labelsize"] = "x-small"
matplotlib.rcParams["xtick.labelsize"] = "x-small"
matplotlib.rcParams["ytick.labelsize"] = "x-small"

p = Path(os.getenv("TWA_ANALYSIS_ROOT"))

df = pd.read_csv(p / "joint" / "rv_astro_disk_more" / "chains" / "current.csv")


df["omegaOuter"] /= deg
df["OmegaOuter"] /= deg
df["tPeriastronOuter"] = df["tPeriastronOuter"] / yr + jd0 - 2450000
pars = ["omegaOuter", "OmegaOuter", "tPeriastronOuter"]
labels = [
    r"$\omega_\mathrm{A}$ $[{}^\circ]$",
    r"$\Omega_\mathrm{outer}$ $[{}^\circ]$",
    r"$T_{0,\mathrm{outer}}$ [JD]",
]
ranges = [(-180, 180), (-60, 160), (1100, 2200)]

levels = 1.0 - np.exp(-0.5 * np.arange(1.0, 2.1, 1.0) ** 2)

# select where v_B is increasing
df_inc = df.loc[df["increasing"] == True]
df_dec = df.loc[df["increasing"] == False]

# \textwidth=7.1in
# \columnsep=0.3125in
# column width = (7.1 - 0.3125)/2 = 3.393

all_samples = df[pars].to_numpy()
inc_samples = df_inc[pars].to_numpy()
dec_samples = df_dec[pars].to_numpy()

xx = 3.393
yy = 3.393
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(xx, yy))

bins = [
    np.linspace(-180, 180, num=30),
    np.linspace(-60, 160, num=30),
    np.linspace(1100, 2200, num=30),
]

# Add a slightly wider white line to make this pop
# corner.corner(
#     all_samples,
#     fig=fig,
#     labels=labels,
#     color="k",
#     plot_datapoints=False,
#     plot_density=False,
#     hist_kwargs={"color": "k", "zorder": 0, "lw": 1.8},
#     bins=bins,
#     levels=levels,
#     no_fill_contours=True,
#     contour_kwargs={"zorder": 0, "linewidths": 1.2, "colors": "w"},
# )

corner.corner(
    inc_samples,
    fig=fig,
    labels=labels,
    color="C0",
    plot_datapoints=False,
    plot_density=False,
    hist_kwargs={"color": "C0", "zorder": 10, "lw": 1.0},
    bins=bins,
    levels=levels,
    no_fill_contours=True,
    contour_kwargs={"zorder": 10, "linewidths": 1.0},
)

corner.corner(
    dec_samples,
    fig=fig,
    labels=labels,
    color="C1",
    plot_datapoints=False,
    range=ranges,
    plot_density=False,
    no_fill_contours=True,
    hist_kwargs={"color": "C1", "zorder": 9, "lw": 1.0},
    bins=bins,
    levels=levels,
    contour_kwargs={"zorder": 9, "linewidths": 1.0},
)


left_offsets = [-0.45, -0.45]
bottom_offsets = [-0.45, -0.45, -0.45]

# Go through all 1D histograms and readjust the height
heights = [2.5e3, 4e3, 8e3]
for j, h in enumerate(heights):
    ax[j, j].set_ylim(0, h)

# go through and fix the labels
for k, offset in enumerate(bottom_offsets):
    ax[-1, k].xaxis.set_label_coords(0.5, offset)
    # ax[-1, k].set_xlabel(label)

for k, offset in enumerate(left_offsets):
    ax[k + 1, 0].yaxis.set_label_coords(offset, 0.5)

b = 0.14
fig.subplots_adjust(left=b, right=1 - b, bottom=b, top=1 - b)

fig.text(0.5, 0.79, r"$v_B\;\uparrow$", color="C0", ha="center")
fig.text(0.5, 0.71, r"$v_B\;\downarrow$", color="C1", ha="center")

fig.savefig("corner.pdf")
