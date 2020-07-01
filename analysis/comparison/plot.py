"""
Compare the omega vs. Omega plots for the Wide-Astro and RV-Astro-Disk-More posteriors.
"""

import os
from pathlib import Path

import arviz as az
import corner
import exoplanet as xo
import matplotlib.pyplot as plt
import pymc3 as pm
import pandas as pd

import twa.data as d
from twa.constants import *
from twa.plot_utils import efficient_autocorr, efficient_trace
from twa import joint

plotdir = Path("figures")
diagnostics = False

if not os.path.isdir(plotdir):
    os.makedirs(plotdir)

p = Path(os.getenv("TWA_ANALYSIS_ROOT"))

# load the df's directly from the CSV
df_astro = pd.read_csv(p / "wide" / "chains" / "current.csv")
df_joint = pd.read_csv(p / "joint" / "rv_astro_disk_more" / "chains" / "current.csv")

# convert the relevant params
df_astro["omega"] /= deg
df_astro["Omega"] /= deg

labels = [r"$\omega$", r"$\Omega$"]
range = [(-180, 180), (-180, 180)]

fig = corner.corner(df_astro[["omega", "Omega"]], labels=labels, range=range)
fig.savefig(plotdir / "corner-astro.png", dpi=120)


df_joint["omegaOuter"] /= deg
df_joint["OmegaOuter"] /= deg

fig = corner.corner(df_joint[["omegaOuter", "OmegaOuter"]], labels=labels, range=range)
fig.savefig(plotdir / "corner-joint.png", dpi=120)

