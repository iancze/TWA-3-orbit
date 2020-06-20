import arviz as az
import corner
import exoplanet as xo
import matplotlib.pyplot as plt
import pymc3 as pm
import os
import pandas as pd

import src.joint.prior.model as m
from src.plot_utils import efficient_autocorr, efficient_trace
import src.data as d
from src.constants import *

plotdir = "figures/joint/prior/"

if not os.path.isdir(plotdir):
    os.makedirs(plotdir)

# make a nice corner plot of the variables we care about
df = pd.read_csv("chains/joint/prior/current.csv")

# convert all params
df["omegaInner"] /= deg
df["OmegaInner"] /= deg
df["inclInner"] /= deg

df["omegaOuter"] /= deg
df["OmegaOuter"] /= deg
df["inclOuter"] /= deg

df["thetaDiskInner"] /= deg 
df["thetaInnerOuter"] /= deg 
df["thetaDiskOuter"] /= deg

df["POuter"] /= yr

# just the inner parameters
inner = [
    "MAb",
    "MA",
    "aInner",
    "PInner",
    "eInner",
    "omegaInner",
    "OmegaInner",
    "inclInner",
    "tPeriastronInner",
]
fig = corner.corner(df[inner])
fig.savefig(f"{plotdir}corner-inner.png", dpi=120)

# just the outer parameters
outer = [
    "MA",
    "MB",
    "aOuter",
    "POuter",
    "omegaOuter",
    "OmegaOuter",
    "eOuter",
    "inclOuter",
    "gammaOuter",
    "tPeriastronOuter",
]
fig = corner.corner(df[outer])
fig.savefig(f"{plotdir}corner-outer.png", dpi=120)

# masses
masses = ["MAa", "MAb", "MA", "MB", "Mtot"]
fig = corner.corner(df[masses])
fig.savefig(f"{plotdir}corner-masses.png", dpi=120)


# posterior on periastron passage 
# r_p = a * (1 - e)
df["r_p"] = df["aOuter"] * (1 - df["eOuter"]) # au

# mutual inclination between inner orbit and outer orbit
muts = ["thetaDiskInner", "thetaInnerOuter", "thetaDiskOuter"] 
fig = corner.corner(df[muts])
fig.savefig(f"{plotdir}corner-muts.png", dpi=120)
