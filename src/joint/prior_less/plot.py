import arviz as az
import corner
import exoplanet as xo
import matplotlib.pyplot as plt
import pymc3 as pm
import os

import src.joint.rv_astro_disk_less.model as m
from src.plot_utils import efficient_autocorr, efficient_trace
import src.data as d
from src.constants import *

plotdir = "figures/joint/prior_less/"

if not os.path.isdir(plotdir):
    os.makedirs(plotdir)

trace = pm.load_trace(directory="chains/joint/prior_less", model=m.model)

# view summary
df = az.summary(trace, var_names=m.all_vars)
print(df)

# write summary to disk
f = open(f"{plotdir}summary.txt", "w")
df.to_string(f)
f.close()


with az.rc_context(rc={"plot.max_subplots": 80}):
    stem = f"{plotdir}autocorr" + "{:}.png"
    efficient_autocorr(trace, var_names=m.all_vars, figstem=stem)

    # make a traceplot
    stem = f"{plotdir}trace" + "{:}.png"
    efficient_trace(trace, var_names=m.all_vars, figstem=stem)


# make a nice corner plot of the variables we care about
df = pm.trace_to_dataframe(trace)

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
muts = ["thetaDiskInner", "thetaInnerOuter", "thetaDiskOuter", "r_p"] 
fig = corner.corner(df[muts])
fig.savefig(f"{plotdir}corner-muts.png", dpi=120)
