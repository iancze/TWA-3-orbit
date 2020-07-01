import os
from pathlib import Path

import arviz as az
import corner
import exoplanet as xo
import matplotlib.pyplot as plt
import pymc3 as pm

import model as m
import twa.data as d
from twa.constants import *
from twa.plot_utils import efficient_autocorr, efficient_trace
from twa import joint

plotdir = Path("figures")
diagnostics = True

if not os.path.isdir(plotdir):
    os.makedirs(plotdir)

trace = pm.load_trace(directory="chains", model=m.model)

if diagnostics:
    ar_data = az.from_pymc3(trace=trace)
    # view summary
    df = az.summary(ar_data, var_names=m.all_vars)
    print(df)

    # write summary to disk
    f = open(plotdir / "summary.txt", "w")
    df.to_string(f)
    f.close()

    with az.rc_context(rc={"plot.max_subplots": 80}):
        stem = str(plotdir / "autocorr{:}.png")
        efficient_autocorr(ar_data, var_names=m.all_vars, figstem=stem)

        # make a traceplot
        stem = str(plotdir / "trace{:}.png")
        efficient_trace(ar_data, var_names=m.all_vars, figstem=stem)

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

    # save the final df
    df.to_csv("chains/trace.csv")

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
    fig.savefig(plotdir / "corner-inner.png", dpi=120)

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
    fig.savefig(plotdir / "corner-outer.png", dpi=120)

    # masses
    masses = ["MAa", "MAb", "MA", "MB", "Mtot"]
    fig = corner.corner(df[masses])
    fig.savefig(plotdir / "corner-masses.png", dpi=120)

    # posterior on periastron passage
    # r_p = a * (1 - e)
    df["rp"] = df["aOuter"] * (1 - df["eOuter"])  # au

    # mutual inclination between inner orbit and outer orbit
    muts = ["thetaDiskInner", "thetaInnerOuter", "thetaDiskOuter", "rp"]
    fig = corner.corner(df[muts])
    fig.savefig(plotdir / "corner-muts.png", dpi=120)

df, df_decreasing = joint.sort_samples(trace, m)
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
fig.savefig(plotdir / "corner-increasing.png", dpi=120)

import sys

sys.exit()

fig = joint.plot_interior_RV(trace, m)
fig.savefig(plotdir / "RV_inner.pdf")

fig = joint.plot_sep_pa(trace, m)
fig.savefig(plotdir / "sep_pa.pdf")
