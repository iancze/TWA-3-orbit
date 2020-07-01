import arviz as az
import corner
import exoplanet as xo
import matplotlib.pyplot as plt
import pymc3 as pm
import os
from pathlib import Path

import model as m
import twa.data as d
from twa import wide
from twa.constants import *
from twa.plot_utils import plot_cline, plot_nodes
from twa.plot_utils import efficient_autocorr, efficient_trace

plotdir = Path("figures")
diagnostics = True

if not os.path.isdir(plotdir):
    os.makedirs(plotdir)

trace = pm.load_trace(directory="chains", model=m.model)

if diagnostics:
    ar_data = az.from_pymc3(trace=trace)

    # view summary
    df = az.summary(ar_data, var_names=m.all_vars)

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
    samples = pm.trace_to_dataframe(
        trace, varnames=["Mtot", "a", "P", "e", "incl", "omega", "Omega", "phi"],
    )
    samples["P"] /= yr
    samples["incl"] /= deg
    samples["omega"] /= deg
    samples["Omega"] /= deg
    fig = corner.corner(samples)
    fig.savefig(plotdir / "corner.png")

import sys

sys.exit()

fig = wide.plot_sep_pa(trace, m)
fig.savefig(plotdir / "sep_pa.pdf")


fig = wide.plot_sky(trace, m)
fig.savefig(plotdir / "sky.pdf")
