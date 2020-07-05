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
from twa import joint

plotdir = Path("figures")

trace = pm.load_trace(directory="chains", model=m.model)

df = pm.trace_to_dataframe(trace)
joint.plot_triangles(df, plotdir)

diagnostics = False
if diagnostics:
    joint.plot_summaries(trace, m, plotdir)

# select where v_B is increasing
df_inc = df.loc[df["increasing"] == True]
joint.plot_triangles(df_inc, plotdir / "inc")

df_dec = df.loc[df["increasing"] == False]
joint.plot_triangles(df_dec, plotdir / "dec")


# fig = joint.gamma_A_posterior(trace, m)
# fig.savefig(plotdir / "gamma_A.pdf")


fig = joint.plot_interior_RV(trace, m)
fig.savefig(plotdir / "RV_inner.pdf")

fig = joint.plot_sep_pa(trace, m)
fig.savefig(plotdir / "sep_pa.pdf")

