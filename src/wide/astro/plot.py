import arviz as az
import corner
import exoplanet as xo
import matplotlib.pyplot as plt
import pymc3 as pm
import os 

import src.wide.astro.model as m
import src.data as d
from src.constants import *
from src.plot_utils import plot_cline, plot_nodes

plotdir = "figures/wide/astro/"

if not os.path.isdir(plotdir):
    os.makedirs(plotdir)

trace = pm.load_trace(directory="chains/wide/astro", model=m.model)

# view summary
df = az.summary(trace, var_names=m.all_vars)
print(df)

# write summary to disk
f = open(f"{plotdir}summary.txt", "w")
df.to_string(f)
f.close()


with az.rc_context(rc={"plot.max_subplots": 80}):
    # autocorrelation
    az.plot_autocorr(trace, var_names=m.sample_vars)
    plt.savefig(f"{plotdir}autocorr.png")

    # make a traceplot
    az.plot_trace(trace, var_names=m.all_vars)
    plt.savefig(f"{plotdir}trace.png")


# make a nice corner plot of the variables we care about
samples = pm.trace_to_dataframe(
    trace,
    varnames=[
        "Mtot",
        "a",
        "P",
        "e",
        "incl",
        "omega",
        "Omega",
        "phi"
    ],
)
samples["P"] /= yr
samples["incl"] /= deg
samples["omega"] /= deg
samples["Omega"] /= deg
fig = corner.corner(samples)
fig.savefig(f"{plotdir}corner.png")


# t_period = pm.Deterministic("tPeriod", t_fine * P)

#     # save some samples on a fine orbit for plotting purposes
#     rho, theta = orbit.get_relative_angles(t_period)
#     rho_save_sky = pm.Deterministic("rhoSaveSky", rho / au_to_R_sun / dpc)
#     theta_save_sky = pm.Deterministic("thetaSaveSky", theta)

#     rho, theta = orbit.get_relative_angles(t_data)
#     rho_save_data = pm.Deterministic("rhoSaveData", rho / au_to_R_sun / dpc)
#     theta_save_data = pm.Deterministic("thetaSaveData", theta)