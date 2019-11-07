import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.io import ascii
from astropy import units as u
from astropy import constants
import astropy

# load the exoplanet part
import pymc3 as pm
import theano.tensor as tt

import corner

import exoplanet as xo
from exoplanet.distributions import Angle

with model:
    map_sol = xo.optimize()


sampler = xo.PyMC3Sampler(window=100, finish=500)
with model:
    burnin = sampler.tune(tune=2500, step_kwargs=dict(target_accept=0.95))
    trace = sampler.sample(draws=3000)

pm.traceplot(
    trace,
    varnames=[
        "mparallax",
        "a_ang",
        "a",
        "e",
        "P",
        "omega",
        "Omega",
        "cosIncl",
        "offsetKeck",
        "offsetFeros",
        "offsetDupont",
        "logjittercfa",
        "logjitterkeck",
        "logjitterferos",
        "logjitterdupont",
    ],
)


# To assess the quality of the fit, we should go and plot the fit and residuals for all of the data points individually and together.
#
# The phase-folding plot only really works for a fixed value of `tperi`, `P`. So, we can plot the MAP phase fold. But other than that it only makes sense to plot the orbit scatter on the actual series of points (minus any offset, too).

# In[23]:


# plot everything ontop in a single plot

pkw = {"marker": ".", "ls": ""}
ekw = {"marker": ".", "ms": 5.0, "ls": "", "elinewidth": 1.2}


def get_phase(dates, pos):
    return ((dates - pos["tPeri"]) % pos["P"]) / pos["P"]


# nsamples = 10
# choices = np.random.choice(np.arange(len(trace)), size=nsamples)

# just choose one representative sample
np.random.seed(43)
choice = np.random.choice(np.arange(len(trace)))

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8, 4))

with model:

    pos = trace[choice]

    tperi = pos["tPeri"]
    P = pos["P"]

    # calculate the errors for each instrument
    cfa_err1 = np.sqrt(cfa1[2] ** 2 + np.exp(2 * pos["logjittercfa"]))
    cfa_err2 = np.sqrt(cfa2[2] ** 2 + np.exp(2 * pos["logjittercfa"]))

    keck_err1 = np.sqrt(keck1[2] ** 2 + np.exp(2 * pos["logjitterkeck"]))
    keck_err2 = np.sqrt(keck2[2] ** 2 + np.exp(2 * pos["logjitterkeck"]))

    feros_err1 = np.sqrt(feros1[2] ** 2 + np.exp(2 * pos["logjitterferos"]))
    feros_err2 = np.sqrt(feros2[2] ** 2 + np.exp(2 * pos["logjitterferos"]))

    dupont_err1 = np.sqrt(dupont1[2] ** 2 + np.exp(2 * pos["logjitterdupont"]))
    dupont_err2 = np.sqrt(dupont2[2] ** 2 + np.exp(2 * pos["logjitterdupont"]))

    # plot RV1 model
    ax[0].axhline(pos["gamma"], lw=1.0, color="k", ls=":")
    ax[0].plot(xs_phase, pos["RV1Dense"], zorder=-1)

    # at data locations
    ax[0].errorbar(get_phase(cfa1[0], pos), cfa1[1], yerr=cfa_err1, **ekw, zorder=0)
    ax[0].errorbar(
        get_phase(keck1[0], pos),
        keck1[1] - pos["offsetKeck"],
        yerr=keck_err1,
        **ekw,
        zorder=0
    )
    ax[0].errorbar(
        get_phase(feros1[0], pos),
        feros1[1] - pos["offsetFeros"],
        yerr=feros_err1,
        **ekw,
        zorder=0
    )
    ax[0].errorbar(
        get_phase(dupont1[0], pos),
        dupont1[1] - pos["offsetDupont"],
        yerr=dupont_err1,
        **ekw,
        zorder=0
    )

    # plot RV2
    ax[1].axhline(pos["gamma"], lw=1.0, color="k", ls=":")
    ax[1].plot(xs_phase, pos["RV2Dense"], zorder=-1)

    # at data locations
    ax[1].errorbar(get_phase(cfa2[0], pos), cfa2[1], yerr=cfa_err2, **ekw, zorder=0)
    ax[1].errorbar(
        get_phase(keck2[0], pos),
        keck2[1] - pos["offsetKeck"],
        yerr=keck_err2,
        **ekw,
        zorder=0
    )
    ax[1].errorbar(
        get_phase(feros2[0], pos),
        feros2[1] - pos["offsetFeros"],
        yerr=feros_err2,
        **ekw,
        zorder=0
    )
    ax[1].errorbar(
        get_phase(dupont2[0], pos),
        dupont2[1] - pos["offsetDupont"],
        yerr=dupont_err2,
        **ekw,
        zorder=0
    )

    ax[1].set_xlim(0.0, 1.0)
    ax[0].set_ylabel(r"$v_\mathrm{Aa}$ $[\mathrm{km s}^{-1}]$")
    ax[1].set_ylabel(r"$v_\mathrm{Ab}$ $[\mathrm{km s}^{-1}]$")
    ax[1].set_xlabel("phase")

fig.subplots_adjust(top=0.98, bottom=0.18, hspace=0.05)
fig.savefig("A_sb_orbit.pdf")


# In[25]:


samples = pm.trace_to_dataframe(
    trace, varnames=["P", "e", "gamma", "omega", "Omega", "incl", "MA", "MAa", "MAb"]
)
samples["omega"] /= deg
samples["Omega"] /= deg
samples["incl"] /= deg
corner.corner(samples)


# In[ ]:

