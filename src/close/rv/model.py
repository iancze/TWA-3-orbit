import astropy
import exoplanet as xo
import numpy as np
import pandas as pd
import re

# load the exoplanet part
import pymc3 as pm
import theano.tensor as tt
from astropy import constants
from astropy import units as u
from astropy.io import ascii
from astropy.time import Time
from exoplanet.distributions import Angle

import src.notebook_setup  # run the DFM commands
from src.constants import *


def get_arrays(asciiTable, errDict=None, jitter=False):
    """
    Reformat ascii tables into pure numpy arrays of the right dimension.
    """

    output = []

    for star in ["Aa", "Ab"]:

        # get the RVs
        rv = asciiTable["RV_" + star]

        if type(rv) is astropy.table.column.MaskedColumn:
            mask = ~rv.mask  # values we want to keep when indexing
        else:
            mask = np.ones(len(rv), dtype="bool")

        rv = np.ascontiguousarray(rv[mask])
        date = np.ascontiguousarray(asciiTable["HJD"][mask])

        if errDict is None:
            err = np.ascontiguousarray(asciiTable["sigma_" + star][mask])
        else:
            err = np.ones(len(date), dtype=np.float64) * errDict[star]

        if jitter:
            err = (
                np.ones(len(date), dtype=np.float64) * 0.1
            )  # [km/s] assume a small error, since we'll infer.

        assert len(date) == len(rv), "date - rv length mismatch"
        assert len(date) == len(err), "date - err length mismatch"

        tup = (date, rv, err)

        output.append(tup)

    return output


jitter = True  # Do this to infer w/ jitter
datadir = "data/close/"

# load all the data
data_cfa = ascii.read(f"{datadir}cfa.dat")
# cfa errors are provided in table
cfa1, cfa2 = get_arrays(data_cfa, jitter=jitter)

data_keck = ascii.read(f"{datadir}keck.dat", format="tab", fill_values=[("X", 0)])
err_keck = {"Aa": 0.63, "Ab": 0.85, "B": 0.59}  # km/s
keck1, keck2 = get_arrays(data_keck, err_keck, jitter=jitter)

data_feros = ascii.read(f"{datadir}feros.dat")
err_feros = {"Aa": 2.61, "Ab": 3.59, "B": 2.60}  # km/s
feros1, feros2 = get_arrays(data_feros, err_feros, jitter=jitter)

data_dupont = ascii.read(f"{datadir}dupont.dat", fill_values=[("X", 0)])
err_dupont = {"Aa": 1.46, "Ab": 2.34, "B": 3.95}  # km/s
dupont1, dupont2 = get_arrays(data_dupont, err_dupont, jitter=jitter)

data = [data_cfa, data_keck, data_feros, data_dupont]

with pm.Model() as model:

    # parameters
    # P, gamma, Ka, Kb, e, omegaA, T0
    # Delta CfA - Keck
    # Delta CfA - Feros
    # Delta CfA - du Pont
    # jitter for each instrument?

    # Parameters
    logKAa = pm.Uniform(
        "logKAa", lower=0, upper=np.log(100), testval=np.log(25)
    )  # km/s
    logKAb = pm.Uniform(
        "logKAb", lower=0, upper=np.log(100), testval=np.log(25)
    )  # km/s

    KAa = pm.Deterministic("KAa", tt.exp(logKAa))
    KAb = pm.Deterministic("KAb", tt.exp(logKAb))

    logP = pm.Uniform(
        "logP", lower=0, upper=np.log(50.0), testval=np.log(34.87846)
    )  # days

    P = pm.Deterministic("P", tt.exp(logP))

    e = pm.Uniform("e", lower=0, upper=1, testval=0.62)

    omega = Angle("omega", testval=80.5 * deg)  # omega_Aa

    gamma = pm.Uniform("gamma", lower=0, upper=20, testval=10.1)

    t_periastron = pm.Uniform(
        "tPeri", lower=52690.0, upper=52720.0, testval=52704.55
    )  # + 2400000 days

    orbit = xo.orbits.KeplerianOrbit(
        period=P, ecc=e, t_periastron=t_periastron, omega=omega
    )

    # since we have 4 instruments, we need to predict 4 different dataseries
    def get_RVs(t1, t2, offset):
        """
        Helper function for RVs. Closure should encapsulate current K1, K2 values, I hope.

        Args:
            orbit: exoplanet object
            t1: times to query for star 1
            t2 : times to query for star 2
            offset: (km/s)
            
        Returns:
            (rv1, rv2) [km/s] evaluated at those times with offset applied
       """
        rv1 = (
            1e-3 * orbit.get_radial_velocity(t1, 1e3 * tt.exp(logKAa)) + gamma + offset
        )  # km/s
        rv2 = (
            1e-3 * orbit.get_radial_velocity(t2, -1e3 * tt.exp(logKAb)) + gamma + offset
        )  # km/s

        return (rv1, rv2)

    offset_keck = pm.Normal("offsetKeck", mu=0.0, sd=5.0)  # km/s
    offset_feros = pm.Normal("offsetFeros", mu=0.0, sd=5.0)  # km/s
    offset_dupont = pm.Normal("offsetDupont", mu=0.0, sd=5.0)  # km/s

    # expects m/s
    # dates are the first entry in each tuple of (date, rv, err)
    rv1_cfa, rv2_cfa = get_RVs(cfa1[0], cfa2[0], 0.0)
    rv1_keck, rv2_keck = get_RVs(keck1[0], keck2[0], offset_keck)
    rv1_feros, rv2_feros = get_RVs(feros1[0], feros2[0], offset_feros)
    rv1_dupont, rv2_dupont = get_RVs(dupont1[0], dupont2[0], offset_dupont)

    logjit_cfa = pm.Uniform(
        "logjittercfa", lower=-5.0, upper=np.log(10), testval=np.log(1.0)
    )
    logjit_keck = pm.Uniform(
        "logjitterkeck", lower=-5.0, upper=np.log(10), testval=np.log(1.0)
    )
    logjit_feros = pm.Uniform(
        "logjitterferos", lower=-5.0, upper=np.log(10), testval=np.log(1.0)
    )
    logjit_dupont = pm.Uniform(
        "logjitterdupont", lower=-5.0, upper=np.log(10), testval=np.log(1.0)
    )
    jit_cfa = pm.Deterministic("jitCfa", tt.exp(logjit_cfa))
    jit_keck = pm.Deterministic("jitKeck", tt.exp(logjit_keck))
    jit_feros = pm.Deterministic("jitFeros", tt.exp(logjit_feros))
    jit_dupont = pm.Deterministic("jitDupont", tt.exp(logjit_dupont))

    # get the total errors
    def get_err(rv_err, logjitter):
        return tt.sqrt(rv_err ** 2 + tt.exp(2 * logjitter))

    # define the likelihoods
    pm.Normal(
        "cfaRV1Obs", mu=rv1_cfa, observed=cfa1[1], sd=get_err(cfa1[2], logjit_cfa)
    )
    pm.Normal(
        "cfaRV2Obs", mu=rv2_cfa, observed=cfa2[1], sd=get_err(cfa2[2], logjit_cfa)
    )
    pm.Normal(
        "keckRV1Obs", mu=rv1_keck, observed=keck1[1], sd=get_err(keck1[2], logjit_keck)
    )
    pm.Normal(
        "keckRV2Obs", mu=rv2_keck, observed=keck2[1], sd=get_err(keck2[2], logjit_keck)
    )
    pm.Normal(
        "ferosRV1Obs",
        mu=rv1_feros,
        observed=feros1[1],
        sd=get_err(feros1[2], logjit_feros),
    )
    pm.Normal(
        "ferosRV2Obs",
        mu=rv2_feros,
        observed=feros2[1],
        sd=get_err(feros2[2], logjit_feros),
    )
    pm.Normal(
        "dupontRV1Obs",
        mu=rv1_dupont,
        observed=dupont1[1],
        sd=get_err(dupont1[2], logjit_dupont),
    )
    pm.Normal(
        "dupontRV2Obs",
        mu=rv2_dupont,
        observed=dupont2[1],
        sd=get_err(dupont2[2], logjit_dupont),
    )

# iterate through the list of free_RVs in the model to get things like
# ['logKAa_interval__', etc...] then use a regex to strip away
# the transformations (in this case, _interval__ and _angle__)
# \S corresponds to any character that is not whitespace
# https://docs.python.org/3/library/re.html
sample_vars = [re.sub("_\S*__", "", var.name) for var in model.free_RVs]

all_vars = [
    var.name
    for var in model.unobserved_RVs
    if ("_interval__" not in var.name) and ("_angle__" not in var.name)
]

