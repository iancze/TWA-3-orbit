import astropy
import exoplanet as xo
import numpy as np
import pandas as pd
import re

# load the exoplanet part
import pymc3 as pm
import theano.tensor as tt

from exoplanet.distributions import Angle

import src.notebook_setup  # run the DFM commands
from src.constants import *
import src.data as d

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

    # relative to jd0
    t_periastron = pm.Uniform(
        "tPeri", lower=1130.0, upper=1180.0, testval=1145.0
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
    rv1_cfa, rv2_cfa = get_RVs(d.cfa1[0], d.cfa2[0], 0.0)
    rv1_keck, rv2_keck = get_RVs(d.keck1[0], d.keck2[0], offset_keck)
    rv1_feros, rv2_feros = get_RVs(d.feros1[0], d.feros2[0], offset_feros)
    rv1_dupont, rv2_dupont = get_RVs(d.dupont1[0], d.dupont2[0], offset_dupont)

    logjit_cfa = pm.Uniform(
        "logjittercfa", lower=-5.0, upper=np.log(5), testval=np.log(1.0)
    )
    logjit_keck = pm.Uniform(
        "logjitterkeck", lower=-5.0, upper=np.log(5), testval=np.log(1.0)
    )
    logjit_feros = pm.Uniform(
        "logjitterferos", lower=-5.0, upper=np.log(5), testval=np.log(1.0)
    )
    logjit_dupont = pm.Uniform(
        "logjitterdupont", lower=-5.0, upper=np.log(5), testval=np.log(1.0)
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
        "cfaRV1Obs", mu=rv1_cfa, observed=d.cfa1[1], sd=get_err(d.cfa1[2], logjit_cfa)
    )
    pm.Normal(
        "cfaRV2Obs", mu=rv2_cfa, observed=d.cfa2[1], sd=get_err(d.cfa2[2], logjit_cfa)
    )
    pm.Normal(
        "keckRV1Obs",
        mu=rv1_keck,
        observed=d.keck1[1],
        sd=get_err(d.keck1[2], logjit_keck),
    )
    pm.Normal(
        "keckRV2Obs",
        mu=rv2_keck,
        observed=d.keck2[1],
        sd=get_err(d.keck2[2], logjit_keck),
    )
    pm.Normal(
        "ferosRV1Obs",
        mu=rv1_feros,
        observed=d.feros1[1],
        sd=get_err(d.feros1[2], logjit_feros),
    )
    pm.Normal(
        "ferosRV2Obs",
        mu=rv2_feros,
        observed=d.feros2[1],
        sd=get_err(d.feros2[2], logjit_feros),
    )
    pm.Normal(
        "dupontRV1Obs",
        mu=rv1_dupont,
        observed=d.dupont1[1],
        sd=get_err(d.dupont1[2], logjit_dupont),
    )
    pm.Normal(
        "dupontRV2Obs",
        mu=rv2_dupont,
        observed=d.dupont2[1],
        sd=get_err(d.dupont2[2], logjit_dupont),
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

