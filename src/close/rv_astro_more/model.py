import astropy
import theano
import pymc3 as pm
import exoplanet as xo
import numpy as np
import pandas as pd
import re

# load the exoplanet part
import theano.tensor as tt
from astropy.io import ascii
from exoplanet.distributions import Angle

from src.constants import *
import src.data as d


with pm.Model() as model:

    # We'll include the parallax data as a prior on the parallax value
    mparallax = pm.Normal("mparallax", mu=27.31, sd=0.12)  # milliarcsec GAIA DR2
    parallax = pm.Deterministic("parallax", 1e-3 * mparallax)  # arcsec

    a_ang = pm.Uniform("aAng", 0.1, 10.0, testval=3.51)  # milliarcsec

    # the semi-major axis in au
    a = pm.Deterministic("a", 1e-3 * a_ang / parallax)  # au

    logP = pm.Uniform(
        "logP", lower=np.log(20.0), upper=np.log(50.0), testval=np.log(34.87846)
    )  # days
    P = pm.Deterministic("P", tt.exp(logP))

    e = pm.Uniform("e", lower=0, upper=1, testval=0.62)
    omega = Angle("omega", testval=80.5 * deg)  # omega_Aa
    Omega = Angle(
        "Omega", testval=110.0 * deg
    )  # - pi to pi # estimated assuming same as CB disk
    gamma = pm.Uniform("gamma", lower=0, upper=20, testval=10.1)  # km/s

    # uniform on cos incl. testpoint assuming same as CB disk.
    cos_incl = pm.Uniform(
        "cosIncl", lower=-1.0, upper=0.0, testval=np.cos(132.0 * deg)
    )  # radians, 0 to 180 degrees
    incl = pm.Deterministic("incl", tt.arccos(cos_incl))

    # Since we're doing an RV + astrometric fit, M2 now becomes a parameter of the model
    # use a bounded normal to enforce positivity
    PosNormal = pm.Bound(pm.Normal, lower=0.0)
    MAb = PosNormal("MAb", mu=0.3, sd=0.5, testval=0.3)  # solar masses

    t_periastron = pm.Uniform(
        "tPeri", lower=1130.0, upper=1170.0, testval=1159.00
    )  # + 2400000 days

    orbit = xo.orbits.KeplerianOrbit(
        a=a * au_to_R_sun,
        period=P,
        ecc=e,
        t_periastron=t_periastron,
        omega=omega,
        Omega=Omega,
        incl=incl,
        m_planet=MAb,
    )

    # now that we have a physical scale defined, the total mass of the system makes sense
    MA = pm.Deterministic("MA", orbit.m_total)
    MAa = pm.Deterministic("MAa", MA - MAb)

    # since we have 4 instruments, we need to predict 4 different dataseries
    def get_RVs(t1, t2, offset):
        """
        Helper function for RVs. 

        Args:
            t1: times to query for star 1
            t2 : times to query for star 2
            offset: (km/s)
            
        Returns:
            (rv1, rv2) [km/s] evaluated at those times with offset applied
       """
        # get the RV predictions
        # get_star_velocity and get_planet_velocity return (v_x, v_y, v_z) tuples, so we only need the v_z vector
        # but, note that since +Z points towards the observer, we actually want v_radial = -v_Z (see conv)
        rv1 = conv * orbit.get_star_velocity(t1)[2] + gamma + offset
        rv2 = conv * orbit.get_planet_velocity(t2)[2] + gamma + offset

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

    # define the RV likelihoods
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

    # get the astrometric predictions
    # since there is only one measurement no jitter
    rho_model, theta_model = orbit.get_relative_angles(
        d.anthonioz[0], parallax
    )  # arcsec

    # evaluate the astrometric likelihood functions
    pm.Normal("obs_rho", mu=rho_model, observed=d.anthonioz[1], sd=d.anthonioz[2])
    theta_diff = tt.arctan2(
        tt.sin(theta_model - d.anthonioz[3]), tt.cos(theta_model - d.anthonioz[3])
    )  # wrap-safe
    pm.Normal("obs_theta", mu=theta_diff, observed=0.0, sd=d.anthonioz[4])

# iterate through the list of free_RVs in the model to get things like
# ['logKAa_interval__', etc...] then use a regex to strip away
# the transformations (in this case, _interval__ and _angle__)
# \S corresponds to any character that is not whitespace
# https://docs.python.org/3/library/re.html
sample_vars = [re.sub("_\S*__", "", var.name) for var in model.free_RVs]

all_vars = [
    var.name
    for var in model.unobserved_RVs
    if ("_interval__" not in var.name)
    and ("_angle__" not in var.name)
    and ("_lowerbound__" not in var.name)
]
