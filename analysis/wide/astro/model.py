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

from twa.constants import *
import twa.data as d

zeros = np.zeros_like(d.wds[0])

with pm.Model() as model:

    # 27.31 mas +/- 0.12 mas # from GAIA
    mparallax = pm.Normal("mparallax", mu=27.31, sd=0.12)  # milliarcsec GAIA DR2
    parallax = pm.Deterministic("parallax", 1e-3 * mparallax)  # arcsec

    # a_ang = pm.Uniform("aAng", 0.1, 4.0, testval=2.0) # arcsec
    a_ang = pm.Gamma("aAng", alpha=3.0, beta=1.5, testval=1.9)  # arcsec

    # the semi-major axis in au
    a = pm.Deterministic("a", a_ang / parallax)  # au

    # we expect the period to be somewhere in the range of 400 years,
    # so we'll set a broad prior on logP
    logP = pm.Normal("logP", mu=np.log(400), sd=0.8)
    P = pm.Deterministic("P", tt.exp(logP) * yr)  # days

    # omega = Angle("omega", testval=180 * deg)  # - pi to pi
    omega = Angle("omega")  # - pi to pi

    # because we don't have RV information in this model,
    # Omega and Omega + 180 are degenerate.
    # I think flipping Omega also advances omega and phi by pi
    # So, just limit it to -90 to 90 degrees
    # Omega_intermediate = Angle("OmegaIntermediate")  # - pi to pi
    # Omega = pm.Deterministic("Omega", Omega_intermediate / 2 + np.pi / 2)  # 0 to pi
    Omega = Angle("Omega")

    phi = Angle("phi")  # phase (Mean anom) at t = 0

    n = 2 * np.pi * tt.exp(-logP) / yr

    t_periastron = pm.Deterministic("tPeri", (phi + omega) / n)

    # definitely going clockwise, so just enforce i > 90
    cos_incl = pm.Uniform("cosIncl", lower=-1.0, upper=0.0, testval=np.cos(3.0))
    incl = pm.Deterministic("incl", tt.arccos(cos_incl))

    e = pm.Uniform("e", lower=0.0, upper=1.0, testval=0.2)

    # n.b. that we include an extra conversion for a, because exoplanet expects a in R_sun
    orbit = xo.orbits.KeplerianOrbit(
        a=a * au_to_R_sun,
        t_periastron=t_periastron,
        period=P,
        incl=incl,
        ecc=e,
        omega=omega,
        Omega=Omega,
    )

    # now that we have a physical scale defined, we can also calculate the total mass of the system
    Mtot = pm.Deterministic("Mtot", orbit.m_total)

    # put a reasonable prior on the total mass, to prevent it from going too high
    pm.Potential(
        "MtotPrior", pm.HalfNormal.dist(sigma=3.0).logp(Mtot)
    )  # centered on 0, sigma = 3

    rho_model, theta_model = orbit.get_relative_angles(
        d.wds[0], parallax
    )  # the rho, theta model values

    # add jitter terms to both separation and position angle
    log_rho_s = pm.Normal("logRhoS", mu=np.log(np.median(d.wds[2])), sd=2.0)
    log_theta_s = pm.Normal("logThetaS", mu=np.log(np.median(d.wds[4])), sd=2.0)

    rho_tot_err = tt.sqrt(d.wds[2] ** 2 + tt.exp(2 * log_rho_s))
    theta_tot_err = tt.sqrt(d.wds[4] ** 2 + tt.exp(2 * log_theta_s))

    pm.Normal("obs_rho", mu=rho_model, observed=d.wds[1], sd=rho_tot_err)

    # we need to calculate the difference
    theta_diff = tt.arctan2(
        tt.sin(theta_model - d.wds[3]), tt.cos(theta_model - d.wds[3])
    )
    pm.Normal("obs_theta", mu=theta_diff, observed=zeros, sd=theta_tot_err)


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
    and ("_log__" not in var.name)
]
