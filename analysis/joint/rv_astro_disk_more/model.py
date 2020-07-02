import os
import re
from pathlib import Path

import astropy
import exoplanet as xo
import numpy as np
import pandas as pd
import pymc3 as pm
import theano

# load the exoplanet part
import theano.tensor as tt
from astropy.io import ascii
from exoplanet.distributions import Angle

import twa.data as d
from twa.constants import *

zeros = np.zeros_like(d.wds[0])

# get the root datadir from environment variables
p = Path(os.getenv("TWA_DATA_ROOT"))
diskdir = p / "disk"

# flip the covariance of the inclination samples, i.e., rederive it
flatchain = np.load(diskdir / "flatchain.npy")
disk_samples = d.flatchain[:, [0, 9, 10]]
disk_samples[:, 2] -= 90.0  # convert Omega conventions
disk_samples[:, 1] = 180.0 - disk_samples[:, 1]  # make > 90
disk_samples[:, [1, 2]] *= deg  # convert *to* radians
mass_samples, incl_samples, Omega_samples = disk_samples.T

# can we evaluate the multivariate normal approximation to these correlations?
disk_mu = np.mean(disk_samples, axis=0)
disk_cov = np.cov(disk_samples, rowvar=False)

with pm.Model() as model:

    # We'll include the parallax data as a prior on the parallax value
    mparallax = pm.Normal("mparallax", mu=27.31, sd=0.12)  # milliarcsec GAIA DR2
    parallax = pm.Deterministic("parallax", 1e-3 * mparallax)  # arcsec

    a_ang_inner = pm.Uniform("aAngInner", 0.1, 10.0, testval=4.613)  # milliarcsec

    # the semi-major axis in au
    a_inner = pm.Deterministic("aInner", 1e-3 * a_ang_inner / parallax)  # au

    logP_inner = pm.Uniform(
        "logPInner", lower=np.log(20), upper=np.log(50.0), testval=np.log(34.88)
    )  # days
    P_inner = pm.Deterministic("PInner", tt.exp(logP_inner))

    e_inner = pm.Uniform("eInner", lower=0, upper=1, testval=0.63)

    omega_inner = Angle("omegaInner", testval=1.415)  # omega_Aa
    Omega_inner = Angle("OmegaInner", testval=1.821)

    # constrained to be i > 90
    cos_incl_inner = pm.Uniform("cosInclInner", lower=-1.0, upper=0.0, testval=-0.659)
    incl_inner = pm.Deterministic("inclInner", tt.arccos(cos_incl_inner))

    MAb = pm.Normal("MAb", mu=0.29, sd=0.5, testval=0.241)  # solar masses

    t_periastron_inner = pm.Uniform(
        "tPeriastronInner", lower=1140.0, upper=1180.0, testval=1159.57
    )  # + 2400000 + jd0

    orbit_inner = xo.orbits.KeplerianOrbit(
        a=a_inner * au_to_R_sun,
        period=P_inner,
        ecc=e_inner,
        t_periastron=t_periastron_inner,
        omega=omega_inner,
        Omega=Omega_inner,
        incl=incl_inner,
        m_planet=MAb,
    )

    # derived properties from inner orbit
    MA = pm.Deterministic("MA", orbit_inner.m_total)
    MAa = pm.Deterministic("MAa", MA - MAb)

    # we expect the period to be somewhere in the range of 25 years,
    # so we'll set a broad prior on logP
    logP_outer = pm.Normal("logPOuter", mu=5.9, sd=0.8, testval=5.7)  # yrs
    P_outer = pm.Deterministic("POuter", tt.exp(logP_outer) * yr)  # days

    omega_outer = Angle("omegaOuter")  # - pi to pi
    Omega_outer = Angle("OmegaOuter")  # - pi to pi

    phi_outer = Angle("phiOuter")

    n = 2 * np.pi * tt.exp(-logP_outer) / yr  # radians per day

    t_periastron_outer = pm.Deterministic(
        "tPeriastronOuter", (phi_outer + omega_outer) / n
    )

    cos_incl_outer = pm.Uniform("cosInclOuter", lower=-1.0, upper=0.0, testval=-0.70)
    incl_outer = pm.Deterministic("inclOuter", tt.arccos(cos_incl_outer))
    e_outer = pm.Uniform("eOuter", lower=0.0, upper=1.0, testval=0.1)
    gamma_outer = pm.Uniform(
        "gammaOuter", lower=0, upper=15, testval=10.0
    )  # km/s on CfA RV scale

    # We would like to use MB as a parameter, but we have already defined Mtot via the
    # semi-major axis and period, and we have already defined MA from the inner orbit.
    # Mtot = pm.Deterministic("Mtot", calc_Mtot(a_outer, P_outer))
    # MB = pm.Deterministic("MB", Mtot - MA) # solar masses
    # we can put a fairly strong prior on MB from the spectral type, too.
    PosNormal = pm.Bound(pm.Normal, lower=0.0)
    MB = PosNormal("MB", mu=0.3, sd=0.5, testval=0.3)
    Mtot = pm.Deterministic("Mtot", MA + MB)

    # instead, MB becomes a parameter, and then a_outer is calculated from Mtot and P_outer
    a_outer = pm.Deterministic("aOuter", calc_a(Mtot, P_outer))
    a_ang_outer = pm.Deterministic("aAngOuter", a_outer * parallax)

    orbit_outer = xo.orbits.KeplerianOrbit(
        a=a_outer * au_to_R_sun,
        period=P_outer,
        ecc=e_outer,
        t_periastron=t_periastron_outer,
        omega=omega_outer,
        Omega=Omega_outer,
        incl=incl_outer,
        m_planet=MB,
    )

    # parameters to shift the RV scale onto the CfA scale
    offset_keck = pm.Normal("offsetKeck", mu=0.0, sd=5.0, testval=-1.244)  # km/s
    offset_feros = pm.Normal("offsetFeros", mu=0.0, sd=5.0, testval=1.256)  # km/s
    offset_dupont = pm.Normal("offsetDupont", mu=0.0, sd=5.0, testval=-0.173)  # km/s

    def get_gamma_A(t):
        """
        Helper function to determine gamma of the inner orbit.

        Args:
            t: time to query

        Returns:
            gamma_A (t) : [km/s]
        """

        gamma_A = conv * orbit_outer.get_star_velocity(t)[2]

        return gamma_A

    # get the radial velocity predictions for RV B
    rv3_keck = (
        conv * orbit_outer.get_planet_velocity(d.keck3[0])[2]
        + gamma_outer
        + offset_keck
    )

    # Boolean variable to check whether the outer orbit increases in velocity
    # over the observational baseline
    pm.Deterministic("increasing", rv3_keck[-1] > rv3_keck[0])

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
        # get_star_velocity and get_planet_velocity return (v_x, v_y, v_z) tuples,
        # so we only need the v_z vector
        # but, note that since +Z points towards the observer,
        # we actually want v_radial = -v_Z (see conv)
        rv1 = (
            conv * orbit_inner.get_star_velocity(t1)[2]
            + gamma_outer
            + get_gamma_A(t1)
            + offset
        )
        rv2 = (
            conv * orbit_inner.get_planet_velocity(t2)[2]
            + gamma_outer
            + get_gamma_A(t2)
            + offset
        )

        return (rv1, rv2)

    # dates are the first entry in each tuple of (date, rv, err)
    rv1_cfa, rv2_cfa = get_RVs(d.cfa1[0], d.cfa2[0], 0.0)
    rv1_keck, rv2_keck = get_RVs(d.keck1[0], d.keck2[0], offset_keck)
    rv1_feros, rv2_feros = get_RVs(d.feros1[0], d.feros2[0], offset_feros)
    rv1_dupont, rv2_dupont = get_RVs(d.dupont1[0], d.dupont2[0], offset_dupont)

    logjit_cfa = pm.Uniform("logjittercfa", lower=-5.0, upper=np.log(5), testval=1.3)
    logjit_keck = pm.Uniform("logjitterkeck", lower=-5.0, upper=np.log(5), testval=-0.1)
    logjit_feros = pm.Uniform(
        "logjitterferos", lower=-5.0, upper=np.log(5), testval=1.29
    )
    logjit_dupont = pm.Uniform(
        "logjitterdupont", lower=-5.0, upper=np.log(5), testval=0.71
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
        "keckRV3Obs",
        mu=rv3_keck,
        observed=d.keck3[1],
        sd=get_err(d.keck3[2], logjit_keck),
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
    # since there is only one Anthonioz measurement, we won't use jitter
    rho_inner, theta_inner = orbit_inner.get_relative_angles(
        d.anthonioz[0], parallax
    )  # arcsec

    # astrometric likelihood functions
    pm.Normal("obsRhoInner", mu=rho_inner, observed=d.anthonioz[1], sd=d.anthonioz[2])
    # wrap-safe
    theta_diff_inner = tt.arctan2(
        tt.sin(theta_inner - d.anthonioz[3]), tt.cos(theta_inner - d.anthonioz[3])
    )
    pm.Normal("obsThetaInner", mu=theta_diff_inner, observed=0.0, sd=d.anthonioz[4])

    rho_outer, theta_outer = orbit_outer.get_relative_angles(
        d.wds[0], parallax
    )  # arcsec

    # add jitter terms to both separation and position angle
    log_rho_s = pm.Normal(
        "logRhoS", mu=np.log(np.median(d.wds[2])), sd=2.0, testval=-4.8
    )
    log_theta_s = pm.Normal(
        "logThetaS", mu=np.log(np.median(d.wds[4])), sd=2.0, testval=-3.7
    )

    rho_tot_err = tt.sqrt(d.wds[2] ** 2 + tt.exp(2 * log_rho_s))
    theta_tot_err = tt.sqrt(d.wds[4] ** 2 + tt.exp(2 * log_theta_s))

    pm.Normal("obsRhoOuter", mu=rho_outer, observed=d.wds[1], sd=rho_tot_err)
    theta_diff_outer = tt.arctan2(
        tt.sin(theta_outer - d.wds[3]), tt.cos(theta_outer - d.wds[3])
    )
    pm.Normal("obsThetaOuter", mu=theta_diff_outer, observed=zeros, sd=theta_tot_err)

    # disk dynamical likelihoods

    # generate i_disk from the range of samples
    i_disk = pm.Uniform(
        "iDisk",
        lower=np.min(incl_samples),
        upper=np.max(incl_samples),
        testval=np.mean(incl_samples),
    )

    # generate Omega_disk from range of samples
    Omega_disk = pm.Uniform(
        "OmegaDisk",
        lower=np.min(Omega_samples),
        upper=np.max(Omega_samples),
        testval=np.mean(Omega_samples),
    )

    disk_observed = tt.as_tensor_variable([MA, i_disk, Omega_disk])

    pm.MvNormal("obs_disk", mu=disk_mu, cov=disk_cov, observed=disk_observed)

    # calculate the mutual inclination angles

    # between the inner binary and disk
    theta_disk_inner = pm.Deterministic(
        "thetaDiskInner",
        tt.arccos(
            tt.cos(i_disk) * tt.cos(incl_inner)
            + tt.sin(i_disk) * tt.sin(incl_inner) * tt.cos(Omega_disk - Omega_inner)
        ),
    )

    # between the inner binary and outer binary
    theta_inner_outer = pm.Deterministic(
        "thetaInnerOuter",
        tt.arccos(
            tt.cos(incl_inner) * tt.cos(incl_outer)
            + tt.sin(incl_inner)
            * tt.sin(incl_outer)
            * tt.cos(Omega_inner - Omega_outer)
        ),
    )

    # between the disk and outer binary
    theta_disk_outer = pm.Deterministic(
        "thetaDiskOuter",
        tt.arccos(
            tt.cos(i_disk) * tt.cos(incl_outer)
            + tt.sin(i_disk) * tt.sin(incl_outer) * tt.cos(Omega_disk - Omega_outer)
        ),
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
    if ("_interval__" not in var.name)
    and ("_angle__" not in var.name)
    and ("_lowerbound__" not in var.name)
    and ("_log__" not in var.name)
    and (var.name != "increasing")  # arviz doesn't play well with booleans
]

