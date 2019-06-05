import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy
from astropy.time import Time
from astropy.io import ascii
from astropy import units as u
from astropy import constants

# import the relevant packages
import pymc3 as pm
import theano.tensor as tt

import exoplanet as xo
import exoplanet.orbits
from exoplanet.distributions import Angle
import corner  # https://corner.readthedocs.io


deg = np.pi/180. # radians / degree
yr = 365.25 # days / year
au_to_R_sun = (constants.au / constants.R_sun).value # conversion constant

# grab the formatted data
data = ascii.read("data/visual_data_besselian.csv", format="csv", fill_values=[("X", '0')])

# convert years
jds = Time(np.ascontiguousarray(data["epoch"]), format="byear").jd

data["rho_err"][data["rho_err"].mask == True] = 0.05
data["PA_err"][data["PA_err"].mask == True] = 5.0

# convert all masked frames to be raw np arrays, since theano has issues with astropy masked columns
rho_data = np.ascontiguousarray(data["rho"], dtype=float) # arcsec
rho_err = np.ascontiguousarray(data["rho_err"], dtype=float)

# the position angle measurements come in degrees in the range [0, 360].
# we need to convert this to radians in the range [-pi, pi]
theta_data = np.ascontiguousarray(data["PA"] * deg, dtype=float)
theta_data[theta_data > np.pi] -= 2 * np.pi

theta_err = np.ascontiguousarray(data["PA_err"] * deg) # radians

# Load the Keck radial velocity data for A (gamma) and B.
#
# We also need to make sure that we put the measurements onto the same scale, which, for now, is the CfA scale. So, we'll want to use the offset parameter we found from the spectroscopic binary fit to offset these to the same scale, or, ultimately use the ALMA scale and offset everything to that. In the hierarchical triple fit we'll want to keep everything consistent in the corrections.
#
# For this fit, we're just going to use everything on the Keck scale.
# We'll assume that the $\gamma$ velocity for A is the same at the average of the Keck epochs. Ultimately, $\gamma_A$ will become a function of $t$.


def get_arrays(asciiTable, errDict=None, jitter=False):
    """
    Reformat ascii tables into pure numpy arrays of the right dimension.
    """

    output = []

    star = "B"

    # get the RVs
    rv = asciiTable["RV_" + star]

    if type(rv) is astropy.table.column.MaskedColumn:
        mask = ~rv.mask # values we want to keep when indexing
    else:
        mask = np.ones(len(rv), dtype="bool")

    rv = np.ascontiguousarray(rv[mask])
    date = np.ascontiguousarray(asciiTable["HJD"][mask]) + 2400000

    if errDict is None:
        err = np.ascontiguousarray(asciiTable["sigma_" + star][mask])
    else:
        err = np.ones(len(date), dtype=np.float64) * errDict[star]

    if jitter:
        err = np.ones(len(date), dtype=np.float64) * 0.4 # [km/s] assume a small error, since we'll infer.

    assert len(date) == len(rv), "date - rv length mismatch"
    assert len(date) == len(err), "date - err length mismatch"

    return (date, rv, err)


# Do this to infer w/ jitter
jitter=True

data_keck = ascii.read("data/keck.dat", format="tab", fill_values=[("X", 0)])
err_keck = {"Aa":0.63, "Ab":0.85, "B":0.59} # km/s
keck_B = get_arrays(data_keck, err_keck, jitter=jitter)

# date is HJD + 2400000

keck_jd = np.average(keck_B[0])
keck_A_RV = 8.63 # [km/s] for now, use gamma from Kellogg solutions
keck_B_RV = np.average(keck_B[1])
keck_err = 0.4 # [km/s] guesstimate from average


def calc_Mtot(a, P):
    '''
    Calculate the total mass of the system using Kepler's third law.

    Args:
        a (au) semi-major axis
        P (days) period

    Returns:
        Mtot (M_sun) total mass of system (M_primary + M_secondary)
    '''

    day_to_s = (1 * u.day).to(u.s).value
    au_to_m = (1 * u.au).to(u.m).value
    kg_to_M_sun = (1 * u.kg).to(u.M_sun).value

    return 4 * np.pi**2 * (a * au_to_m)**3 / (constants.G.value * (P * day_to_s)**2) * kg_to_M_sun


# Set up the model in PyMC3
zeros = np.zeros_like(jds)
jd0 = jds[0]
t_fine = np.linspace(0, 1, num=500)
t_data = np.linspace(-yr, jds[-1] - jd0 + yr)

# convert from R_sun / day to km/s
# and from v_r = - v_Z
output_units = u.km / u.s
conv = -(1 * u.R_sun / u.day).to(output_units).value

xs_phase = np.linspace(0, 1, num=500)

with pm.Model() as model:

    # We'll include the parallax data as a prior on the parallax value

    # 27.31 mas +/- 0.12 mas # from GAIA
    mparallax = pm.Normal("mparallax", mu=27.31, sd=0.12) # milliarcsec
    parallax = pm.Deterministic("parallax", 1e-3 * mparallax) # arcsec
    a_ang = pm.Uniform("a_ang", 0.1, 4.0, testval=2.0) # arcsec

    # the semi-major axis in au
    a = pm.Deterministic("a", a_ang / parallax) # au

    # we expect the period to be somewhere in the range of 25 years,
    # so we'll set a broad prior on logP
    logP = pm.Normal("logP", mu=np.log(400), sd=1.0)
    P = pm.Deterministic("P", tt.exp(logP) * yr) # days

    omega = Angle("omega", testval=1.8) # - pi to pi
    Omega = Angle("Omega", testval=-0.7) # - pi to pi

    phi = Angle("phi", testval=2.)

    n = 2*np.pi*tt.exp(-logP) / yr

    t_periastron = (phi + omega) / n

    cos_incl = pm.Uniform("cosIncl", lower=-1., upper=1.0, testval=np.cos(3.0))
    incl = pm.Deterministic("incl", tt.arccos(cos_incl))
    e = pm.Uniform("e", lower=0.0, upper=1.0, testval=0.3)
    gamma_keck = pm.Uniform("gammaKeck", lower=5, upper=10, testval=7.5) # km/s on Keck RV scale

    MB = pm.Normal("MB", mu=0.3, sd=0.5) # solar masses

    orbit = xo.orbits.KeplerianOrbit(a=a*au_to_R_sun, period=P, ecc=e, t_periastron=t_periastron,
                                     omega=omega, Omega=Omega, incl=incl, m_planet=MB)

    # now that we have a physical scale defined, the total mass of the system makes sense
    Mtot = pm.Deterministic("Mtot", orbit.m_total)
    MA = pm.Deterministic("MA", Mtot - MB)

    rho_model, theta_model = orbit.get_relative_angles(jds - jd0, parallax) # the rho, theta model values

    # add jitter terms to both separation and position angle
    log_rho_s = pm.Normal("logRhoS", mu=np.log(np.median(rho_err)), sd=2.0)
    log_theta_s = pm.Normal("logThetaS", mu=np.log(np.median(theta_err)), sd=2.0)

    rho_tot_err = tt.sqrt(rho_err**2 + tt.exp(2*log_rho_s))
    theta_tot_err = tt.sqrt(theta_err**2 + tt.exp(2*log_theta_s))

    pm.Normal("obs_rho", mu=rho_model, observed=rho_data, sd=rho_tot_err)

    # we need to calculate the difference
    theta_diff = tt.arctan2(tt.sin(theta_model - theta_data), tt.cos(theta_model - theta_data))
    pm.Normal("obs_theta", mu=theta_diff, observed=zeros, sd=theta_tot_err)

    # get the radial velocity predictions for primary and secondary
    rvA = conv * orbit.get_star_velocity(keck_jd - jd0)[2] + gamma_keck
    rvB = conv * orbit.get_planet_velocity(keck_B[0] - jd0)[2] + gamma_keck

    # evaluate the RV likelihood functions
    pm.Normal("obs_A", mu=rvA, observed=keck_A_RV, sd=keck_err)
    pm.Normal("obs_B", mu=rvB, observed=keck_B[1], sd=keck_B[2])


    # save some samples on a fine orbit for plotting purposes
    t_period = pm.Deterministic("tPeriod", t_fine * P)

    rho, theta = orbit.get_relative_angles(t_period, parallax)
    rho_save_sky = pm.Deterministic("rhoSaveSky", rho)
    theta_save_sky = pm.Deterministic("thetaSaveSky", theta)

    rho, theta = orbit.get_relative_angles(t_data, parallax)
    rho_save_data = pm.Deterministic("rhoSaveData", rho)
    theta_save_data = pm.Deterministic("thetaSaveData", theta)


    # save RV plots
    t_dense = pm.Deterministic("tDense", xs_phase * P + jd0)
    rv1_dense = pm.Deterministic("RV1Dense", conv * orbit.get_star_velocity(t_dense - jd0)[2] + gamma_keck)
    rv2_dense = pm.Deterministic("RV2Dense", conv * orbit.get_planet_velocity(t_dense - jd0)[2] + gamma_keck)


with model:
    map_sol0 = xo.optimize(vars=[a_ang, phi])
    map_sol1 = xo.optimize(map_sol0, vars=[a_ang, phi, omega, Omega])
    map_sol2 = xo.optimize(map_sol1, vars=[a_ang, logP, phi, omega, Omega, incl, e])
    map_sol3 = xo.optimize(map_sol2)


# now let's actually explore the posterior for real
sampler = xo.PyMC3Sampler(finish=500, chains=4)
with model:
    burnin = sampler.tune(tune=2000, step_kwargs=dict(target_accept=0.9))
    trace = sampler.sample(draws=3000)

pm.backends.ndarray.save_trace(trace, directory="current", overwrite=True)
