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


fig, ax = plt.subplots(nrows=1)

with model:
    rho = xo.eval_in_model(rho_model, map_sol3)
    theta = xo.eval_in_model(theta_model, map_sol3)


    # plot the data
    xs = rho_data * np.cos(theta_data) # X is north
    ys = rho_data * np.sin(theta_data) # Y is east
    ax.plot(ys, xs, "ko")


    # plot the orbit
    xs = rho * np.cos(theta) # X is north
    ys = rho * np.sin(theta) # Y is east
    ax.plot(ys, xs, ".")


    ax.set_ylabel(r"$\Delta \delta$ ['']")
    ax.set_xlabel(r"$\Delta \alpha \cos \delta$ ['']")
    ax.invert_xaxis()
    ax.plot(0,0, "k*")
    ax.set_aspect("equal", "datalim")



mkw = {"marker":".", "color":"C1"}

# pos = model.test_point
pos = map_sol3

with model:

    fig, ax = plt.subplots(nrows=4, sharex=True)

    ax[0].plot(jds, rho_data, "ko")
    ax[0].plot(jds, xo.eval_in_model(rho_model, pos), **mkw)
    ax[0].errorbar(jds, rho_data, yerr=rho_err, ls="", **mkw)
    ax[0].set_ylabel(r'$\rho\,$ ["]')

    ax[1].plot(jds, theta_data, "ko")
    ax[1].plot(jds, xo.eval_in_model(theta_model, pos), **mkw)
    ax[1].errorbar(jds, theta_data, yerr=theta_err, ls="", **mkw)
    ax[1].set_ylabel(r'P.A. [radians]')

    model_A = xo.eval_in_model(rvA, pos)
    ax[2].axhline(pos["gammaKeck"])
    ax[2].plot(keck_jd, keck_A_RV, "ko")
    ax[2].plot(map_sol3["tDense"], map_sol3["RV1Dense"], color="C0")
    ax[2].plot(keck_jd, model_A, **mkw)


    model_B = xo.eval_in_model(rvB, pos)
    ax[3].axhline(pos["gammaKeck"])
    ax[3].plot(keck_jd, keck_B_RV, "ko")
    ax[3].plot(map_sol3["tDense"], map_sol3["RV2Dense"], color="C0")
    ax[3].plot(keck_B[0], model_B, **mkw)
    print(model_A, model_B)



pm.summary(trace)

pm.traceplot(trace, varnames=["a_ang", "logP", "omega", "Omega", "e", "incl", "phi"])




samples = pm.trace_to_dataframe(trace, varnames=["a_ang", "P", "omega", "Omega", "e",
                                        "incl", "phi", "MA", "MB", "gammaKeck"])

samples["P"] /= yr
samples["omega"] /= deg
samples["Omega"] /= deg
samples["incl"] /= deg
f = 0.999
# corner.corner(samples, range=[f, f, f, [100, 150], f, f, f, [0, 3], [0, 3], f]);
corner.corner(samples, range=[f, f, f, f, f, f, f, [0, 3], [0, 3], f]);


# In[153]:


# plot the orbits on the figure

# we can plot the maximum posterior solution to see

pkw = {'marker':".", "color":"k", 'ls':""}
ekw = {'color':"C1", 'ls':""}

fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(6,8))
ax[0].set_ylabel(r'$\rho\,$ ["]')
ax[1].set_ylabel(r'P.A. [radians]')
ax[2].set_ylabel(r'$v_\mathrm{A}$')
ax[3].set_ylabel(r'$v_\mathrm{B}$')
ax[3].set_xlabel("JD [days]")

nsamples = 50
choices = np.random.choice(np.arange(len(trace)), size=nsamples)

# get map sol for tot_rho_err

tot_rho_err = np.sqrt(rho_err**2 + np.exp(2 * np.median(trace["logRhoS"])))
tot_theta_err = np.sqrt(theta_err**2 + np.exp(2 * np.median(trace["logThetaS"])))


fig_sky, ax_sky = plt.subplots(nrows=1, figsize=(4,4))
fig_phase, ax_phase = plt.subplots(nrows=3, figsize=(5,6))




with model:
    # iterate through trace object
    for i in choices:

        pos = trace[i]

        # choose the color based upon Omega family
        if pos["Omega"] > 50 * deg:
            color = "C0"
        else:
            color = "C1"


        t_pred = pos["tPeriod"]
        rho_pred = pos["rhoSaveSky"]
        theta_pred = pos["thetaSaveSky"]

        x_pred = rho_pred * np.cos(theta_pred) # X north
        y_pred = rho_pred * np.sin(theta_pred) # Y east

        ax[0].plot(jd0 + t_data, pos["rhoSaveData"], color=color, lw=0.8, alpha=0.7, zorder=0)
        ax[1].plot(jd0 + t_data, pos["thetaSaveData"], color=color, lw=0.8, alpha=0.7, zorder=0)

        ax[2].plot(pos["tDense"], pos["RV1Dense"], color=color, lw=0.5, alpha=0.6)
        ax[2].set_ylim(7, 10)

        ax[3].plot(pos["tDense"], pos["RV2Dense"], color=color, lw=0.5, alpha=0.6)
        ax[3].set_xlim(np.min(jds) - 200, np.max(jds) + 200)
        ax[3].set_ylim(6.5, 8)


        ax_phase[0].plot(pos["tDense"], pos["RV1Dense"], color=color, lw=0.5, alpha=0.6)
        ax_phase[1].plot(pos["tDense"], pos["RV2Dense"], color=color, lw=0.5, alpha=0.6)
        diff = pos["RV1Dense"] - pos["RV2Dense"]
        max_amp = np.max(np.abs(diff))
        ax_phase[2].axvline(keck_jd, color="0.8")
        ax_phase[2].plot(pos["tDense"], diff / max_amp, lw=0.8, color=color, alpha=0.8)


        ax_sky.plot(y_pred, x_pred, color=color, lw=0.8, alpha=0.7)



ax[0].plot(jds, rho_data, **pkw, zorder=20)
ax[0].errorbar(jds, rho_data, yerr=tot_rho_err, **ekw, zorder=19)

ax[1].plot(jds, theta_data, **pkw, zorder=20)
ax[1].errorbar(jds, theta_data, yerr=tot_theta_err, **ekw, zorder=19)


ax_sky.plot(ys, xs, "ko")
ax_sky.set_ylabel(r"$\Delta \delta$ ['']")
ax_sky.set_xlabel(r"$\Delta \alpha \cos \delta$ ['']")
ax_sky.invert_xaxis()
ax_sky.plot(0,0, "k*")
ax_sky.set_aspect("equal", "datalim")

pm.backends.ndarray.save_trace(trace, directory="may17", overwrite=True)
