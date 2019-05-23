import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams["savefig.dpi"] = 100
rcParams["figure.dpi"] = 100
rcParams["font.size"] = 16
rcParams["text.usetex"] = False
rcParams["font.family"] = ["sans-serif"]
rcParams["font.sans-serif"] = ["cmss10"]
rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.io import ascii
from astropy import units as u
from astropy import constants


# import the relevant packages
import pymc3 as pm
import theano.tensor as tt

import corner  # https://corner.readthedocs.io
import exoplanet as xo
import exoplanet.orbits
from exoplanet.distributions import Angle


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

with pm.Model() as parallax_model:

    # 27.31 mas +/- 0.12 mas # from GAIA
    parallax = pm.Normal("parallax", mu=27.31, sd=0.12) # milliarcsec
    a_ang = pm.Uniform("a_ang", 0.1, 4.0, testval=2.0) # arcsec

    # the distance to the source in parcsecs
    dpc = pm.Deterministic("dpc", 1e3 / parallax)

    # the semi-major axis in au
    a = pm.Deterministic("a", a_ang * dpc) # au

    # we expect the period to be somewhere in the range of 25 years,
    # so we'll set a broad prior on logP
    logP = pm.Normal("logP", mu=np.log(400), sd=1.0)
    P = pm.Deterministic("P", tt.exp(logP) * yr) # days

    omega = Angle("omega", testval=180 * deg) # - pi to pi
    Omega = Angle("Omega", testval=0 * deg) # - pi to pi

    phi = Angle("phi", testval=2.)

    n = 2*np.pi*tt.exp(-logP) / yr

    t_periastron = (phi + omega) / n

    cos_incl = pm.Uniform("cosIncl", lower=-1., upper=1.0, testval=np.cos(3.0))
    incl = pm.Deterministic("incl", tt.arccos(cos_incl))
    e = pm.Uniform("e", lower=0.0, upper=1.0, testval=0.3)

    # n.b. that we include an extra conversion for a, because exoplanet expects a in R_sun
    orbit = xo.orbits.KeplerianOrbit(a=a * au_to_R_sun, t_periastron=t_periastron, period=P,
                                   incl=incl, ecc=e, omega=omega, Omega=Omega)

    # now that we have a physical scale defined, we can also calculate the total mass of the system
    Mtot = pm.Deterministic("Mtot", orbit.m_total)
    Mtot_kepler = pm.Deterministic("MtotKepler", calc_Mtot(a, P))

    rho_phys, theta_model = orbit.get_relative_angles(jds - jd0) # the rho, theta model values

    # because we've specified a physical value for a, a is now actually in units of R_sun
    # So, we'll want to convert back to arcsecs
    rho_model = (rho_phys / au_to_R_sun) / dpc # arcsec

    # add jitter terms to both separation and position angle
    log_rho_s = pm.Normal("logRhoS", mu=np.log(np.median(rho_err)), sd=2.0)
    log_theta_s = pm.Normal("logThetaS", mu=np.log(np.median(theta_err)), sd=2.0)

    rho_tot_err = tt.sqrt(rho_err**2 + tt.exp(2*log_rho_s))
    theta_tot_err = tt.sqrt(theta_err**2 + tt.exp(2*log_theta_s))

    pm.Normal("obs_rho", mu=rho_model, observed=rho_data, sd=rho_tot_err)

    # we need to calculate the difference
    theta_diff = tt.arctan2(tt.sin(theta_model - theta_data), tt.cos(theta_model - theta_data))
    pm.Normal("obs_theta", mu=theta_diff, observed=zeros, sd=theta_tot_err)

    t_period = pm.Deterministic("tPeriod", t_fine * P)

    # save some samples on a fine orbit for plotting purposes
    rho, theta = orbit.get_relative_angles(t_period)
    rho_save_sky = pm.Deterministic("rhoSaveSky", rho / au_to_R_sun / dpc)
    theta_save_sky = pm.Deterministic("thetaSaveSky", theta)

    rho, theta = orbit.get_relative_angles(t_data)
    rho_save_data = pm.Deterministic("rhoSaveData", rho / au_to_R_sun / dpc)
    theta_save_data = pm.Deterministic("thetaSaveData", theta)


with parallax_model:
    trace = pm.backends.ndarray.load_trace("current")


sample_vars = ["parallax", "a_ang", "logP", "omega", "Omega", "e", "cosIncl", "phi", "logRhoS", "logThetaS"]
print(pm.summary(trace, varnames=sample_vars))

trace_plot = pm.traceplot(trace, varnames=sample_vars)
plt.savefig("plots/traceplot.pdf")


samples = pm.trace_to_dataframe(trace, varnames=sample_vars)

fig = corner.corner(samples)
fig.savefig("plots/corner_sample_vars.png")



samples = pm.trace_to_dataframe(trace, varnames=["Mtot", "MtotKepler", "a", "P", "omega", "Omega", "e", "incl", "phi"])
samples["P"] /= yr
fig = corner.corner(samples, range=[[0, 4], [0, 4], 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99])
fig.savefig("plots/corner_present_vars.png")

# we can plot the maximum posterior solution to see
pkw = {'marker':".", "color":"k", 'ls':""}
ekw = {'color':"C1", 'ls':""}

fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(6,8))
ax[0].set_ylabel(r'$\rho\,$ ["]')
ax[1].set_ylabel(r'$\rho$ residuals')
ax[2].set_ylabel(r'P.A. [radians]')
ax[3].set_ylabel(r'P.A. residuals')

nsamples = 50
choices = np.random.choice(len(trace), size=nsamples)

# get map sol for tot_rho_err

tot_rho_err = np.sqrt(rho_err**2 + np.exp(2 * np.median(trace["logRhoS"])))
tot_theta_err = np.sqrt(theta_err**2 + np.exp(2 * np.median(trace["logThetaS"])))

fig_sky, ax_sky = plt.subplots(nrows=1, figsize=(4,4))
ax[0].set_ylabel(r'$\rho\,$ ["]')
ax[1].set_ylabel(r'$\rho$ residuals')
ax[2].set_ylabel(r'P.A. [radians]')
ax[3].set_ylabel(r'P.A. residuals')
ax[3].set_xlabel("JD [days]")

with parallax_model:
    # iterate through trace object
    for i in choices:

        pos = trace[i]

        t_pred = pos["tPeriod"]
        rho_pred = pos["rhoSaveSky"]
        theta_pred = pos["thetaSaveSky"]

        x_pred = rho_pred * np.cos(theta_pred) # X north
        y_pred = rho_pred * np.sin(theta_pred) # Y east

        ax[0].plot(jd0 + t_data, pos["rhoSaveData"], color="C0", lw=0.8, alpha=0.7)

        ax[1].plot(jds, rho_data - xo.eval_in_model(rho_model, pos), **pkw, alpha=0.4)

        ax[2].plot(jd0 + t_data, pos["thetaSaveData"], color="C0", lw=0.8, alpha=0.7)

        ax[3].plot(jds, theta_data - xo.eval_in_model(theta_model, pos), **pkw, alpha=0.4)

        ax_sky.plot(y_pred, x_pred, color="C0", lw=0.8, alpha=0.7)


ax[0].plot(jds, rho_data, **pkw)
ax[0].errorbar(jds, rho_data, yerr=tot_rho_err, **ekw)

ax[1].axhline(0.0, color="0.5")
ax[1].errorbar(jds, np.zeros_like(jds), yerr=tot_rho_err, **ekw)

ax[2].plot(jds, theta_data, **pkw)
ax[2].errorbar(jds, theta_data, yerr=tot_theta_err, **ekw)

ax[3].axhline(0.0, color="0.5")
ax[3].errorbar(jds, np.zeros_like(jds), yerr=tot_theta_err, **ekw)
fig.savefig("plots/posterior_rho_sep.pdf")


xs = rho_data * np.cos(theta_data) # X is north
ys = rho_data * np.sin(theta_data) # Y is east

ax_sky.plot(ys, xs, "ko")
ax_sky.set_ylabel(r"$\Delta \delta$ ['']")
ax_sky.set_xlabel(r"$\Delta \alpha \cos \delta$ ['']")
ax_sky.invert_xaxis()
ax_sky.plot(0,0, "k*")
ax_sky.set_aspect("equal", "datalim")
fig_sky.subplots_adjust(left=0.18, right=0.82)
fig_sky.savefig("plots/posterior_sky.pdf")
