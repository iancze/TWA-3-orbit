import matplotlib.pyplot as plt
import exoplanet as xo
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np
from . import data as d
import theano.tensor as tt
from .gas_vel import plot_gas
from astropy.time import Time
from .constants import *


def plot_sep_pa(trace, m):
    # plot separation and PA across the observed dates
    ts_obs = np.linspace(
        np.min(d.wds[0]) - 300, np.max(d.wds[0]) + 300, num=1000
    )  # days

    with m.model:
        predict_fine = m.orbit.get_relative_angles(ts_obs, m.parallax)

    # we can plot the maximum posterior solution to see
    # pkw = {'marker':".", "color":"k", 'ls':""}
    ekw = {"color": "C1", "marker": "o", "ls": ""}

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(6, 4))
    ax[0].set_ylabel(r'$\rho\,$ ["]')
    ax[1].set_ylabel(r"P.A. [radians]")
    ax[1].set_xlabel("JD [days]")

    for sample in xo.get_samples_from_trace(trace, size=20):

        # we'll want to cache these functions when we evaluate many samples
        rho_fine, theta_fine = xo.eval_in_model(
            predict_fine, point=sample, model=m.model
        )

        ax[0].plot(ts_obs, rho_fine, "C0")
        ax[1].plot(ts_obs, theta_fine, "C0")

    # get map sol for tot_rho_err
    tot_rho_err = np.sqrt(d.wds[2] ** 2 + np.exp(2 * np.median(trace["logRhoS"])))
    tot_theta_err = np.sqrt(d.wds[4] ** 2 + np.exp(2 * np.median(trace["logThetaS"])))

    # ax[0].plot(d.wds[0], d.wds[1], **pkw)
    ax[0].errorbar(d.wds[0], d.wds[1], yerr=tot_rho_err, **ekw)

    # ax[1].plot(jds, theta_data, **pkw)
    ax[1].errorbar(d.wds[0], d.wds[3], yerr=tot_theta_err, **ekw)


def plot_sky(trace, m):
    # plot sky position for a full orbit
    xs_phase = np.linspace(0, 1, num=1000)

    with m.model:
        ts_full = xs_phase * m.P + m.t_periastron
        predict_full = m.orbit.get_relative_angles(ts_full, m.parallax)

    fig_sky, ax_sky = plt.subplots(nrows=1, figsize=(4, 4))

    for sample in xo.get_samples_from_trace(trace, size=20):

        # we'll want to cache these functions when we evaluate many samples
        rho_full, theta_full = xo.eval_in_model(
            predict_full, point=sample, model=m.model
        )

        x_full = rho_full * np.cos(theta_full)  # X North
        y_full = rho_full * np.sin(theta_full)
        ax_sky.plot(y_full, x_full, color="C0", lw=0.8, alpha=0.7)

    xs = d.wds[1] * np.cos(d.wds[3])  # X is north
    ys = d.wds[1] * np.sin(d.wds[3])  # Y is east

    ax_sky.plot(ys, xs, "ko")
    ax_sky.set_ylabel(r"$\Delta \delta$ ['']")
    ax_sky.set_xlabel(r"$\Delta \alpha \cos \delta$ ['']")
    ax_sky.invert_xaxis()
    ax_sky.plot(0, 0, "k*")
    ax_sky.set_aspect("equal", "datalim")
    fig_sky.subplots_adjust(left=0.18, right=0.82)

    return fig
