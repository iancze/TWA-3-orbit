import arviz as az
import corner
import exoplanet as xo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from astropy.time import Time
import os

from . import data as d
from .constants import *
from .gas_vel import plot_gas
from .plot_utils import efficient_autocorr, efficient_trace


def plot_summaries(trace, m, plotdir):
    ar_data = az.from_pymc3(trace=trace)
    # view summary
    df = az.summary(ar_data, var_names=m.all_vars)
    print(df)

    # write summary to disk
    f = open(plotdir / "summary.txt", "w")
    df.to_string(f)
    f.close()

    with az.rc_context(rc={"plot.max_subplots": 80}):
        stem = str(plotdir / "autocorr{:}.png")
        efficient_autocorr(ar_data, var_names=m.all_vars, figstem=stem)

        # make a traceplot
        stem = str(plotdir / "trace{:}.png")
        efficient_trace(ar_data, var_names=m.all_vars, figstem=stem)


def plot_triangles(df, plotdir):

    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

    df = df.copy()

    # make a nice corner plot of the variables we care about

    # convert all params
    df["omegaInner"] /= deg
    df["OmegaInner"] /= deg
    df["inclInner"] /= deg

    df["omegaOuter"] /= deg
    df["OmegaOuter"] /= deg
    df["inclOuter"] /= deg

    df["thetaDiskInner"] /= deg
    df["thetaInnerOuter"] /= deg
    df["thetaDiskOuter"] /= deg

    df["POuter"] /= yr

    # just the inner parameters
    inner = [
        "MAb",
        "MA",
        "aInner",
        "PInner",
        "eInner",
        "omegaInner",
        "OmegaInner",
        "inclInner",
        "tPeriastronInner",
    ]
    fig = corner.corner(df[inner])
    fig.savefig(plotdir / "corner-inner.png", dpi=120)

    # posterior on periastron passage
    # r_p = a * (1 - e)
    df["rp"] = df["aOuter"] * (1 - df["eOuter"])  # au

    # just the outer parameters
    outer = [
        "MA",
        "MB",
        "aOuter",
        "POuter",
        "omegaOuter",
        "OmegaOuter",
        "eOuter",
        "inclOuter",
        "gammaOuter",
        "tPeriastronOuter",
        "rp",
    ]
    fig = corner.corner(df[outer])
    fig.savefig(plotdir / "corner-outer.png", dpi=120)

    # masses
    masses = ["MAa", "MAb", "MA", "MB", "Mtot"]
    fig = corner.corner(df[masses])
    fig.savefig(plotdir / "corner-masses.png", dpi=120)

    # mutual inclination between inner orbit and outer orbit
    muts = ["thetaDiskInner", "thetaInnerOuter", "thetaDiskOuter"]
    fig = corner.corner(df[muts])
    fig.savefig(plotdir / "corner-muts.png", dpi=120)


def plot_interior_RV(trace, m):
    """
    plot the interior RV

    Args: 
        trace : pymc3 trace
        m : the reference to the model module
    """

    # choose the plot styles
    color_dict = {"CfA": "C5", "Keck": "C2", "FEROS": "C3", "du Pont": "C4"}
    hkw = {"lw": 1.0, "color": "0.4", "ls": ":"}
    pkw = {"ls": "-", "lw": 0.5, "color": "C0"}
    ekw = {"marker": ".", "ms": 2.5, "ls": "", "elinewidth": 0.6, "zorder": 20}

    # set the figure dimensions

    lmargin = 0.47
    rmargin = lmargin
    bmargin = 0.4
    tmargin = 0.05
    mmargin = 0.07
    mmmargin = 0.15

    ax_height = 1.0
    ax_r_height = 0.4

    # \textwidth=7.1in
    # \columnsep=0.3125in
    # column width = (7.1 - 0.3125)/2 = 3.393

    xx = 3.393
    ax_width = xx - lmargin - rmargin

    yy = bmargin + tmargin + 2 * mmargin + mmmargin + 2 * ax_height + 2 * ax_r_height

    # fill out the axes

    fig = plt.figure(figsize=(xx, yy))
    ax1 = fig.add_axes(
        [lmargin / xx, 1 - (tmargin + ax_height) / yy, ax_width / xx, ax_height / yy]
    )
    ax1_r = fig.add_axes(
        [
            lmargin / xx,
            1 - (tmargin + mmargin + ax_height + ax_r_height) / yy,
            ax_width / xx,
            ax_r_height / yy,
        ]
    )
    ax1_r.axhline(0.0, **hkw)
    ax2 = fig.add_axes(
        [
            lmargin / xx,
            (bmargin + mmargin + ax_r_height) / yy,
            ax_width / xx,
            ax_height / yy,
        ]
    )
    ax2_r = fig.add_axes([lmargin / xx, bmargin / yy, ax_width / xx, ax_r_height / yy])
    ax2_r.axhline(0.0, **hkw)

    xs_phase = np.linspace(0, 1, num=500)

    # define new Theano variables to get the continuous model RVs and the
    # discrete models RVs to plot
    with m.model:
        # the reason why we can refer to m.P_inner here (rather than, say)
        # m.model.P_inner, is because P_inner is actually in the scope of the
        # model module. It is in the *context* of the pymc3 model.model.
        # so we need both, for it to make sense.
        ts_phases = xs_phase * m.P_inner + m.t_periastron_inner  # new theano var
        rv1, rv2 = m.get_RVs(ts_phases, ts_phases, 0.0)

        # We want to plot the model and data phased to a single reference orbit
        # this means we need to take out any additional motion to gamma_A relative
        # to the reference orbit from both the model and data
        gamma_A_epoch = m.get_gamma_A(ts_phases[0])

        err_dict = {
            "CfA": m.logjit_cfa,
            "Keck": m.logjit_keck,
            "FEROS": m.logjit_feros,
            "du Pont": m.logjit_dupont,
        }

        offset_dict = {
            "CfA": 0.0,
            "Keck": m.offset_keck,
            "FEROS": m.offset_feros,
            "du Pont": m.offset_dupont,
        }

        def get_centered_rvs(label, data1, data2):
            """
            Return the data and model discrete velocities centered on the reference 
            epoch bary velocity
            """
            t1 = tt.as_tensor_variable(data1[0])
            t2 = tt.as_tensor_variable(data2[0])

            offset = offset_dict[label]
            d1_corrected = data1[1] - offset - m.get_gamma_A(t1) + gamma_A_epoch
            d2_corrected = data2[1] - offset - m.get_gamma_A(t2) + gamma_A_epoch

            rv1_base, rv2_base = m.get_RVs(t1, t2, 0.0)
            rv1_corrected = rv1_base - m.get_gamma_A(t1) + gamma_A_epoch
            rv2_corrected = rv2_base - m.get_gamma_A(t2) + gamma_A_epoch

            e1 = np.sqrt(data1[2] ** 2 + np.exp(2 * err_dict[label]))
            e2 = np.sqrt(data2[2] ** 2 + np.exp(2 * err_dict[label]))

            return (
                t1,
                d1_corrected,
                e1,
                rv1_corrected,
                t2,
                d2_corrected,
                e2,
                rv2_corrected,
            )

        # repack to yield a set of corrected data, errors, and corrected model velocities
        rv_cfa0 = get_centered_rvs("CfA", d.cfa1, d.cfa2)
        rv_keck0 = get_centered_rvs("Keck", d.keck1, d.keck2)
        rv_feros0 = get_centered_rvs("FEROS", d.feros1, d.feros2)
        rv_dupont0 = get_centered_rvs("du Pont", d.dupont1, d.dupont2)

    # data, then phase them, then plot them.
    def phase_and_plot(rv_package, label):
        """
        Calculate inner orbit radial velocities and residuals for a given dataset and star.

        Args:
            data: tuple containing (date, rv, err)
            model: model rvs evaluated at date (with no offset)
            a: primary matplotlib axes
            a_r: residual matplotlib axes
            label: instrument that acquired data

        Returns:
            None
        """
        (
            t1,
            d1_corrected,
            e1,
            rv1_corrected,
            t2,
            d2_corrected,
            e2,
            rv2_corrected,
        ) = rv_package

        phase1 = get_phase(t1)
        phase2 = get_phase(t2)

        # evaluate the model with the current parameter settings
        color = color_dict[label]

        ax1.errorbar(phase1, d1_corrected, yerr=e1, label=label, **ekw, color=color)
        ax1_r.errorbar(
            phase1, d1_corrected - rv1_corrected, yerr=e1, **ekw, color=color
        )

        ax2.errorbar(phase2, d2_corrected, yerr=e2, label=label, **ekw, color=color)
        ax2_r.errorbar(
            phase2, d2_corrected - rv2_corrected, yerr=e2, **ekw, color=color
        )

    # plot many samples for the RV trace
    for sample in xo.get_samples_from_trace(trace, size=10):
        rv1_m = xo.eval_in_model(rv1, point=sample, model=m.model)
        rv2_m = xo.eval_in_model(rv2, point=sample, model=m.model)

        ax1.plot(xs_phase, rv1_m, **pkw)
        ax2.plot(xs_phase, rv2_m, **pkw)

    # get just a single sample to plot the data and model
    for sample in xo.get_samples_from_trace(trace, size=1):

        # create a phasing function local to these sampled values of
        # t_periastron and period
        def get_phase(dates):
            return ((dates - sample["tPeriastronInner"]) % sample["PInner"]) / sample[
                "PInner"
            ]

        # Plot the data and residuals for this sample
        phase_and_plot(xo.eval_in_model(rv_cfa0, point=sample, model=m.model), "CfA")
        phase_and_plot(xo.eval_in_model(rv_keck0, point=sample, model=m.model), "Keck")
        phase_and_plot(
            xo.eval_in_model(rv_feros0, point=sample, model=m.model), "FEROS"
        )
        phase_and_plot(
            xo.eval_in_model(rv_dupont0, point=sample, model=m.model), "du Pont"
        )

    ax1.legend(
        loc="upper left",
        fontsize="xx-small",
        labelspacing=0.5,
        handletextpad=0.2,
        borderpad=0.4,
        borderaxespad=1.0,
    )

    ax1.set_ylabel(r"$v_\mathrm{Aa}$ [$\mathrm{km s}^{-1}$]", labelpad=0)
    ax1_r.set_ylabel(r"$O-C$", labelpad=-1)

    ax2.set_ylabel(r"$v_\mathrm{Ab}$ [$\mathrm{km s}^{-1}$]", labelpad=-5)
    ax2_r.set_ylabel(r"$O-C$", labelpad=-2)
    ax2_r.set_xlabel("phase")

    ax = [ax1, ax1_r, ax2, ax2_r]
    for a in ax:
        a.set_xlim(0, 1)

    ax1.xaxis.set_ticklabels([])
    ax1_r.xaxis.set_ticklabels([])
    ax2.xaxis.set_ticklabels([])

    return fig


def jd_to_year(t):
    """
    Convert JD referenced to jd0 into a year.
    """

    return Time(t + jd0, format="jd").byear


# Custom routine for plotting polar error bars onto cartesian plane
def plot_errorbar(ax, thetas, rhos, theta_errs, rho_errs, **kwargs):
    """
    All kwargs sent to plot. Thetas are in radians.
    """

    thetas = np.atleast_1d(thetas)
    rhos = np.atleast_1d(rhos)
    theta_errs = np.atleast_1d(theta_errs)
    rho_errs = np.atleast_1d(rho_errs)

    assert len(thetas) == len(rhos) == len(theta_errs) == len(rho_errs)

    for theta, rho, theta_err, rho_err in zip(thetas, rhos, theta_errs, rho_errs):

        # In this notation, x has been flipped.
        # x = rho * np.sin(theta)
        # y = rho * np.cos(theta)

        # Calculate two lines, one for theta_err, one for rho_err

        # theta error
        # theta error needs a small correction in the rho direction in order to go right through the data point
        delta = rho * (1 / np.cos(theta_err) - 1)
        xs = (rho + delta) * np.sin(np.array([theta - theta_err, theta + theta_err]))
        ys = (rho + delta) * np.cos(np.array([theta - theta_err, theta + theta_err]))
        ax.plot(xs, ys, **kwargs)

        # rho error
        xs = np.sin(theta) * np.array([(rho - rho_err), (rho + rho_err)])
        ys = np.cos(theta) * np.array([(rho - rho_err), (rho + rho_err)])
        ax.plot(xs, ys, **kwargs)


def plot_sep_pa(trace, m):

    # Plot the astrometric fits.
    # 1 square panel on left, then 3 rows of panels on right (rho, theta, v_B)

    # set up the figure

    # plot styles
    pkw = {"marker": "o", "ms": 3, "color": "k", "ls": "", "zorder": 20}
    lkw = {"ls": "-", "lw": 0.5}
    ekw = {
        "marker": "o",
        "ms": 3,
        "ls": "",
        "elinewidth": 0.8,
        "zorder": 20,
        "color": "k",
    }
    kekw = {**ekw, "color": "maroon"}

    xx = 7.1  # [in] textwidth
    hmargin = 0.0
    tmargin = 0.45
    bmargin = 0.5
    lmargin = 0.5
    rmargin = 0.5
    mmargin = 0.75

    ax_width = (xx - lmargin - rmargin - mmargin) / 2
    cax_frac = 0.7
    cax_margin = 0.05
    cax_height = 0.08
    ax_height = ax_width
    sax_height = (ax_height - hmargin) / 3

    yy = ax_height + tmargin + bmargin

    fig = plt.figure(figsize=(xx, yy))

    ax_sky = fig.add_axes([lmargin / xx, bmargin / yy, ax_width / xx, ax_height / yy])
    ax_sky.set_xlabel(r"$\Delta \alpha \cos \delta\; [{}^{\prime\prime}]$")
    ax_sky.set_ylabel(r"$\Delta \delta\; [{}^{\prime\prime}]$")
    # ax_sky.invert_xaxis()

    cax = fig.add_axes(
        [
            (lmargin + (1 - cax_frac) * ax_width / 2) / xx,
            (bmargin + ax_height + cax_margin) / yy,
            cax_frac * ax_width / xx,
            cax_height / yy,
        ]
    )

    # load and plot the gas image here; routine in gas.py
    # frame width also set there
    plot_gas(ax_sky, cax)

    # set colorbar label
    cax.tick_params(
        axis="both",
        labelsize="small",
        labeltop=True,
        labelbottom=False,
        which="both",
        direction="in",
        bottom=False,
        top=True,
        pad=2,
    )
    cax.xaxis.set_label_position("top")
    cax.set_xlabel(r"$v_\mathrm{BARY} \quad {\rm[km\,s^{-1}]}$", labelpad=2)

    # plot the star
    ax_sky.plot(0, 0, "*", ms=5, color="k", mew=0.1, zorder=99)
    # plot the sky positions
    X_ABs = d.wds[1] * np.cos(d.wds[3])  # north
    Y_ABs = d.wds[1] * np.sin(d.wds[3])  # east
    ax_sky.plot(Y_ABs, X_ABs, **pkw)
    plot_errorbar(ax_sky, d.wds[3], d.wds[1], d.wds[4], d.wds[2], color="C0", lw=0.8)

    # make the right-column plots
    yr_lim = (1990, 2020)
    ax_sep = fig.add_axes(
        [
            (lmargin + ax_width + mmargin) / xx,
            (2 * (sax_height + hmargin) + bmargin) / yy,
            ax_width / xx,
            sax_height / yy,
        ]
    )
    ax_pa = fig.add_axes(
        [
            (lmargin + ax_width + mmargin) / xx,
            (sax_height + hmargin + bmargin) / yy,
            ax_width / xx,
            sax_height / yy,
        ]
    )
    ax_V = fig.add_axes(
        [
            (lmargin + ax_width + mmargin) / xx,
            bmargin / yy,
            ax_width / xx,
            sax_height / yy,
        ]
    )

    # get the mean value of all samples to use for errorbars and offset
    sep_err = np.sqrt(d.wds[2] ** 2 + np.exp(2 * np.mean(trace["logRhoS"])))
    ax_sep.errorbar(jd_to_year(d.wds[0]), d.wds[1], yerr=sep_err, **ekw)
    ax_sep.set_ylabel(r"$\rho\;[{}^{\prime\prime}]$")
    ax_sep.set_xlim(*yr_lim)
    ax_sep.xaxis.set_ticklabels([])

    pa_err = np.sqrt(d.wds[4] ** 2 + np.exp(2 * np.mean(trace["logThetaS"])))
    ax_pa.errorbar(jd_to_year(d.wds[0]), d.wds[3] / deg + 360, yerr=pa_err / deg, **ekw)
    ax_pa.set_ylabel(r"$\theta\ [{}^\circ]$")
    ax_pa.set_xlim(*yr_lim)
    ax_pa.xaxis.set_ticklabels([])

    VB_err = np.sqrt(d.keck3[2] ** 2 + np.exp(2 * np.mean(trace["logjitterkeck"])))
    ax_V.errorbar(
        jd_to_year(d.keck3[0]),
        d.keck3[1] - np.mean(trace["offsetKeck"]),
        yerr=VB_err,
        **kekw,
        label="Keck",
    )

    ax_V.set_xlim(*yr_lim)
    ax_V.set_ylim(6, 11)
    ax_V.set_ylabel(r"$v_\mathrm{B} \quad [\mathrm{km\;s}^{-1}$]")
    ax_V.set_xlabel("Epoch [yr]")
    ax_V.legend(
        loc="upper left",
        fontsize="xx-small",
        labelspacing=0.5,
        handletextpad=0.2,
        borderpad=0.4,
        borderaxespad=1.0,
    )

    ######
    # Plot the orbits on the  sep_pa.pdf figure
    ######

    # define new Theano variables to get the
    # 1) absolute X,Y,Z positions evaluated over the full orbital period
    # 2) the sep, PA values evaluated over the observational window
    # 3) the vB evaluated over the observational window

    # phases for the full orbital period
    phases = np.linspace(0, 1.0, num=500)

    # times for the observational window (unchanging)
    t_yr = Time(np.linspace(*yr_lim, num=500), format="byear").byear  # [yrs]
    t_obs = Time(np.linspace(*yr_lim, num=500), format="byear").jd - jd0  # [days]

    with m.model:
        # the reason why we can refer to m.orbit_outer here (rather than, say)
        # m.model.orbit_outer, is because orbit_outer is actually in the scope of the
        # model module. It is in the *context* of the pymc3 model.model.
        # so we need both, for it to make sense.

        # convert phases to times stretching the full orbital period [days]
        t_period = phases * m.P_outer  # [days]

        # 1) absolute X,Y,Z positions evaluated over the full orbital period
        pos_outer = m.orbit_outer.get_relative_position(t_period, m.parallax)

        # 2) the sep, PA values evaluated over the observational window
        angles_outer = m.orbit_outer.get_relative_angles(t_obs, m.parallax)

        # 3) the vB evaluated over the observational window
        rv3 = (
            conv * m.orbit_outer.get_planet_velocity(t_obs)[2]
            + m.gamma_outer
            # do not include the offset, because we are plotting in the CfA frame
            # + m.offset_keck
        )

    # iterate among several samples to plot the orbits on the sep_pa.pdf figure.
    for sample in xo.get_samples_from_trace(trace, size=30):

        vBs = xo.eval_in_model(rv3, point=sample, model=m.model)
        # we can select orbits which have a higher vB[-1] than vB[0]
        # and colorize them

        # increasing = vBs[-1] > vBs[0]
        if sample["increasing"]:
            lkw["color"] = "C0"
            ax_V.plot(t_yr, vBs, **lkw, zorder=1)
        else:
            lkw["color"] = "C1"
            ax_V.plot(t_yr, vBs, **lkw, zorder=0)

        rho, theta = xo.eval_in_model(angles_outer, point=sample, model=m.model)
        ax_sep.plot(t_yr, rho, **lkw)
        ax_pa.plot(t_yr, theta / deg + 360, **lkw)

        X, Y, Z = xo.eval_in_model(pos_outer, point=sample, model=m.model)
        ax_sky.plot(Y, X, **lkw)

    return fig
