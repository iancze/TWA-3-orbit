import arviz as az
import corner
import exoplanet as xo
import matplotlib.pyplot as plt
import pymc3 as pm

import src.close.rv_astro_less.model as m
import src.data as d
from src.constants import *

plotdir = "figures/close/rv_astro_less/"

trace = pm.load_trace(directory="chains/close/rv_astro_less", model=m.model)

# view summary
df = az.summary(trace, var_names=m.all_vars)
print(df)

# write summary to disk
f = open(f"{plotdir}summary.txt", "w")
df.to_string(f)
f.close()


with az.rc_context(rc={"plot.max_subplots": 80}):
    # autocorrelation
    az.plot_autocorr(trace, var_names=m.sample_vars)
    plt.savefig(f"{plotdir}autocorr.png")

    # make a traceplot
    az.plot_trace(trace, var_names=m.all_vars)
    plt.savefig(f"{plotdir}trace.png")


# make a nice corner plot of the variables we care about
samples = pm.trace_to_dataframe(
    trace,
    varnames=["MAa", "MAb", "a", "P", "e", "gamma", "incl", "omega", "Omega", "tPeri"],
)
samples["incl"] /= deg
samples["omega"] /= deg
samples["Omega"] /= deg
fig = corner.corner(samples)
fig.savefig(f"{plotdir}corner.png")


# open the model context and add new variables
# that will be useful for plotting
color_dict = {"CfA": "C0", "Keck": "C1", "FEROS": "C2", "du Pont": "C3"}
xs_phase = np.linspace(0, 1, num=500)

# define new Theano variables to get what we want to plot
with m.model:
    ts_phases = xs_phase * m.P + m.t_periastron  # new theano var
    rv1, rv2 = m.get_RVs(ts_phases, ts_phases, 0.0)

    # get the predictions at the times of the data with *no* offset
    rv_cfa0 = m.get_RVs(d.cfa1[0], d.cfa2[0], 0.0)
    rv_keck0 = m.get_RVs(d.keck1[0], d.keck2[0], 0.0)
    rv_feros0 = m.get_RVs(d.feros1[0], d.feros2[0], 0.0)
    rv_dupont0 = m.get_RVs(d.dupont1[0], d.dupont2[0], 0.0)


# plot phase-folded RV orbits

# set up the figure dimensions and plot the data
lmargin = 0.5
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

hkw = {"lw": 1.0, "color": "0.4", "ls": ":"}
pkw = {"ls": "-", "lw": 1.5, "color": "k"}
ekw = {"marker": ".", "ms": 3.0, "ls": "", "elinewidth": 0.8, "zorder": 20}


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


def phase_and_plot(data, model, a, a_r, label):
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

    phase = get_phase(data[0])

    # evaluate the model with the current parameter settings
    offset = offset_dict[label]
    d = data[1] - offset
    resid = d - model
    err = np.sqrt(data[2] ** 2 + np.exp(2 * err_dict[label]))

    color = color_dict[label]
    a.errorbar(phase, d, yerr=err, label=label, **ekw, color=color)
    a_r.errorbar(phase, resid, yerr=err, **ekw, color=color)


# just use model.test_point
for sample in xo.get_samples_from_trace(trace, size=1):

    # we'll want to cache these functions when we evaluate many samples
    rv1_m = xo.eval_in_model(rv1, point=sample, model=m.model)
    rv2_m = xo.eval_in_model(rv2, point=sample, model=m.model)

    ax1.plot(xs_phase, rv1_m, **pkw)
    ax2.plot(xs_phase, rv2_m, **pkw)

    err_dict = {
        "CfA": sample["logjittercfa"],
        "Keck": sample["logjitterkeck"],
        "FEROS": sample["logjitterferos"],
        "du Pont": sample["logjitterdupont"],
    }
    offset_dict = {
        "CfA": 0.0,
        "Keck": sample["offsetKeck"],
        "FEROS": sample["offsetFeros"],
        "du Pont": sample["offsetDupont"],
    }

    # create a phasing function local to these sampled values of
    # t_periastron and period
    def get_phase(dates):
        return ((dates - sample["tPeri"]) % sample["P"]) / sample["P"]

    rv_cfa0_1, rv_cfa0_2 = xo.eval_in_model(rv_cfa0, point=sample, model=m.model)
    rv_keck0_1, rv_keck0_2 = xo.eval_in_model(rv_keck0, point=sample, model=m.model)
    rv_feros0_1, rv_feros0_2 = xo.eval_in_model(rv_feros0, point=sample, model=m.model)
    rv_dupont0_1, rv_dupont0_2 = xo.eval_in_model(
        rv_dupont0, point=sample, model=m.model
    )

    # Plot the data and residuals for this sample
    phase_and_plot(d.cfa1, rv_cfa0_1, ax1, ax1_r, "CfA")
    phase_and_plot(d.keck1, rv_keck0_1, ax1, ax1_r, "Keck")
    phase_and_plot(d.feros1, rv_feros0_1, ax1, ax1_r, "FEROS")
    phase_and_plot(d.dupont1, rv_dupont0_1, ax1, ax1_r, "du Pont")

    phase_and_plot(d.cfa2, rv_cfa0_2, ax2, ax2_r, "CfA")
    phase_and_plot(d.keck2, rv_keck0_2, ax2, ax2_r, "Keck")
    phase_and_plot(d.feros2, rv_feros0_2, ax2, ax2_r, "FEROS")
    phase_and_plot(d.dupont2, rv_dupont0_2, ax2, ax2_r, "du Pont")


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

# fig.savefig(f"{outdir}inner_RV.png")
fig.savefig(f"{plotdir}RV.pdf")


# plot predicted sep and pa vs time plots
ts_astro_dense = np.linspace(
    np.min(d.anthonioz[0]) - 40, np.max(d.anthonioz[0]) + 40, num=2000
)
with m.model:
    rv_ast = m.get_RVs(ts_astro_dense, ts_astro_dense, 0.0)
    rho_dense, theta_dense = m.orbit.get_relative_angles(ts_astro_dense, m.parallax)


# create a sep / pa figure
# 4 panels, specify RV on top as well
fig, ax = plt.subplots(nrows=4, sharex=True)

for sample in xo.get_samples_from_trace(trace, size=20):
    rv1_dense, rv2_dense = xo.eval_in_model(rv_ast, point=sample, model=m.model)

    rdense = xo.eval_in_model(rho_dense, point=sample, model=m.model)
    tdense = xo.eval_in_model(theta_dense, point=sample, model=m.model)

    ax[0].plot(ts_astro_dense, rv1_dense, **pkw)
    ax[1].plot(ts_astro_dense, rv2_dense, **pkw)

    ax[2].plot(ts_astro_dense, rdense, **pkw)
    ax[3].plot(ts_astro_dense, tdense / deg, **pkw)


# plot the actual data on top
ax[0].errorbar(d.anthonioz[0], d.anthonioz[1], yerr=d.anthonioz[2], **ekw)
ax[0].set_xlabel(r"$\rho [{}^{\prime\prime}]$")
ax[1].errorbar(d.anthonioz[0], d.anthonioz[3] / deg, yerr=d.anthonioz[4] / deg, **ekw)
ax[1].set_xlabel(r"$\theta [{}^\circ]$")

fig.savefig(f"{plotdir}sep_pa.pdf")
