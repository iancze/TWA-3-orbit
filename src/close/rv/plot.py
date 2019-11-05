import corner
import matplotlib.pyplot as plt


# date is HJD + 2400000

fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(8, 8))

pkw = {"marker": ".", "ls": ""}

for d in data:
    ax[0].plot(d["HJD"], d["RV_Aa"], **pkw)
    ax[0].set_ylabel(r"$v_\mathrm{Aa}$ [km/s]")

    ax[1].plot(d["HJD"], d["RV_Ab"], **pkw)
    ax[1].set_ylabel(r"$v_\mathrm{Ab}$ [km/s]")

    ax[2].plot(d["HJD"], d["RV_B"], **pkw)
    ax[2].set_ylabel(r"$v_\mathrm{B}$ [km/s]")

#     ax[1].plot(d[])


# try phase folding the data for Aa and Ab
P_A = 34.87846  # pm 0.00090 days
# P_A = 35.245521 #

pkw = {"marker": ".", "ls": ""}

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8, 6))

for d in data:
    ax[0].plot(d["HJD"] % P_A, d["RV_Aa"], **pkw)
    ax[0].set_ylabel(r"$v_\mathrm{Aa}$ [km/s]")

    ax[1].plot(d["HJD"] % P_A, d["RV_Ab"], **pkw)
    ax[1].set_ylabel(r"$v_\mathrm{Ab}$ [km/s]")


xs_phase = np.linspace(0, 1, num=500)


pm.traceplot(
    trace,
    varnames=[
        "logP",
        "logKAa",
        "logKAb",
        "e",
        "omega",
        "tPeri",
        "offsetKeck",
        "offsetFeros",
        "offsetDupont",
        "logjittercfa",
        "logjitterkeck",
        "logjitterferos",
        "logjitterdupont",
    ],
)


with model:
    rv1 = xo.eval_in_model(rv1_dense, map_sol)
    rv2 = xo.eval_in_model(rv2_dense, map_sol)

    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].plot(xs_phase, rv1)
    ax[1].plot(xs_phase, rv2)


# To assess the quality of the fit, we should go and plot the fit and residuals for all of the data points individually and together.
#
# The phase-folding plot only really works for a fixed value of `tperi`, `P`. So, we can plot the MAP phase fold. But other than that it only makes sense to plot the orbit scatter on the actual series of points (minus any offset, too).


pkw = {"marker": ".", "ls": ""}
ekw = {"marker": ".", "ls": ""}


def get_phase(dates, P, tperi):
    return ((dates - tperi) % P) / P


# nsamples = 10
# choices = np.random.choice(np.arange(len(trace)), size=nsamples)

# just choose one representative sample
np.random.seed(43)
choice = np.random.choice(np.arange(len(trace)))

fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(8, 8))

with model:

    pos = trace[choice]

    tperi = pos["tPeri"]
    P = pos["P"]
    logjit = pos["logjittercfa"]

    err1 = np.sqrt(cfa1[2] ** 2 + np.exp(2 * logjit))
    err2 = np.sqrt(cfa2[2] ** 2 + np.exp(2 * logjit))

    phase1 = get_phase(cfa1[0], P, tperi)
    rv1 = xo.eval_in_model(rv1_cfa, pos)
    ax[0].errorbar(phase1, cfa1[1], yerr=err1, **ekw, zorder=0)
    ax[0].plot(phase1, rv1, **pkw, zorder=1)

    ax[1].axhline(0.0, color="k", lw=0.5)
    ax[1].errorbar(phase1, cfa1[1] - rv1, yerr=err1, **ekw)

    phase2 = get_phase(cfa2[0], P, tperi)
    rv2 = xo.eval_in_model(rv2_cfa, pos)
    ax[2].errorbar(phase2, cfa2[1], yerr=err2, **ekw, zorder=0)
    ax[2].plot(phase2, rv2, **pkw, zorder=1)

    ax[3].axhline(0.0, color="k", lw=0.5)
    ax[3].errorbar(phase2, cfa2[1] - rv2, yerr=err2, **ekw)


pkw = {"marker": ".", "ls": ""}
ekw = {"marker": ".", "ls": ""}


def get_phase(dates, P, tperi):
    return ((dates - tperi) % P) / P


# just choose one representative sample
np.random.seed(41)
choice = np.random.choice(np.arange(len(trace)))

fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(8, 8))

with model:

    pos = trace[choice]

    tperi = pos["tPeri"]
    P = pos["P"]

    logjit = pos["logjitterkeck"]

    err1 = np.sqrt(keck1[2] ** 2 + np.exp(2 * logjit))
    err2 = np.sqrt(keck2[2] ** 2 + np.exp(2 * logjit))

    phase1 = get_phase(keck1[0], P, tperi)
    rv1 = xo.eval_in_model(rv1_keck, pos)
    ax[0].errorbar(phase1, keck1[1], yerr=err1, **ekw, zorder=0)
    ax[0].plot(phase1, rv1, **pkw, zorder=1)

    ax[1].axhline(0.0, color="k", lw=0.5)
    ax[1].errorbar(phase1, keck1[1] - rv1, yerr=err1, **ekw)

    phase2 = get_phase(keck2[0], P, tperi)
    rv2 = xo.eval_in_model(rv2_keck, pos)
    ax[2].errorbar(phase2, keck2[1], yerr=err2, **ekw, zorder=0)
    ax[2].plot(phase2, rv2, **pkw, zorder=1)

    ax[3].axhline(0.0, color="k", lw=0.5)
    ax[3].errorbar(phase2, keck2[1] - rv2, yerr=err2, **ekw)


pkw = {"marker": ".", "ls": ""}
ekw = {"marker": ".", "ls": ""}


def get_phase(dates, P, tperi):
    return ((dates - tperi) % P) / P


# just choose one representative sample
np.random.seed(43)
choice = np.random.choice(np.arange(len(trace)))

fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(8, 8))

with model:

    pos = trace[choice]

    tperi = pos["tPeri"]
    P = pos["P"]
    logjit = pos["logjitterferos"]

    err1 = np.sqrt(feros1[2] ** 2 + np.exp(2 * logjit))
    err2 = np.sqrt(feros2[2] ** 2 + np.exp(2 * logjit))

    phase1 = get_phase(feros1[0], P, tperi)
    rv1 = xo.eval_in_model(rv1_feros, pos)
    ax[0].errorbar(phase1, feros1[1], yerr=err1, **ekw, zorder=0)
    ax[0].plot(phase1, rv1, **pkw, zorder=1)

    ax[1].axhline(0.0, color="k", lw=0.5)
    ax[1].errorbar(phase1, feros1[1] - rv1, yerr=err1, **ekw)

    phase2 = get_phase(feros2[0], P, tperi)
    rv2 = xo.eval_in_model(rv2_feros, pos)
    ax[2].errorbar(phase2, feros2[1], yerr=err2, **ekw, zorder=0)
    ax[2].plot(phase2, rv2, **pkw, zorder=1)

    ax[3].axhline(0.0, color="k", lw=0.5)
    ax[3].errorbar(phase2, feros2[1] - rv2, yerr=err2, **ekw)


pkw = {"marker": ".", "ls": ""}
ekw = {"marker": ".", "ls": ""}


def get_phase(dates, P, tperi):
    return ((dates - tperi) % P) / P


# nsamples = 10
# choices = np.random.choice(np.arange(len(trace)), size=nsamples)

# just choose one representative sample
np.random.seed(43)
choice = np.random.choice(np.arange(len(trace)))

fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(8, 8))

with model:

    pos = trace[choice]

    tperi = pos["tPeri"]
    P = pos["P"]

    logjit = pos["logjitterdupont"]

    err1 = np.sqrt(dupont1[2] ** 2 + np.exp(2 * logjit))
    err2 = np.sqrt(dupont2[2] ** 2 + np.exp(2 * logjit))

    phase1 = get_phase(dupont1[0], P, tperi)
    rv1 = xo.eval_in_model(rv1_dupont, pos)
    ax[0].errorbar(phase1, dupont1[1], yerr=err1, **ekw, zorder=0)
    ax[0].plot(phase1, rv1, **pkw, zorder=1)

    ax[1].axhline(0.0, color="k", lw=0.5)
    ax[1].errorbar(phase1, dupont1[1] - rv1, yerr=err1, **ekw)

    phase2 = get_phase(dupont2[0], P, tperi)
    rv2 = xo.eval_in_model(rv2_dupont, pos)
    ax[2].errorbar(phase2, dupont2[1], yerr=err2, **ekw, zorder=0)
    ax[2].plot(phase2, rv2, **pkw, zorder=1)

    ax[3].axhline(0.0, color="k", lw=0.5)
    ax[3].errorbar(phase2, dupont2[1] - rv2, yerr=err2, **ekw)


# In[167]:


# plot everything ontop in a single plot

pkw = {"marker": ".", "ls": ""}
ekw = {"marker": ".", "ms": 5.0, "ls": "", "elinewidth": 1.2}


def get_phase(dates, pos):
    return ((dates - pos["tPeri"]) % pos["P"]) / pos["P"]


# nsamples = 10
# choices = np.random.choice(np.arange(len(trace)), size=nsamples)

# just choose one representative sample
np.random.seed(43)
choice = np.random.choice(np.arange(len(trace)))

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8, 4))

with model:

    pos = trace[choice]

    tperi = pos["tPeri"]
    P = pos["P"]

    # calculate the errors for each instrument
    cfa_err1 = np.sqrt(cfa1[2] ** 2 + np.exp(2 * pos["logjittercfa"]))
    cfa_err2 = np.sqrt(cfa2[2] ** 2 + np.exp(2 * pos["logjittercfa"]))

    keck_err1 = np.sqrt(keck1[2] ** 2 + np.exp(2 * pos["logjitterkeck"]))
    keck_err2 = np.sqrt(keck2[2] ** 2 + np.exp(2 * pos["logjitterkeck"]))

    feros_err1 = np.sqrt(feros1[2] ** 2 + np.exp(2 * pos["logjitterferos"]))
    feros_err2 = np.sqrt(feros2[2] ** 2 + np.exp(2 * pos["logjitterferos"]))

    dupont_err1 = np.sqrt(dupont1[2] ** 2 + np.exp(2 * pos["logjitterdupont"]))
    dupont_err2 = np.sqrt(dupont2[2] ** 2 + np.exp(2 * pos["logjitterdupont"]))

    # plot RV1 model
    ax[0].axhline(pos["gamma"], lw=1.0, color="k", ls=":")
    ax[0].plot(xs_phase, pos["RV1Dense"], zorder=-1)

    # at data locations
    ax[0].errorbar(get_phase(cfa1[0], pos), cfa1[1], yerr=cfa_err1, **ekw, zorder=0)
    ax[0].errorbar(
        get_phase(keck1[0], pos),
        keck1[1] - pos["offsetKeck"],
        yerr=keck_err1,
        **ekw,
        zorder=0
    )
    ax[0].errorbar(
        get_phase(feros1[0], pos),
        feros1[1] - pos["offsetFeros"],
        yerr=feros_err1,
        **ekw,
        zorder=0
    )
    ax[0].errorbar(
        get_phase(dupont1[0], pos),
        dupont1[1] - pos["offsetDupont"],
        yerr=dupont_err1,
        **ekw,
        zorder=0
    )

    # plot RV2
    ax[1].axhline(pos["gamma"], lw=1.0, color="k", ls=":")
    ax[1].plot(xs_phase, pos["RV2Dense"], zorder=-1)

    # at data locations
    ax[1].errorbar(get_phase(cfa2[0], pos), cfa2[1], yerr=cfa_err2, **ekw, zorder=0)
    ax[1].errorbar(
        get_phase(keck2[0], pos),
        keck2[1] - pos["offsetKeck"],
        yerr=keck_err2,
        **ekw,
        zorder=0
    )
    ax[1].errorbar(
        get_phase(feros2[0], pos),
        feros2[1] - pos["offsetFeros"],
        yerr=feros_err2,
        **ekw,
        zorder=0
    )
    ax[1].errorbar(
        get_phase(dupont2[0], pos),
        dupont2[1] - pos["offsetDupont"],
        yerr=dupont_err2,
        **ekw,
        zorder=0
    )

    ax[1].set_xlim(0.0, 1.0)
    ax[0].set_ylabel(r"$v_\mathrm{Aa}$ $[\mathrm{km s}^{-1}]$")
    ax[1].set_ylabel(r"$v_\mathrm{Ab}$ $[\mathrm{km s}^{-1}]$")
    ax[1].set_xlabel("phase")

fig.subplots_adjust(top=0.98, bottom=0.18, hspace=0.05)
fig.savefig("A_sb_orbit.pdf")


samples = pm.trace_to_dataframe(
    trace,
    varnames=[
        "P",
        "KAa",
        "KAb",
        "e",
        "gamma",
        "omega",
        "offsetKeck",
        "offsetFeros",
        "offsetDupont",
        "jitCfa",
        "jitKeck",
        "jitFeros",
        "jitDupont",
    ],
)
samples["omega"] /= deg
corner.corner(samples)
