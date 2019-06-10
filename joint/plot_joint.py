import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from matplotlib import rcParams
# rcParams["text.usetex"] = False

from astropy.time import Time
from astropy.io import ascii
from astropy import units as u
from astropy import constants
import astropy

import theano
import theano.tensor as tt

import exoplanet as xo
from exoplanet.distributions import Angle

from gas import plot_gas

deg = np.pi/180. # radians / degree
yr = 365.25 # days / year

au_to_R_sun = (constants.au / constants.R_sun).value # conversion constant

# convert from R_sun / day to km/s
# and from v_r = - v_Z
output_units = u.km / u.s
conv = -(1 * u.R_sun / u.day).to(output_units).value

# Hierarchical triple orbit simultaneously fit with RV and astrometry for both tight inner binary and wide outer binary: inner: `parallax`, `P_A`, `a_A_ang`, `M_Ab`, `e_A`, `i_A`, `omega_Aa`, `Omega_Aa`,  outer: `P_B`, `a_B_ang`, `e_AB`, `i_AB`, `omega_A`, `Omega_A`, `gamma_AB`. `M_A` is derived from inner orbit and fed to outer orbit.  `gamma_A` is essentially the RV prediction of A, and is derived from outer orbit and fed to the inner orbit. This has 15 orbital parameters. Adding 4 RV offsets, 2 * 4 RV jitter terms, and 2 astrometric jitter terms makes it 30 parameters total.

# for now, subtract from all dates for start of astrometric timeseries
jd0 = 2448630.485039346

def get_arrays(asciiTable, errDict=None, jitter=False):
    """
    Reformat ascii tables into pure numpy arrays of the right dimension.
    """

    output = []

    for star in ["Aa", "Ab"]:

        # get the RVs
        rv = asciiTable["RV_" + star]

        if type(rv) is astropy.table.column.MaskedColumn:
            mask = ~rv.mask # values we want to keep when indexing
        else:
            mask = np.ones(len(rv), dtype="bool")

        rv = np.ascontiguousarray(rv[mask])
        date = np.ascontiguousarray(asciiTable["HJD"][mask]) + 2400000 - jd0

        if errDict is None:
            err = np.ascontiguousarray(asciiTable["sigma_" + star][mask])
        else:
            err = np.ones(len(date), dtype=np.float64) * errDict[star]

        if jitter:
            err = np.ones(len(date), dtype=np.float64) * 0.1 # [km/s] assume a small error, since we'll infer.

        assert len(date) == len(rv), "date - rv length mismatch"
        assert len(date) == len(err), "date - err length mismatch"

        tup = (date, rv, err)

        output.append(tup)

    return output

# Do this to infer w/ jitter
jitter=True

# load all the data
data_cfa = ascii.read("../close/data/cfa.dat")
# cfa errors are provided in table
cfa1,cfa2 = get_arrays(data_cfa, jitter=jitter)

data_keck = ascii.read("../close/data/keck.dat", format="tab", fill_values=[("X", 0)])
err_keck = {"Aa":0.63, "Ab":0.85, "B":0.59} # km/s
keck1,keck2 = get_arrays(data_keck, err_keck, jitter=jitter)

data_feros = ascii.read("../close/data/feros.dat")
err_feros = {"Aa":2.61, "Ab":3.59, "B":2.60} # km/s
feros1,feros2 = get_arrays(data_feros, err_feros, jitter=jitter)

data_dupont = ascii.read("../close/data/dupont.dat", fill_values=[("X", 0)])
err_dupont = {"Aa":1.46, "Ab":2.34, "B":3.95} # km/s
dupont1,dupont2 = get_arrays(data_dupont, err_dupont, jitter=jitter)

data = [data_cfa, data_keck, data_feros, data_dupont]


# specifically load the B velocities
mask = ~data_keck["RV_B"].mask

keck3 = (np.ascontiguousarray(data_keck["HJD"][mask]) + 2400000 - jd0,
         np.ascontiguousarray(data_keck["RV_B"][mask]), 0.2 * np.ones(np.sum(mask), dtype=np.float64))


# load the astrometric points for the close orbit

# keep in mind that the primary and secondary stars *could* be switched
# separation is in milliarcseconds
int_data = ascii.read("../close/data/int_data.dat")

astro_jd = int_data["epoch"][0] - jd0
rho_data = int_data["sep"][0] * 1e-3 # arcsec
rho_err = int_data["sep_err"][0] * 1e-3 # arcsec
theta_data = int_data["pa"][0] * deg # radians
theta_err = int_data["pa_err"][0] * deg # radians

anthonioz = (astro_jd, rho_data, rho_err, theta_data, theta_err)

# load the astrometry for the wide orbit
data = ascii.read("../wide/data/visual_data_besselian.csv", format="csv", fill_values=[("X", '0')])

# convert years
jds = Time(np.ascontiguousarray(data["epoch"]), format="byear").jd - jd0

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

wds = (jds, rho_data, rho_err, theta_data, theta_err)


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


def calc_a(Mtot, P):
    '''
    Calculate the semi-major axis using Kepler's third law

    Args:
        Mtot (Msun) total mass
        P (days) period

    Returns:
        a (au)
    '''

    day_to_s = (1 * u.day).to(u.s).value
    au_to_m = (1 * u.au).to(u.m).value
    kg_to_M_sun = (1 * u.kg).to(u.M_sun).value

    return (((Mtot / kg_to_M_sun) * constants.G.value * (P * day_to_s)**2) / (4 * np.pi**2))**(1/3) / au_to_m


xs_phase = np.linspace(0, 1, num=500)
t_data = np.linspace(-yr, wds[0][-1] + yr)

# make theano functions which calculate the orbital quantities from our samples

# Orbital summary plots

# 1) double-lined RV solution w/ residuals

# rvAa(t)
# rvAb(t)
# correct the data from each RV instrument based on offset
# one draw only, since P and offsets don't scatter well in this space

# 2) plot of X,Y (Ab rel Aa)
# multiple posterior draws

# 3) plot of rvB(t) over keck datarange, with increasing RV_B orbits highlighted

# 4) plot of X,Y (B rel A), with increasing RV_B orbits highlighted

t = tt.vector("times")

mparallax = tt.dscalar('mparallax')
parallax = 1e-3 * mparallax # arcsec

a_ang_inner = tt.dscalar("a_ang_inner") # milliarcsec

# the semi-major axis in au
a_inner = 1e-3 * a_ang_inner / parallax # au

logP_inner = tt.dscalar("logP_inner") # days
P_inner = tt.exp(logP_inner)

e_inner = tt.dscalar("e_inner")

omega_inner = tt.dscalar("omega_inner") # omega_Aa
Omega_inner = tt.dscalar("Omega_inner")

cos_incl_inner = tt.dscalar("cos_incl_inner")
incl_inner = tt.arccos(cos_incl_inner)

MAb = tt.dscalar("MAb")

t_periastron_inner = tt.dscalar("t_periastron_inner")

orbit_inner = xo.orbits.KeplerianOrbit(a=a_inner*au_to_R_sun, period=P_inner, ecc=e_inner,
                  t_periastron=t_periastron_inner, omega=omega_inner, Omega=Omega_inner,
                                           incl=incl_inner, m_planet=MAb)

MA = orbit_inner.m_total

logP_outer = tt.dscalar("logP_outer") # yrs
P_outer = tt.exp(logP_outer) * yr # days

omega_outer = tt.dscalar("omega_outer") # - pi to pi
Omega_outer = tt.dscalar("Omega_outer") # - pi to pi

phi_outer = tt.dscalar("phi_outer")

n = 2*np.pi*tt.exp(-logP_outer) / yr # radians per day
t_periastron_outer = (phi_outer + omega_outer) / n

cos_incl_outer = tt.dscalar("cos_incl_outer")
incl_outer = tt.arccos(cos_incl_outer)
e_outer = tt.dscalar("e_outer")
gamma_outer = tt.dscalar("gamma_outer") # km/s on CfA RV scale

MB = tt.dscalar("MB")
Mtot = MA + MB

# instead, what if MB becomes a parameter, and then a_outer is calculated from Mtot and P_outer?
a_outer = calc_a(Mtot, P_outer)

orbit_outer = xo.orbits.KeplerianOrbit(a=a_outer*au_to_R_sun, period=P_outer, ecc=e_outer,
        t_periastron=t_periastron_outer, omega=omega_outer, Omega=Omega_outer,
                                 incl=incl_outer, m_planet=MB)

gamma_A = conv * orbit_outer.get_star_velocity(t)[2]

rv_Aa = conv * orbit_inner.get_star_velocity(t)[2] + gamma_outer + gamma_A
rv_Ab = conv * orbit_inner.get_planet_velocity(t)[2] + gamma_outer + gamma_A
rv_B = conv * orbit_outer.get_planet_velocity(t)[2] + gamma_outer

sky_inner = orbit_inner.get_relative_angles(t, parallax) # arcsec
sky_outer = orbit_outer.get_relative_angles(t, parallax) # arcsec

all_pars = [t, mparallax, MAb, logP_outer, a_ang_inner, logP_inner, e_inner, omega_inner, Omega_inner, cos_incl_inner, t_periastron_inner, omega_outer, Omega_outer, phi_outer, cos_incl_outer, e_outer, gamma_outer, MB]

f_rv_Aa = theano.function(all_pars, rv_Aa, on_unused_input='ignore')
f_rv_Ab = theano.function(all_pars, rv_Ab, on_unused_input='ignore')
f_rv_B = theano.function(all_pars, rv_B, on_unused_input='ignore')

f_sky_inner = theano.function(all_pars, sky_inner, on_unused_input='ignore')
f_sky_outer = theano.function(all_pars, sky_outer, on_unused_input='ignore')

df = pd.read_csv("current.csv")

np.random.seed(42)
# also choose a sample at random and use the starting position
sample_pars = ['mparallax', 'MAb', 'a_ang_inner', 'logP_inner', 'e_inner', 'omega_inner', 'Omega_inner', 'cos_incl_inner', 't_periastron_inner', 'logP_outer', 'omega_outer', 'Omega_outer', 'phi_outer', 'cos_incl_outer', 'e_outer', 'gamma_outer', 'MB']

all_vars=['mparallax', 'MAb', 'a_ang_inner', 'logP_inner', 'e_inner', 'omega_inner', 'Omega_inner', 'cos_incl_inner', 't_periastron_inner', 'logP_outer', 'omega_outer', 'Omega_outer', 'phi_outer', 'cos_incl_outer', 'e_outer', 'gamma_outer', 'MB', 'offsetKeck', 'offsetFeros', 'offsetDupont', 'logRhoS', 'logThetaS', 'logjittercfa', 'logjitterkeck', 'logjitterferos', 'logjitterdupont']

row0 = df.sample()
point = {par:row0[par].item() for par in sample_pars}
apoint = {par:row0[par].item() for par in all_vars}

P_inner = np.exp(point["logP_inner"]) # days

def get_phase(dates):
    return ((dates - point["t_periastron_inner"]) % P_inner) / P_inner

lmargin = 0.58
rmargin = lmargin
bmargin = 0.4
tmargin = 0.05
mmargin = 0.07
mmmargin = 0.15

ax_height = 1.0
ax_r_height = 0.4

xx = 3.5
ax_width = xx - lmargin - rmargin

yy = bmargin + tmargin + 2 * mmargin + mmmargin + 2 * ax_height + 2 * ax_r_height

hkw = {"lw":1.0, "color":"0.4", "ls":":"}
pkw = {"ls":"-", "lw":1.5, "color":"k"}
ekw = {"marker":".", "ms":3.0, "ls":"", "elinewidth":0.8, "zorder":20}

xs_phase = np.linspace(0, 1, num=500)
ts_phases = xs_phase * P_inner + point["t_periastron_inner"]
point['times'] = ts_phases

fig = plt.figure(figsize=(xx,yy))

ax1 = fig.add_axes([lmargin/xx, 1 - (tmargin + ax_height)/yy, ax_width/xx, ax_height/yy])
ax1.plot(xs_phase, f_rv_Aa(**point), **pkw)

ax1_r = fig.add_axes([lmargin/xx, 1 - (tmargin + mmargin + ax_height + ax_r_height)/yy, ax_width/xx, ax_r_height/yy])
ax1_r.axhline(0.0, **hkw)

ax2 = fig.add_axes([lmargin/xx, (bmargin + mmargin + ax_r_height)/yy, ax_width/xx, ax_height/yy])
ax2.plot(xs_phase, f_rv_Ab(**point), **pkw)

ax2_r = fig.add_axes([lmargin/xx, bmargin/yy, ax_width/xx, ax_r_height/yy])
ax2_r.axhline(0.0, **hkw)


err_dict = {"CfA":apoint["logjittercfa"], "Keck":apoint["logjitterkeck"], "FEROS":apoint['logjitterferos'], "du Pont":apoint["logjitterdupont"]}
offset_dict = {"CfA":0.0, "Keck":apoint["offsetKeck"], "FEROS":apoint["offsetFeros"], "du Pont":apoint["offsetDupont"]}
color_dict = {"CfA":"C0", "Keck":"C1", "FEROS":"C2", "du Pont":"C3"}

def plot_data(data, f, a, a_r, label):
    phase = get_phase(data[0])
    point["times"] = data[0]
    model = f(**point)
    offset = offset_dict[label]
    d = data[1] - offset
    resid = d - model
    err = np.sqrt(data[2]**2 + np.exp(2 * err_dict[label]))

    color = color_dict[label]
    a.errorbar(phase, d, yerr=err, label=label, **ekw, color=color)
    a_r.errorbar(phase, resid, yerr=err, **ekw, color=color)

plot_data(cfa1, f_rv_Aa, ax1, ax1_r, "CfA")
plot_data(keck1, f_rv_Aa, ax1, ax1_r, "Keck")
plot_data(feros1, f_rv_Aa, ax1, ax1_r, "FEROS")
plot_data(dupont1, f_rv_Aa, ax1, ax1_r, "du Pont")

plot_data(cfa2, f_rv_Ab, ax2, ax2_r, "CfA")
plot_data(keck2, f_rv_Ab, ax2, ax2_r, "Keck")
plot_data(feros2, f_rv_Ab, ax2, ax2_r, "FEROS")
plot_data(dupont2, f_rv_Ab, ax2, ax2_r, "du Pont")

ax1.legend(loc="upper left", fontsize="xx-small", labelspacing=0.5, handletextpad=0.2, borderpad=0.4, borderaxespad=1.0)

ax1.set_ylabel(r"$v_\mathrm{Aa}$ [$\mathrm{km s}^{-1}$]")
ax1_r.set_ylabel(r"$O-C$")

ax2.set_ylabel(r"$v_\mathrm{Ab}$ [$\mathrm{km s}^{-1}$]")
ax2_r.set_ylabel(r"$O-C$")
ax2_r.set_xlabel("phase")

ax = [ax1, ax1_r, ax2, ax2_r]
for a in ax:
    a.set_xlim(0, 1)

ax1.xaxis.set_ticklabels([])
ax1_r.xaxis.set_ticklabels([])
ax2.xaxis.set_ticklabels([])

fig.savefig("inner_RV.pdf")
fig.savefig("inner_RV.png", dpi=300)

# point["times"] = np.atleast_1d(anthonioz[0])
# rho_inner, theta_inner = f_sky_inner(**point)
# print(rho_inner, theta_inner)

def jd_to_year(t):
    '''
    Convert JD referenced to jd0 into a year.
    '''
    return Time(t + jd0, format="jd").byear


# Custom routine for plotting polar error bars onto cartesian plane
def plot_errorbar(ax, thetas, rhos, theta_errs, rho_errs, **kwargs):
    '''
    All kwargs sent to plot. Thetas are in radians.
    '''

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
        delta = rho * (1/np.cos(theta_err) - 1)
        xs = (rho + delta) * np.sin(np.array([theta - theta_err, theta + theta_err]))
        ys = (rho + delta) * np.cos(np.array([theta - theta_err, theta + theta_err]))
        ax.plot(xs, ys, **kwargs)

        # rho error
        xs = np.sin(theta) * np.array([(rho - rho_err), (rho + rho_err)])
        ys = np.cos(theta) * np.array([(rho - rho_err), (rho + rho_err)])
        ax.plot(xs, ys, **kwargs)

# Plot the astrometric fits.
# 1 square panel on left, then 3 rows of panels on right (rho, theta, v_B)

ax_width = 2.7
hmargin = 0.0
ax_height = 2.7
sax_height = (ax_height - hmargin)/3
tmargin = 0.1
bmargin = 0.5
lmargin = 0.5
rmargin = 0.5
mmargin = 0.75

pkw = {"marker":"o", "ms":3, "color":"C0", "ls":"", "zorder":20}
ekw = {"marker":"o", "ms":3, "ls":"", "elinewidth":0.8, "zorder":20}

xx = 2 * ax_width + lmargin + rmargin + mmargin
yy = ax_height + tmargin + bmargin

fig = plt.figure(figsize=(xx,yy))

ax_sky = fig.add_axes([lmargin/xx, bmargin/yy, ax_width/xx, ax_height/yy])
ax_sky.set_xlabel(r'$\Delta \alpha \cos \delta\; [{}^{\prime\prime}]$')
ax_sky.set_ylabel(r'$\Delta \delta\; [{}^{\prime\prime}]$')
# ax_sky.invert_xaxis()

# load and plot the gas image here; routine in gas.py
# frame width also set there
plot_gas(ax_sky)


# plot the star
ax_sky.plot(0,0, "*", ms=5, color="k", mew=0.1, zorder=99)
# plot the sky positions
X_ABs = wds[1] * np.cos(wds[3]) # north
Y_ABs = wds[1] * np.sin(wds[3]) # east
ax_sky.plot(Y_ABs, X_ABs, **pkw)
plot_errorbar(ax_sky, wds[3], wds[1], wds[4], wds[2], color="C0", lw=0.8)

# make the right-column plots
yr_lim = (1990, 2020)
ax_sep = fig.add_axes([(lmargin + ax_width + mmargin)/xx, (2 * (sax_height + hmargin) + bmargin)/yy, ax_width/xx, sax_height/yy])
ax_pa = fig.add_axes([(lmargin + ax_width + mmargin)/xx, (sax_height + hmargin + bmargin)/yy, ax_width/xx, sax_height/yy])
ax_V = fig.add_axes([(lmargin + ax_width + mmargin)/xx, bmargin/yy, ax_width/xx, sax_height/yy])

sep_err = np.sqrt(wds[2]**2 + np.exp(2 * apoint["logRhoS"]))
ax_sep.errorbar(jd_to_year(wds[0]), wds[1], yerr=sep_err, **ekw)
ax_sep.set_ylabel(r'$\rho\;[{}^{\prime\prime}]$')
ax_sep.set_xlim(*yr_lim)
ax_sep.xaxis.set_ticklabels([])

pa_err = np.sqrt(wds[4]**2 + np.exp(2 * apoint["logThetaS"]))
ax_pa.errorbar(jd_to_year(wds[0]), wds[3]/deg + 360, yerr=pa_err/deg, **ekw)
ax_pa.set_ylabel(r'$\theta\ [{}^\circ]$')
ax_pa.set_xlim(*yr_lim)
ax_pa.xaxis.set_ticklabels([])

kekw = {**ekw, "color":"C1"}
VB_err = np.sqrt(keck3[2]**2 + np.exp(2 * apoint["logjitterkeck"]))
ax_V.errorbar(jd_to_year(keck3[0]), keck3[1] - apoint["offsetKeck"], yerr=VB_err, **kekw, label="Keck")
ax_V.set_xlim(*yr_lim)
ax_V.set_ylim(6, 11)
ax_V.set_ylabel(r"$v_\mathrm{B} \quad [\mathrm{km\;s}^{-1}$]")
ax_V.set_xlabel("Epoch [yr]")
ax_V.legend(loc="upper left", fontsize="xx-small", labelspacing=0.5, handletextpad=0.2, borderpad=0.4, borderaxespad=1.0)


pos_outer = orbit_outer.get_relative_position(t, parallax)
f_pos_outer = theano.function(all_pars, pos_outer, on_unused_input='ignore')

# draw random samples and plot them on the figures
nsamples = 20
samples = df.sample(n=nsamples)

t_yr = np.linspace(*yr_lim, num=500)
t_obs = Time(t_yr, format="byear").jd - jd0

dkw = {"linewidth":0.5}

for index, row in samples.iterrows():

    # make the dictionary as before
    point = {par:row[par].item() for par in sample_pars}

    # choose the color based upon Omega family
    if point["Omega_outer"] > 50 * deg:
        color = "C3"
    else:
        color = "0.5"

    # calculate time ranges going from current point onward in time
    P_outer = np.exp(point["logP_outer"]) * yr # days
    t_full_period = np.linspace(0, P_outer, num=500)

    point["times"] = t_full_period

    # get the relative positions
    X,Y,Z = f_pos_outer(**point)

    ax_sky.plot(Y, X, lw=0.5, color=color, zorder=10)

    point["times"] = t_obs

    # get the sep, pa, and V_B for the observed ranges
    rho, theta = f_sky_outer(**point)
    ax_sep.plot(t_yr, rho, **dkw, color=color, alpha=0.7)
    ax_pa.plot(t_yr, theta/deg + 360, **dkw, color=color, alpha=0.7)

    vBs = f_rv_B(**point)
    ax_V.plot(t_yr, vBs, **dkw, color=color, alpha=0.7)

    # we can select orbits which have a higher vB[-1] than vB[0]
    # and colorize them



fig.savefig("sep_pa.png", dpi=320)
fig.savefig("sep_pa.pdf")
