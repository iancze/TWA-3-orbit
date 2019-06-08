import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import rcParams
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
# a draw only, since P and offsets don't scatter well in this space

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

df = pd.read_csv("current2.csv")

np.random.seed(42)
# also choose a sample at random and use the starting position
sample_pars = ['mparallax', 'MAb', 'a_ang_inner', 'logP_inner', 'e_inner', 'omega_inner', 'Omega_inner', 'cos_incl_inner', 't_periastron_inner', 'logP_outer', 'omega_outer', 'Omega_outer', 'phi_outer', 'cos_incl_outer', 'e_outer', 'gamma_outer', 'MB']

row0 = df.sample()
point = {par:row0[par].item() for par in sample_pars}

P_inner = np.exp(point["logP_inner"]) # days

def get_phase(dates):
    return ((dates - point["t_periastron_inner"]) % P_inner) / P_inner

lmargin = 0.5
rmargin = lmargin
bmargin = 0.4
tmargin = 0.05
mmargin = 0.07
mmmargin = 0.15

ax_height = 1.0
ax_r_height = 0.3

xx = 3.5
ax_width = xx - lmargin - rmargin

yy = bmargin + tmargin + 2 * mmargin + mmmargin + 2 * ax_height + 2 * ax_r_height

pkw = {"marker":".", "ls":""}
ekw = {"marker":".", "ms":5.0, "ls":"", "elinewidth":1.2}

xs_phase = np.linspace(0, 1, num=500)
ts_phases = xs_phase * P_inner + point["t_periastron_inner"]
point['times'] = ts_phases

fig = plt.figure(figsize=(xx,yy))

ax1 = fig.add_axes([lmargin/xx, 1 - (tmargin + ax_height)/yy, ax_width/xx, ax_height/yy])
ax1.plot(xs_phase, f_rv_Aa(**point))

ax1_r = fig.add_axes([lmargin/xx, 1 - (tmargin + mmargin + ax_height + ax_r_height)/yy, ax_width/xx, ax_r_height/yy])

ax2 = fig.add_axes([lmargin/xx, (bmargin + mmargin + ax_r_height)/yy, ax_width/xx, ax_height/yy])
ax2.plot(xs_phase, f_rv_Ab(**point))

ax2_r = fig.add_axes([lmargin/xx, bmargin/yy, ax_width/xx, ax_r_height/yy])

def plot_data(ax, ax_r, offset_label):
    # get the data, apply offset, and plot residuals for instruments
    pass

ax1.set_ylabel(r"$v_\mathrm{Aa}$ [$\mathrm{km s}^{-1}$]")
ax1_r.set_ylabel("O-C")

ax2.set_ylabel(r"$v_\mathrm{Ab}$ [$\mathrm{km s}^{-1}$]")
ax2_r.set_ylabel("O-C")
ax2_r.set_xlabel("phase")
fig.savefig("inner_RV.pdf")
fig.savefig("inner_RV.png", dpi=300)

# point["times"] = np.atleast_1d(anthonioz[0])
# rho_inner, theta_inner = f_sky_inner(**point)
# print(rho_inner, theta_inner)

# # get the total errors
# def get_err(rv_err, logjitter):
#     return tt.sqrt(rv_err**2 + tt.exp(2*logjitter))

#
#     # RV predictions
#     t_dense_RV = xs_phase * P_inner + t_periastron_inner
#
#     rv1, rv2 = get_RVs(t_dense_RV, t_dense_RV, 0.0)
#
#     rv1_dense = pm.Deterministic("RV1Dense", rv1)
#     rv2_dense = pm.Deterministic("RV2Dense", rv2)
#
#     rho_outer, theta_outer = orbit_outer.get_relative_angles(wds[0], parallax) # arcsec
#
#     # add jitter terms to both separation and position angle
#     log_rho_s = pm.Normal("logRhoS", mu=np.log(np.median(wds[2])), sd=2.0)
#     log_theta_s = pm.Normal("logThetaS", mu=np.log(np.median(wds[4])), sd=2.0)
#
#     rho_tot_err = tt.sqrt(wds[2]**2 + tt.exp(2*log_rho_s))
#     theta_tot_err = tt.sqrt(wds[4]**2 + tt.exp(2*log_theta_s))
##
#     # save some samples on a fine orbit for plotting purposes
#     t_period = pm.Deterministic("tPeriod", xs_phase * P_outer)
#
#     rho, theta = orbit_outer.get_relative_angles(t_period, parallax)
#     rho_save_sky = pm.Deterministic("rhoSaveSky", rho)
#     theta_save_sky = pm.Deterministic("thetaSaveSky", theta)
#
#     rho, theta = orbit_outer.get_relative_angles(t_data, parallax)
#     rho_save_data = pm.Deterministic("rhoSaveData", rho)
#     theta_save_data = pm.Deterministic("thetaSaveData", theta)
#
#     rvA_dense = pm.Deterministic("RVADense", get_gamma_A(t_period))
#     rvB_dense = pm.Deterministic("RVBDense", conv * orbit_outer.get_planet_velocity(t_period)[2] +                                  gamma_outer + offset_keck)
#
#
#
# # phase inner orbit
#
# # plot everything ontop in a single plot
#
# pkw = {"marker":".", "ls":""}
# ekw = {"marker":".", "ms":5.0, "ls":"", "elinewidth":1.2}
#
#
# # nsamples = 10
# # choices = np.random.choice(np.arange(len(trace)), size=nsamples)
#
# # just choose one representative sample
# np.random.seed(43)
# # choice = np.random.choice(np.arange(len(trace)))
#
# fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8,4))
#
# fig_sep, ax_sep = plt.subplots(nrows=4, sharex=True, figsize=(6,8))
# fig_sky, ax_sky = plt.subplots(nrows=1, figsize=(4,4))
#
#
# with model:
#
#     pos = map_sol3
# #     pos = model.test_point
#
#     # calculate the errors for each instrument
#     cfa_err1 = np.sqrt(cfa1[2]**2 + np.exp(2 * pos["logjittercfa"]))
#     cfa_err2 = np.sqrt(cfa2[2]**2 + np.exp(2 * pos["logjittercfa"]))
#
#     keck_err1 = np.sqrt(keck1[2]**2 + np.exp(2 * pos["logjitterkeck"]))
#     keck_err2 = np.sqrt(keck2[2]**2 + np.exp(2 * pos["logjitterkeck"]))
#
#     feros_err1 = np.sqrt(feros1[2]**2 + np.exp(2 * pos["logjitterferos"]))
#     feros_err2 = np.sqrt(feros2[2]**2 + np.exp(2 * pos["logjitterferos"]))
#
#     dupont_err1 = np.sqrt(dupont1[2]**2 + np.exp(2 * pos["logjitterdupont"]))
#     dupont_err2 = np.sqrt(dupont2[2]**2 + np.exp(2 * pos["logjitterdupont"]))
#
#
#     ax[0].plot(xs_phase, pos["RV1Dense"])
#     ax[1].plot(xs_phase, pos["RV2Dense"])
#
#     # at data locations
#     ax[0].errorbar(get_phase(cfa1[0], pos), cfa1[1], yerr=cfa_err1, **ekw, zorder=0)
#     ax[0].errorbar(get_phase(keck1[0], pos), keck1[1] - pos["offsetKeck"], yerr=keck_err1, **ekw, zorder=0)
#     ax[0].errorbar(get_phase(feros1[0], pos), feros1[1] - pos["offsetFeros"], yerr=feros_err1, **ekw, zorder=0)
#     ax[0].errorbar(get_phase(dupont1[0], pos), dupont1[1] - pos["offsetDupont"], yerr=dupont_err1, **ekw, zorder=0)
#
#     # at data locations
#     ax[1].errorbar(get_phase(cfa2[0], pos), cfa2[1], yerr=cfa_err2, **ekw, zorder=0)
#     ax[1].errorbar(get_phase(keck2[0], pos), keck2[1] - pos["offsetKeck"], yerr=keck_err2, **ekw, zorder=0)
#     ax[1].errorbar(get_phase(feros2[0], pos), feros2[1] - pos["offsetFeros"], yerr=feros_err2, **ekw, zorder=0)
#     ax[1].errorbar(get_phase(dupont2[0], pos), dupont2[1] - pos["offsetDupont"], yerr=dupont_err2, **ekw, zorder=0)
#
#     ax[1].set_xlim(0.0, 1.0)
#     ax[0].set_ylabel(r"$v_\mathrm{Aa}$ $[\mathrm{km s}^{-1}]$")
#     ax[1].set_ylabel(r"$v_\mathrm{Ab}$ $[\mathrm{km s}^{-1}]$")
#     ax[1].set_xlabel("phase")
#
#
#     t_pred = pos["tPeriod"]
#     rho_pred = pos["rhoSaveSky"]
#     theta_pred = pos["thetaSaveSky"]
#
#     x_pred = rho_pred * np.cos(theta_pred) # X north
#     y_pred = rho_pred * np.sin(theta_pred) # Y east
#
#     ax_sky.plot(y_pred, x_pred, lw=0.8, alpha=0.7)
#
#
#     ax_sep[0].plot(t_data, pos["rhoSaveData"], lw=0.8, alpha=0.7, zorder=0)
#     ax_sep[0].plot(wds[0], wds[1], "ko")
#
#     ax_sep[1].plot(t_data, pos["thetaSaveData"], lw=0.8, alpha=0.7, zorder=0)
#     ax_sep[1].plot(wds[0], wds[3], "ko")
#
#     ax_sep[2].plot(t_pred, pos["RVADense"], lw=0.5, alpha=0.6)
#     ax_sep[3].plot(t_pred, pos["RVBDense"], lw=0.5, alpha=0.6)
# # #     ax_sep[2].set_ylim(7, 10)
#
#     ax_sep[3].set_xlim(- 200, wds[0][-1] + 200)
# # #     ax_sep[3].set_ylim(6.5, 8)
#
#
#
# xs = wds[1] * np.cos(wds[3]) # X is north
# ys = wds[1] * np.sin(wds[3]) # Y is east
#
# ax_sep[0].set_ylabel(r"$\rho$ ['']")
# ax_sep[1].set_ylabel("P.A. ['']")
# ax_sep[2].set_ylabel(r"$v_A$ [km/s]")
# ax_sep[3].set_ylabel(r"$v_B$ [km/s]")
#
# ax_sky.plot(ys, xs, "ko")
# ax_sky.set_ylabel(r"$\Delta \delta$ ['']")
# ax_sky.set_xlabel(r"$\Delta \alpha \cos \delta$ ['']")
# ax_sky.invert_xaxis()
# ax_sky.plot(0,0, "k*")
# ax_sky.set_aspect("equal", "datalim")
#
# # fig.subplots_adjust(top=0.98, bottom=0.18, hspace=0.05)
# fig.savefig("A_sb_orbit.pdf")
# fig_sky.savefig("sky.pdf")
# fig_sep.savefig("sep.pdf")
#
#
# pm.summary(trace)
#
#
# vars = ["MAa", "MAb", "MA", "MB", "Mtot", "t_periastron_inner", "gamma_outer", "a_ang_inner", "logP_inner", "omega_inner",
#         "Omega_inner", "cos_incl_inner", "e_inner", "phi_outer", "omega_outer", "Omega_outer",
#         "cos_incl_outer", "e_outer"]
# pm.traceplot(trace, varnames=vars)
