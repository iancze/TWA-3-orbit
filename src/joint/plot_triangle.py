import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import constants
from matplotlib import rcParams

rcParams["text.usetex"] = False


outdir = "disk/"

deg = np.pi / 180.0  # radians / degree
yr = 365.25  # days / year

au_to_R_sun = (constants.au / constants.R_sun).value  # conversion constant

df = pd.read_csv(f"{outdir}current.csv")

print(df.columns)

# plot the raw parameters
sample_pars = [
    "mparallax",
    "MAb",
    "a_ang_inner",
    "logP_inner",
    "e_inner",
    "omega_inner",
    "Omega_inner",
    "cos_incl_inner",
    "t_periastron_inner",
    "logP_outer",
    "omega_outer",
    "Omega_outer",
    "phi_outer",
    "cos_incl_outer",
    "e_outer",
    "gamma_outer",
    "MB",
    "offsetKeck",
    "offsetFeros",
    "offsetDupont",
    "logRhoS",
    "logThetaS",
    "logjittercfa",
    "logjitterkeck",
    "logjitterferos",
    "logjitterdupont",
]

# also choose a sample at random and use the starting position
row0 = df.sample()
for par in sample_pars:
    print("{:} : {:}".format(par, row0[par].values[0]))


fig = corner.corner(df[sample_pars])
fig.savefig(f"{outdir}corner-sample-pars.png", dpi=120)

# convert all params
df["omega_inner"] /= deg
df["Omega_inner"] /= deg
df["incl_inner"] /= deg

df["omega_outer"] /= deg
df["Omega_outer"] /= deg
df["incl_outer"] /= deg

df["P_outer"] /= yr

# just the inner parameters
inner = [
    "MAb",
    "MA",
    "a_inner",
    "P_inner",
    "e_inner",
    "omega_inner",
    "Omega_inner",
    "incl_inner",
    "t_periastron_inner",
]
fig = corner.corner(df[inner])
fig.savefig(f"{outdir}corner-inner.png", dpi=120)

# just the outer parameters
outer = [
    "MA",
    "MB",
    "a_outer",
    "P_outer",
    "omega_outer",
    "Omega_outer",
    "e_outer",
    "incl_outer",
    "gamma_outer",
    "t_periastron_outer",
]
fig = corner.corner(df[outer])
fig.savefig(f"{outdir}corner-outer.png", dpi=120)

# masses
masses = ["MAa", "MAb", "MA", "MB", "Mtot"]
fig = corner.corner(df[masses])
fig.savefig(f"{outdir}corner-masses.png", dpi=120)


# mutual inclination between inner orbit and outer orbit
