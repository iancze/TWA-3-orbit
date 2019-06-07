import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams["text.usetex"] = False

from astropy import constants
import corner

deg = np.pi/180. # radians / degree
yr = 365.25 # days / year

au_to_R_sun = (constants.au / constants.R_sun).value # conversion constant

df = pd.read_csv("current.csv")

print(df.columns)

# fig = corner.corner(df)
# fig.savefig("corner-all.png", dpi=120)

# all_params = ['MAa', 'MAb', 'MA', 'MB', 'Mtot', 'offsetKeck', 'offsetFeros', 'offsetDupont', 'logRhoS', 'logThetaS', 'parallax',     'a_inner', 'P_inner', 'e_inner', 'omega_inner', 'Omega_inner', 'incl_inner',
#        't_periastron_inner', 'P_outer', 'omega_outer',
#        'Omega_outer', 't_periastron_outer', 'incl_outer', 'e_outer', 'gamma_outer', , 'a_outer',
#        'a_ang_outer', 'logjittercfa', 'logjitterkeck', 'logjitterferos',
#        'logjitterdupont', 'jitCfa', 'jitKeck', 'jitFeros', 'jitDupont'

# convert all params
df["omega_inner"] /= deg
df["Omega_inner"] /= deg
df["incl_inner"] /= deg

df["omega_outer"] /= deg
df["Omega_outer"] /= deg
df["incl_outer"] /= deg

df["P_outer"] /= yr

# just the inner parameters
inner = ['a_inner', 'P_inner', 'e_inner', 'omega_inner', 'Omega_inner', 'incl_inner', 't_periastron_inner']
df_inner = df[inner]
fig = corner.corner(df_inner)
fig.savefig("corner-inner.png", dpi=120)

# just the outer parameters
outer = ['MA', 'MB', 'a_outer', 'P_outer', 'omega_outer', 'Omega_outer', 'e_outer', 'incl_outer', 'gamma_outer', 't_periastron_outer']
df_outer = df[outer]
fig = corner.corner(df_outer)
fig.savefig("corner-outer.png", dpi=120)


# masses
masses = ['MAa', 'MAb', 'MA', 'MB', 'Mtot']
fig = corner.corner(df[masses])
fig.savefig("corner-masses.png", dpi=120)
