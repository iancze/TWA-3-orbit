import numpy as np
from astropy import constants
from astropy import units as u

deg = np.pi / 180.0  # radians / degree
yr = 365.25  # days / year

au_to_R_sun = (constants.au / constants.R_sun).value

# convert from R_sun / day to km/s
# and from v_r = - v_Z
output_units = u.km / u.s
conv = -(1 * u.R_sun / u.day).to(output_units).value

# the reference epoch to subtract from all dates
jd0 = 2451545.0  # Jan 1st, 2000, 12pm noon


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
    """
    Calculate the semi-major axis using Kepler's third law

    Args:
        Mtot (Msun) total mass
        P (days) period

    Returns:
        a (au)
    """

    day_to_s = (1 * u.day).to(u.s).value
    au_to_m = (1 * u.au).to(u.m).value
    kg_to_M_sun = (1 * u.kg).to(u.M_sun).value

    return (
        ((Mtot / kg_to_M_sun) * constants.G.value * (P * day_to_s) ** 2)
        / (4 * np.pi ** 2)
    ) ** (1 / 3) / au_to_m

