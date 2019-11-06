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
