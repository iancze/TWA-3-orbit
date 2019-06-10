import numpy as np
from astropy.io import fits

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# imported from the channel maps package we wrote
import common

# the values by which to shift to center on UZ Tau E
# measured using the dust dataset of UZTauE_final.ms and uvmodelfit with an elliptical Gaussian
# mu_RA = 0.7 # arcsec
# mu_DEC = 0.20 # arcsec

mu_RA = (525 - 478.768 ) * 0.015 # x
mu_DEC = (538.165 - 525) * 0.015 # y from 0, 0
# center is 525, 525

# pixels are 0.015 arcsec


# little convenience helper for loading the fits files
def load_data(fname):
    # load the dust figure
    hdul = fits.open(fname)
    data = hdul[0].data
    header = hdul[0].header

    # get the RA and DEC coords for the dust
    dict = common.get_coords(data, header, radius=3.0/3600, mu_RA=mu_RA, mu_DEC=mu_DEC)

    RA = 3600 * dict["RA"] # [arcsec]
    DEC = 3600 * dict["DEC"] # [arcsec]
    decl, decr = dict["DEC_slice"]
    ral, rar = dict["RA_slice"]
    data = dict["data"] # sliced data

    # get rid of the stokes and frequency channels
    # data = data[0,0]

    ext = (RA[0], RA[-1], DEC[0], DEC[-1]) # [arcsec]

    # Get the beam info from the header, like normal
    BMAJ = 3600 * header["BMAJ"]
    BMIN = 3600 * header["BMIN"]
    BPA = header["BPA"]

    beam = (BMAJ, BMIN, BPA)

    return (data, beam, ext)

cmap = plt.get_cmap("inferno")

fig, ax_dust = plt.subplots(nrows=1, figsize=(3.5,3.0))

# load the dust data
data_dust, beam, ext = load_data("cont.fits")

# print(data_dust)

# equal aspect means that the sky is square
im = ax_dust.imshow(data_dust, cmap=cmap, origin="lower", extent=ext, aspect="equal")
common.plot_beam(ax_dust, beam, xy=(2.5,-2.5), facecolor="0.5", edgecolor="0.5")

ax_dust.plot(0, 0, "k*", ms=3, mec="k", mew=0.2)
ax_dust.set_ylabel(r"$\Delta \delta$ ['']")
ax_dust.set_xlabel(r"$\Delta \alpha \cos \delta$ ['']")

fig.subplots_adjust(bottom=0.18)

fig.savefig("dust.pdf")


fig, ax_gas = plt.subplots(nrows=1, figsize=(3.5,3.0))

# load the dust data
data_gas, beam, ext = load_data("12CO_moments.integrated.fits")

print(data_gas.shape)
# equal aspect means that the sky is square
im = ax_gas.imshow(data_gas, cmap=cmap, origin="lower", extent=ext, aspect="equal")
common.plot_beam(ax_gas, beam, xy=(2.5,-2.5), facecolor="0.5", edgecolor="0.5")

ax_gas.plot(0, 0, "k*", ms=3, mec="k", mew=0.2)
ax_gas.set_ylabel(r"$\Delta \delta$ ['']")
ax_gas.set_xlabel(r"$\Delta \alpha \cos \delta$ ['']")

fig.subplots_adjust(bottom=0.18)

fig.savefig("gas.pdf")
