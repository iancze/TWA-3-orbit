import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator
# imported from the channel maps package we wrote
import common
from astropy.io import fits

# ran bettermoments with
# bettermoments 12CO.fits -method quadratic -clip 6 -rms 1.6e-03

# the values by which to shift to center on UZ Tau E
# measured using the dust dataset of UZTauE_final.ms and uvmodelfit with an elliptical Gaussian
mu_RA = (525 - 478.768 ) * 0.015 # x
mu_DEC = (538.165 - 525) * 0.015 # y from 0, 0
# center is 525, 525

def load_data(fname, mu_RA, mu_DEC):
    # load the dust figure
    hdul = fits.open(fname)
    data = hdul[0].data
    header = hdul[0].header

    nx = header["NAXIS1"]
    ny = header["NAXIS2"]

    assert nx % 2 == 0 and ny % 2 == 0 , "We don't have an even number of pixels, assumptions in the routine are violated."

    # RA coordinates
    CDELT1 = header["CDELT1"] # decimal degrees. Note that not all decimal degrees are the same distance on the sky
    # since this depends on what declination we're at, too!
    CRPIX1 = header["CRPIX1"] - 1. # Now indexed from 0
    # DEC coordinates
    CDELT2 = header["CDELT2"]
    CRPIX2 = header["CRPIX2"] - 1. # Now indexed from 0
    cosdec = np.cos(CDELT2 * np.pi/180)

    dRAs = (np.arange(nx) - nx/2) * CDELT1 * cosdec
    dDECs = (np.arange(ny) - ny/2) * CDELT2

    RA = 3600 * dRAs # [arcsec]
    DEC = 3600 * dDECs # [arcsec]

    ext = (RA[0] - mu_RA, RA[-1] - mu_RA, DEC[0] - mu_DEC, DEC[-1] - mu_DEC) # [arcsec]

    # Get the beam info from the header, like normal
    BMAJ = 3600 * header["BMAJ"]
    BMIN = 3600 * header["BMIN"]
    BPA = header["BPA"]

    beam = (BMAJ, BMIN, BPA)

    return (data, beam, ext)


data, beam, ext = load_data("12CO_v0.fits", mu_RA, mu_DEC)

lmargin = 0.75
rmargin = 0.75
ax_width = 2.
ax_height = 2.
bmargin = 0.5
tmargin = 0.1
cax_sep = 0.1
cax_width = 0.1


xx = lmargin + ax_width + rmargin
yy = bmargin + ax_height + tmargin

fig = plt.figure(figsize=(xx, yy))

ax = fig.add_axes([lmargin/xx, bmargin/yy, ax_width/xx, ax_height/yy])
cax = fig.add_axes([(lmargin + ax_width +  cax_sep)/xx, bmargin/yy, cax_width/xx, ax_height/yy])

ax.set_ylabel(r"$\Delta \delta\,['']$")
ax.set_xlabel(r"$\Delta \alpha \cos \delta\,['']$")

im = ax.imshow(data * 1e-3, cmap=cm.RdBu_r, origin="lower", extent=ext, aspect="auto")
plt.colorbar(im, cax=cax)


cax.set_ylabel(r'$v_\mathrm{LSRK} \quad {\rm(km\,s^{-1})}$', rotation=270., labelpad=13)

ax.xaxis.set_major_locator(MultipleLocator(0.4))
ax.yaxis.set_major_locator(MultipleLocator(0.4))

# Plot the beam
common.plot_beam(ax, beam, xy=(0.65, -0.65))

# shift the image by setting the axes limits

radius = 0.9 # arcsec
ax.set_xlim(radius, -radius)
ax.set_ylim(-radius, radius)
# we have current bounding boxes
# we have future bounding boxes plus RA shift


fig.subplots_adjust(top=0.99, left=0.15, right=0.85, bottom=0.2)
fig.savefig('first-moment.pdf')
fig.savefig('first-moment.png', dpi=300)
