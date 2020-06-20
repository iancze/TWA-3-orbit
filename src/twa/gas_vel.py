import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import LinearSegmentedColormap as LSC
from matplotlib.patches import Ellipse
from matplotlib.ticker import MultipleLocator

# ran bettermoments with
# bettermoments 12CO.fits -method quadratic -clip 6 -rms 1.6e-03


# measured using the dust dataset of UZTauE_final.ms and uvmodelfit with an elliptical Gaussian
mu_RA = 0.0  # (525 - 478.768 ) * 0.015 # x
mu_DEC = 0.0  # (538.165 - 525) * 0.015 # y from 0, 0
# center is 525, 525


def load_data(fname, mu_RA, mu_DEC):
    # load the dust figure
    hdul = fits.open(fname)
    data = hdul[0].data
    header = hdul[0].header

    nx = header["NAXIS1"]
    ny = header["NAXIS2"]

    assert (
        nx % 2 == 0 and ny % 2 == 0
    ), "We don't have an even number of pixels, assumptions in the routine are violated."

    # RA coordinates
    CDELT1 = header[
        "CDELT1"
    ]  # decimal degrees. Note that not all decimal degrees are the same distance on the sky
    # since this depends on what declination we're at, too!
    CRPIX1 = header["CRPIX1"] - 1.0  # Now indexed from 0
    # DEC coordinates
    CDELT2 = header["CDELT2"]
    CRPIX2 = header["CRPIX2"] - 1.0  # Now indexed from 0
    cosdec = np.cos(CDELT2 * np.pi / 180)

    dRAs = (np.arange(nx) - nx / 2) * CDELT1 * cosdec
    dDECs = (np.arange(ny) - ny / 2) * CDELT2

    RA = 3600 * dRAs  # [arcsec]
    DEC = 3600 * dDECs  # [arcsec]

    ext = (RA[0] - mu_RA, RA[-1] - mu_RA, DEC[0] - mu_DEC, DEC[-1] - mu_DEC)  # [arcsec]

    # Get the beam info from the header, like normal
    BMAJ = 3600 * header["BMAJ"]
    BMIN = 3600 * header["BMIN"]
    BPA = header["BPA"]

    beam = (BMAJ, BMIN, BPA)

    return (data, beam, ext)


def plot_beam(ax, beam, xy=(1, -1), facecolor="0.5", edgecolor="0.5"):
    BMAJ, BMIN, BPA = beam
    # print('BMAJ: {:.3f}", BMIN: {:.3f}", BPA: {:.2f} deg'.format(BMAJ, BMIN, BPA))
    # We need to negate the BPA since the rotation is the opposite direction
    # due to flipping RA.
    ax.add_artist(
        Ellipse(
            xy=xy,
            width=BMIN,
            height=BMAJ,
            angle=-BPA,
            facecolor=facecolor,
            linewidth=0.2,
            edgecolor=edgecolor,
        )
    )


def plot_gas(ax, cax):
    """
    Load this dataset and plot it on that ax.
    """

    # shift the image by setting the axes limits
    mu_RA = 0.0
    mu_DEC = 0.0

    radius = 2.9  # arcsec

    v_off = 9.26  # km/s

    data, beam, ext = load_data("12CO_v0.fits", mu_RA, mu_DEC)

    cmap = cm.RdBu_r
    im = ax.imshow(
        data * 1e-3 + v_off, cmap=cmap, origin="lower", extent=ext, aspect="equal"
    )

    # plot the colorbar
    plt.colorbar(im, cax=cax, orientation="horizontal")

    plot_beam(ax, beam, xy=(0.91 * radius, -0.91 * radius))

    ax.set_xlim(radius, -radius)
    ax.set_ylim(-radius, radius)
