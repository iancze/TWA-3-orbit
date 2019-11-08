import astropy
import numpy as np
from astropy import constants
from astropy import units as u
from astropy.io import ascii
from astropy.time import Time

from src.constants import *

jitter = True  # Do this to infer w/ jitter
closedir = "data/close/"
widedir = "data/wide/"
diskdir = "data/disk/"


def get_arrays(asciiTable, errDict=None, jitter=False):
    """
    Reformat ascii tables into pure numpy arrays of the right dimension.
    """

    output = []

    for star in ["Aa", "Ab"]:

        # get the RVs
        rv = asciiTable["RV_" + star]

        if type(rv) is astropy.table.column.MaskedColumn:
            mask = ~rv.mask  # values we want to keep when indexing
        else:
            mask = np.ones(len(rv), dtype="bool")

        rv = np.ascontiguousarray(rv[mask])
        date = np.ascontiguousarray(asciiTable["HJD"][mask]) + 2400000 - jd0

        if errDict is None:
            err = np.ascontiguousarray(asciiTable["sigma_" + star][mask])
        else:
            err = np.ones(len(date), dtype=np.float64) * errDict[star]

        if jitter:
            err = (
                np.ones(len(date), dtype=np.float64) * 0.1
            )  # [km/s] assume a small error, since we'll infer.

        assert len(date) == len(rv), "date - rv length mismatch"
        assert len(date) == len(err), "date - err length mismatch"

        tup = (date, rv, err)

        output.append(tup)

    return output


# load all of the RV data
data_cfa = ascii.read(f"{closedir}cfa.dat")
# cfa errors are provided in table
cfa1, cfa2 = get_arrays(data_cfa, jitter=jitter)

data_keck = ascii.read(f"{closedir}keck.dat", format="tab", fill_values=[("X", 0)])
err_keck = {"Aa": 0.63, "Ab": 0.85, "B": 0.59}  # km/s
keck1, keck2 = get_arrays(data_keck, err_keck, jitter=jitter)

data_feros = ascii.read(f"{closedir}feros.dat")
err_feros = {"Aa": 2.61, "Ab": 3.59, "B": 2.60}  # km/s
feros1, feros2 = get_arrays(data_feros, err_feros, jitter=jitter)

data_dupont = ascii.read(f"{closedir}dupont.dat", fill_values=[("X", 0)])
err_dupont = {"Aa": 1.46, "Ab": 2.34, "B": 3.95}  # km/s
dupont1, dupont2 = get_arrays(data_dupont, err_dupont, jitter=jitter)

rv_data = [data_cfa, data_keck, data_feros, data_dupont]


# load the Anthonioz astrometric data

# keep in mind that the primary and secondary stars *could* be switched
# separation is in milliarcseconds
int_data = ascii.read(f"{closedir}int_data.dat")

astro_jd = int_data["epoch"][0] - jd0
rho_data = int_data["sep"][0] * 1e-3  # arcsec
rho_err = int_data["sep_err"][0] * 1e-3  # arcsec
theta_data = int_data["pa"][0] * deg  # radians
theta_err = int_data["pa_err"][0] * deg  # radians

anthonioz = (astro_jd, rho_data, rho_err, theta_data, theta_err)


# load the wide orbit astrometric dataset
data = ascii.read(
    f"{widedir}visual_data_besselian.csv", format="csv", fill_values=[("X", "0")]
)

# convert years
jds = Time(np.ascontiguousarray(data["epoch"]), format="byear").jd - jd0

data["rho_err"][data["rho_err"].mask == True] = 0.05
data["PA_err"][data["PA_err"].mask == True] = 5.0

# convert all masked frames to be raw np arrays, since theano has issues with astropy masked columns
rho_data = np.ascontiguousarray(data["rho"], dtype=float)  # arcsec
rho_err = np.ascontiguousarray(data["rho_err"], dtype=float)

# the position angle measurements come in degrees in the range [0, 360].
# we need to convert this to radians in the range [-pi, pi]
theta_data = np.ascontiguousarray(data["PA"] * deg, dtype=float)
theta_data[theta_data > np.pi] -= 2 * np.pi

theta_err = np.ascontiguousarray(data["PA_err"] * deg)  # radians

wds = (jds, rho_data, rho_err, theta_data, theta_err)

# load the disk constraints
flatchain = np.load(f"{diskdir}flatchain.npy")
disk_samples = flatchain[:, [0,9,10]] 
disk_samples[:,2] -= 90.0 # convert conventions 
disk_samples[:,[1,2]] *= deg # convert *to* radians
mass_samples, incl_samples, Omega_samples = disk_samples.T

disk_properties = {
    "MA": (np.mean(mass_samples), np.std(mass_samples)),
    "incl": (np.mean(incl_samples), np.std(incl_samples)),
    "Omega": (np.mean(Omega_samples), np.std(Omega_samples)),
}

# can we evaluate the multivariate normal approximation to these correlations?
disk_mu = np.mean(disk_samples, axis=0)
disk_cov = np.cov(disk_samples, rowvar=False)