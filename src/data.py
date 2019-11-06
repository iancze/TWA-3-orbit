import astropy
import numpy as np
from astropy import constants
from astropy import units as u
from astropy.io import ascii
from astropy.time import Time


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
        date = np.ascontiguousarray(asciiTable["HJD"][mask])

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


jitter = True  # Do this to infer w/ jitter
datadir = "data/close/"

# load all the data
data_cfa = ascii.read(f"{datadir}cfa.dat")
# cfa errors are provided in table
cfa1, cfa2 = get_arrays(data_cfa, jitter=jitter)

data_keck = ascii.read(f"{datadir}keck.dat", format="tab", fill_values=[("X", 0)])
err_keck = {"Aa": 0.63, "Ab": 0.85, "B": 0.59}  # km/s
keck1, keck2 = get_arrays(data_keck, err_keck, jitter=jitter)

data_feros = ascii.read(f"{datadir}feros.dat")
err_feros = {"Aa": 2.61, "Ab": 3.59, "B": 2.60}  # km/s
feros1, feros2 = get_arrays(data_feros, err_feros, jitter=jitter)

data_dupont = ascii.read(f"{datadir}dupont.dat", fill_values=[("X", 0)])
err_dupont = {"Aa": 1.46, "Ab": 2.34, "B": 3.95}  # km/s
dupont1, dupont2 = get_arrays(data_dupont, err_dupont, jitter=jitter)

data = [data_cfa, data_keck, data_feros, data_dupont]
