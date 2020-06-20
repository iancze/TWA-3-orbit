import astropy
import theano
import pymc3 as pm
import exoplanet as xo
import numpy as np
import pandas as pd
import re

# load the exoplanet part
import theano.tensor as tt
from astropy.io import ascii
from exoplanet.distributions import Angle
from pymc3.distributions import Interpolated  # to make the disk KDE


from src.constants import *
import src.data as d

from src.close.rv_astro_more.model import *

# flip the covariance of the inclination samples, i.e., rederive it 
diskdir = "data/disk/"
flatchain = np.load(f"{diskdir}flatchain.npy")
disk_samples = d.flatchain[:, [0,9,10]] 
disk_samples[:,2] -= 90.0 # convert Omega conventions 
disk_samples[:,1] = 180. - disk_samples[:,1] # make > 90
disk_samples[:,[1,2]] *= deg # convert *to* radians
mass_samples, incl_samples, Omega_samples = disk_samples.T

# can we evaluate the multivariate normal approximation to these correlations?
disk_mu = np.mean(disk_samples, axis=0)
disk_cov = np.cov(disk_samples, rowvar=False)

# rather than completely reimplement model, just extend it
with model:
    # evaluate the disk likelihood on M_Aa + M_Ab
    # disk mass marginalized over all structure params (including i_disk and Omega_disk)
    # pm.Normal("obs_MA", mu=MA, observed=MA_mu, sd=MA_std)

    # MA is generated from the orbit
    # generate i_disk from the range of samples
    i_disk = pm.Uniform(
        "iDisk",
        lower=np.min(incl_samples),
        upper=np.max(incl_samples),
        testval=np.mean(incl_samples),
    )

    # generate Omega_disk from range of samples
    Omega_disk = pm.Uniform(
        "OmegaDisk",
        lower=np.min(Omega_samples),
        upper=np.max(Omega_samples),
        testval=np.mean(Omega_samples),
    )

    disk_observed = tt.as_tensor_variable([MA, i_disk, Omega_disk])

    pm.MvNormal("obs_disk", mu=disk_mu, cov=disk_cov, observed=disk_observed)

    # calculate the mutual inclination as well
    theta = pm.Deterministic("thetaADisk", tt.arccos(tt.cos(i_disk)*tt.cos(incl) + tt.sin(i_disk) * tt.sin(incl) * tt.cos(Omega_disk - Omega)))

# iterate through the list of free_RVs in the model to get things like
# ['logKAa_interval__', etc...] then use a regex to strip away
# the transformations (in this case, _interval__ and _angle__)
# \S corresponds to any character that is not whitespace
# https://docs.python.org/3/library/re.html
sample_vars = [re.sub("_\S*__", "", var.name) for var in model.free_RVs]

all_vars = [
    var.name
    for var in model.unobserved_RVs
    if ("_interval__" not in var.name)
    and ("_angle__" not in var.name)
    and ("_lowerbound__" not in var.name)
]