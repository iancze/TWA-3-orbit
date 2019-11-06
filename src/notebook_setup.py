# Hide deprecation warnings from Theano
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Hide Theano compilelock warnings
import logging

logger = logging.getLogger("theano.gof.compilelock")
logger.setLevel(logging.ERROR)

import theano

print("theano version: {0}".format(theano.__version__))

import pymc3

print("pymc3 version: {0}".format(pymc3.__version__))

import exoplanet

print("exoplanet version: {0}".format(exoplanet.__version__))
