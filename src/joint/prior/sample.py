import pymc3 as pm
import pandas as pd
import exoplanet as xo
import os 

import src.joint.prior.model as m

with m.model:
    sample_dict = pm.sample_prior_predictive(samples=10000, var_names=m.all_vars)

chaindir = "chains/joint/prior/"

if not os.path.isdir(chaindir):
    os.makedirs(chaindir)


# we can't directly convert this to a DataFrame because it contains some of the bivariaate samples 

# export trace to pandas dataframe 
df = pd.DataFrame.from_dict(sample_dict)
# # and as a CSV, just in case the model spec
# # changes and we have trouble reloading things
df.to_csv(f"{chaindir}current.csv")

# import matplotlib.pyplot as plt  
# plt.hist(trace["thetaInnerOuter"])
# plt.show()

# # # save the samples as a pymc3 object
# # pm.save_trace(trace, directory=chaindir, overwrite=True)

