import pymc3 as pm
import exoplanet as xo
import os

import src.wide.astro.model as m

with m.model:
    trace = pm.sample(
        tune=2000,
        draws=3000,
        # start=map_sol,
        chains=4,
        step=xo.get_dense_nuts_step(target_accept=0.9),
    )

chaindir = "chains/wide/astro/"

if not os.path.isdir(chaindir):
    os.makedirs(chaindir)

# save the samples as a pymc3 object
pm.save_trace(trace, directory=chaindir, overwrite=True)

# and as a CSV, just in case the model spec
# changes and we have trouble reloading things
df = pm.trace_to_dataframe(trace)
df.to_csv(f"{chaindir}current.csv")

# # get the MAP solution to start
# with parallax_model:
#     map_sol0 = xo.optimize(vars=[a_ang, phi])
#     map_sol1 = xo.optimize(map_sol0, vars=[a_ang, phi, omega, Omega])
#     map_sol2 = xo.optimize(map_sol1, vars=[a_ang, logP, phi, omega, Omega, incl, e])
#     map_sol3 = xo.optimize(map_sol2)
