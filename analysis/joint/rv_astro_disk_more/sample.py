import pymc3 as pm
import exoplanet as xo
import os 

import src.joint.rv_astro_disk_more.model as m


with m.model:
    trace = pm.sample(
        tune=6000,
        draws=3000,
        # start=map_sol,
        chains=4,
        step=xo.get_dense_nuts_step(target_accept=0.9),
    )


chaindir = "chains/joint/rv_astro_disk_more/"

if not os.path.isdir(chaindir):
    os.makedirs(chaindir)

# save the samples as a pymc3 object
pm.save_trace(trace, directory=chaindir, overwrite=True)

# and as a CSV, just in case the model spec
# changes and we have trouble reloading things
df = pm.trace_to_dataframe(trace)
df.to_csv(f"{chaindir}current.csv")