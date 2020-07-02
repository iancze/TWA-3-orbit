import os
from pathlib import Path

import exoplanet as xo
import pymc3 as pm

import model as m

if __name__ == "__main__":

    with m.model:
        trace = pm.sample(
            tune=8000,
            draws=7000,
            # start=map_sol,
            chains=4,
            cores=4,
            step=xo.get_dense_nuts_step(target_accept=0.95),
            progressbar=False,
        )

    chaindir = Path("chains")

    if not os.path.isdir(chaindir):
        os.makedirs(chaindir)

    # save the samples as a pymc3 object
    pm.save_trace(trace, directory=chaindir, overwrite=True)

    # and as a CSV, just in case the model spec
    # changes and we have trouble reloading things
    df = pm.trace_to_dataframe(trace)
    df.to_csv(chaindir / "current.csv")
