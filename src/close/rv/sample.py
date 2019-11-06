import pymc3 as pm
import exoplanet as xo
from src.close.rv.model import model

with model:
    map_sol = xo.optimize()

with model:
    trace = pm.sample(
        tune=2500,
        draws=3000,
        start=map_sol,
        chains=4,
        step=xo.get_dense_nuts_step(target_accept=0.9),
    )

# save the samples as a pymc3 object
pm.save_trace(trace, directory="chains/close/rv", overwrite=True)

# and as a CSV, just in case the model spec
# changes and we have trouble reloading things
df = pm.trace_to_dataframe(trace)
df.to_csv(f"chains/close/rv/current.csv")

