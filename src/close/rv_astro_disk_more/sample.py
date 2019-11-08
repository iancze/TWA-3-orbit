import pymc3 as pm
import exoplanet as xo

import src.close.rv_astro_disk_more.model as m

# with m.model:
#     map_sol = xo.optimize(vars=[m.a_ang, m.MAb, m.incl])
#     print(map_sol) 
#     map_sol1 = xo.optimize(start=map_sol, vars=[m.incl, m.Omega])
#     print(map_sol1)


with m.model:
    trace = pm.sample(
        tune=2500,
        draws=3000,
        # start=map_sol,
        chains=4,
        step=xo.get_dense_nuts_step(target_accept=0.9),
    )


# save the samples as a pymc3 object
pm.save_trace(trace, directory="chains/close/rv_astro_disk_more", overwrite=True)

# and as a CSV, just in case the model spec
# changes and we have trouble reloading things
df = pm.trace_to_dataframe(trace)
df.to_csv(f"chains/close/rv_astro_disk_more/current.csv")