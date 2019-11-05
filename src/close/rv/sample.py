from model import model

with model:
    map_sol = xo.optimize()

sampler = xo.PyMC3Sampler(window=100, finish=200)
with model:
    burnin = sampler.tune(tune=2500, start=map_sol, step_kwargs=dict(target_accept=0.9))
    trace = sampler.sample(draws=3000)

# write to chains directory

