import orbitize
import orbitize.driver

# tutorial here: https://github.com/sblunt/orbitize/blob/e104aebd36df1199a714327a4d531e2b6559a830/docs/tutorials/OFTI_tutorial.ipynb

myDriver = orbitize.driver.Driver("data/visual_data.csv", # path to data file
                                  'OFTI', # name of algorithm for orbit-fitting
                                  1, # number of secondary bodies in system
                                  1.00, # total system mass [M_sun]
                                  27.31, # total parallax of system [mas]
                                  mass_err=0.5, # mass error [M_sun]
                                  plx_err=0.12) # parallax error [mas]

s = myDriver.sampler
samples = s.prepare_samples(1000000)

print('samples: ', samples.shape)
print('first element of samples: ', samples[0].shape)

orbits, lnlikes = s.reject(samples)

print(orbits.shape)

# orbits = s.run_sampler(1000)
# orbits[0]

print(s.system.param_idx)

myResults = s.results

print(myResults)

# corner_figure = myResults.plot_corner(param_list=['sma1', 'ecc1', 'inc1', 'aop1', 'pan1', 'epp1'])
# corner_figure.savefig("corner.png", dpi=300)
#
# orbit_figure = myResults.plot_orbits(
#     start_mjd=s.epochs[0] # Minimum MJD for colorbar (here we choose first data epoch)
# )
# #
# orbit_figure.savefig("orbit.png", dpi=300)
