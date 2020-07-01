# TWA-3-orbit

## Installation 

### Pip

It helps to install this locally to a conda environment. In my case, I did

    $ python3 -m venv venv
    $ source venv/bin/activate

Then the package can be installed locally. 

    $ pip install -e .

The requirements can be installed via 

    $ pip install -r requirements.txt

Then install exoplanet

    $ pip install -U exoplanet        

This should keep the whole directory structure much more organized and portable than before.

### Conda 

Try the environment file.

## Analysis Directory

There are three top-level directories for the close, wide, and joint analysis paths. These are outside of the package itself, and are the scripts we run to launch the analysis for each sub-model. The scripts are launched from within those directories. Each orbital scenario typically has a `model.py`, `sample.py` and `plot.py` code in it.


## Orbital fits

Determining the architecture of the TWA 3 hierarchical triple system using a combination of ALMA CO 2-1 observations, radial velocity, and astrometric data. 

Building up to the joint fit, we have also made fits to smaller portions of the data. Each of these can be reproduced from the Makefile.

1. [Done, TWA 3A RV only] Aa-Ab tight binary orbit fit with the double-lined RV dataset. Constrains `P_A`, `K_Aa`, `K_Ab`, `e_A`, `omega_Aa`, `gamma_A`, and `phi_A`, the phase of the orbit measured at some reference epoch. We pretty much reproduce Kellogg et al. results. Some different RV jitter results in general.
2. [Done, TWA 3A RV + astrometry] Aa-Ab tight binary orbit fit with double-lined RVs, the astrometric constraint from Anthonioz15, and the *Gaia* parallax. Constrains the above RV parameters plus `a_A`, `i_A` and `Omega_Aa`. From this, we can derive a posterior on `M_A` and see what it looks like without the disk results. A little tricky to sample in `cos_incl`, since there are two families of < 90 and > 90. Omega, however, is constrained because of the RV (we know which way the secondary star is moving at the epoch of astrometric measurement). Technically, we should also explore a fit where the position angle + pi, since the primary and secondary could be swapped in the Anthonioz dataset. Looks like a good fit, though, with stellar masses indicative of mid-M dwarfs. A 5% dynamical mass will really help the inclination measurement.
3. [Done, TWA 3A RV+astro+disk] Aa-Ab tight binary orbit fit with double-lined RVs, astrometric constraint, parallax, and priors on `M_A` and `gamma_A` from millimeter. The mass constraint on `M_A` does in fact really tighten things up for inclination, but at least in this space, there are still two solutions of i < 90 and > 90. The value of Omega changes *slightly* for each one (just by like 5 degrees). So, comparing this with the disk measurement and calculating which is in fact the lower inclination solution would be interesting.
4. [Done] A-B wide binary orbit fit with the astrometric dataset (WDS) and *Gaia* parallax. Constrains `P_AB`, `a_AB`, `Omega_AB`, `omega_AB`, `e_AB`, `phi_AB`. 
5. [Done] A-B wide binary orbit fit with astrometric dataset and RV information from Keck double-lined solution. Constrains the above astrometric parameters plus stellar masses. If we treat the Keck points as increasing in a statistically significant manner, then this breaks the Omega degeneracy in such a way that the outer triple is at least moderately aligned (if not coplanar) with the inner binary. If we don't assume this, then there is still a degeneracy.
6. [Done] Hierarchical triple orbit simultaneously fit with RV and astrometry for both tight inner binary and wide outer binary: inner: `parallax`, `P_A`, `a_A_ang`, `M_Ab`, `e_A`, `i_A`, `omega_Aa`, `Omega_Aa`,  outer: `P_B`, `a_B_ang`, `e_AB`, `i_AB`, `omega_A`, `Omega_A`, `gamma_AB`. `M_A` is derived from inner orbit and fed to outer orbit.  `gamma_A` is essentially the RV prediction of A, and is derived from outer orbit and fed to the inner orbit. This has 15 orbital parameters. Adding 4 RV offsets, 2 * 4 RV jitter terms, and 2 astrometric jitter terms makes it 30 parameters total.  
7. [Done] Hierarchical triple orbit simultaneously fit including dynamical mass prior on `M_A`.
8. [Done] Same, but now including disk orbit normal (evaluated as multi-dimenisonal Gaussian  from dynamical modeling and directly calculating mutual inclinations for all angles. 


Calculating mutual inclinations *without* the isotropic prior. It would be a good idea to sample from the prior distributions that we have just to demonstrate what the natural phase-space prior is on `theta`.

## Cataloguing the orbital ambiguities 
After completing the disk + rv + astro fits, there are still a few orbital ambiguities that should be addressed. If you want to stretch the data, I would say that we actually have leverage on all of these quantities. However, there are some tricky parts in interpreting the data that we need to be careful to consider.

### i_A (inner binary inclination) > 90 or < 90
From the inner orbit + Anthonioz point + disk-based mass on M_A, there are two degenerate solutions for i_A (below and above 90), each of which has a *slightly* different Omega value. These sub-corner plots are plotted in the IPython notebook for this fit.

    incl: 48.75 +\- 0.89
    Omega: 112.10 +\- 9.46

    incl: 131.25 +\- 0.90
    Omega: 104.44 +\- 9.41

### i_disk > 90 or < 90
In theory, we should be able to tell from the sub-mm emission alone which side of the disk is near. However, the CO emission is sufficiently faint, and we've already invoked a non-standard Nuker profile, that I'm not sure I can do this without error. If I had to guess from the emission alone, it seems like the Southern lobe is the far side, which implies i_disk < 90. This is rather striking because then the disk and the outer binary are retrograde.

There seems like there is the brightness asymmetry for the figure-8 and there is the brightness asymmetry in the C-shape. I should look at well-resolved disks like HD163296 to ascertain which side is near/far. One additional check we can do is that if the binary and disk are coplanar, then Omega_disk should match Omega_binary. 

From the Nuker fit, Omega_disk = 117 +/- 1 degr. If we assume that the disk and binary are coplanar, then this would seem to favor the Omega_inner = 112 solution (i_binary < 90). Which would suggest that the inner binary and outer tertiary are retrograde. If we instead say that these things are not coplanar, then there are several other families of solutions.

I split these up into two families, the i < 90 and i > 90. I also flipped the disk to match the same inclination. Technically there is an alternate solution to this which has the disk != star signs of inclination, but given that the inclinations are so close, I think this is a fairly pathological case. The question really is whether the i < 90 inner binary solution gives a substantially more coplanar fit than the i > 90, because this will be used to compare to the outer binary orientation. Doing this fit, the answer is that both solutions appear to be coplanar. There is enough ambiguity in the Omega_inner constraint that this permits coplanar solutions < 10 mutual inclination between disk and inner binary.

With no sense of rotation for the inner binary, there is a degeneracy between inclination < 90 and > 90 degrees. So, when I fit orbits under both assumptions, I get decent constraints on Omega (position angle of the ascending node) with strangely very similar (though definitely not identical) values. I've been trying to work out whether this is a) a bug, b ) a coincidence c) a bias from doing inference w/ only one astrometric measurement. To illustrate, here are two figures from i < 90 and i > 90, respectively. The secondary star moves from black (periastron) to orange throughout the orbit, and I've labeled the ascending node (receding from observer in my definition with red).

I feel like this must be a coincidence. Naively, for an eccentric orbit like this one (e = 0.6) I would think flipping the inclination angle would really alter the Omegas. But, since the argument of periastron (omega) is 81 degrees (i.e., the secondary is closest to the observer at ~periastron; this is well constrained from the RV curve), the projection of the orbit is nearly symmetric to us. The fact that omega = 81 degrees and not 90 degrees probably explains the small difference in inferred Omega values between i < 90 and i > 90 (Omega = 111 degrees and 104 degrees, respectively.)

### Omega_B (outer binary ascending node)
From the direction of stellar motion, we are confident that we have broken whether i_B is < or > 90 degrees. i_B is clearly > 90, since the secondary is moving *clockwise*.

Breaking the degeneracy of which node is the ascending (moving away from us, by our definition) and the value of omega_outer rests upon whether the velocity of B is greater or less than gamma_A. It also possibly involves the trend that the radial velocity of B is increasing over our baseline.

## Configurations

We are thinking of reporting fits for both i_disk < 90 and i_disk > 90. 

We might also want to report fits for the two modes of Omega_B. 

As a favored fit, I think we should focus on i_disk > 90 (contrary to surface brightness guess but clockwise rotation coplanar with i_B) and the value of Omega_B that yields a positive v_B RV increase. The two solutions for Omega_B are 110 and -20 degrees. On their own, the location of the nodes doesn't necessarily dictate whether the velocity is increasing or decreasing, it is actually the argument of periastron and the epoch of periastron that dictate that. 

At the measurement epoch, the velocity of B is definitely less than that of gamma_A.

# Available Data

## A-B wide orbit

There are about 10 epochs of astrometric observations for the wide visual orbit of A-B.

## Aa-Ab tight orbit

There is one epoch of astrometry splitting this pair (`int_data.dat`, from [Anthonioz et al. 2015](https://ui.adsabs.harvard.edu/#abs/2015A&A...574A..41A/abstract)), as well as many double-lined spectroscopic epochs, taken on 4 different instruments.  These data are all from [Kellogg et al. 2017](https://ui.adsabs.harvard.edu/#abs/2017ApJ...844..168K/abstract)

The dates are listed in HJD - 2,400,000. Generally, the columns of the orbit are something like

    (2,400,000+)	Phase	(km s^-1)	(km s^-1)	(km s^-1)


The following notes are contained in the tables of Kellogg et al.

### du Pont
Notes. RV uncertainties are sigma_Aa = 1.46 km s^-1, sigma_Ab = 2.34 km s^-1, and sigma_B = 3.95 km s^-1 (see text).

There are two epochs which have measurements for only one component.

### CfA
Note. RV uncertainties for the TWA 3A components were determined iteratively from our combined orbital solution (see text) and account for the varying strength of each spectrum.

### FEROS
Note. RV uncertainties are sigma_Aa = 2.61 km s^-1, sigma_Ab = 3.59 km s^-1, and sigma_B = 2.60 km s^-1 (see text).

### KECK
Note. RV uncertainties are sigma_Aa = 0.63 km s^-1, sigma_Ab = 0.85 km s^-1, and sigma_B = 0.59 km s^-1 (see text).
