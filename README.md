# TWA-3-orbit

## Installation 

It helps to install this locally to a conda environment. In my case, I did

        $ conda create --name TWA3 python=3.6
        $ conda activate TWA3
        # install all the required packages
        $ conda install --file requirements.txt # perhaps this should be environment.yml?

`corner.py` needs to be installed with pip.
`arviz` needs to be installed from pip too.

Since exoplanet is under heavy development, we'll install this from source. With the environment still activated, go to the `exoplanet` source directory, and `python setup.py install`. (If you're satisfied with the latest stable version, you can just install exoplanet with conda). 

Then return to this directory and install the project as editable.

        $ pip install -e .

This should keep the whole directory structure much more organized and portable than before.

The idea is that all analysis scripts will be run from the root directory of this project. E.g., 

    $ python src/joint/sample.py 

and not from within individual folders 

    # don't do this 
    $ cd src/joint 
    $ python sample.py

This way we are always operating from the root directory and don't have to worry about traversing up and across the folder tree.

## Introduction

Determining the architecture of the TWA 3 hierarchical triple system using a combination of ALMA CO 2-1 observations, radial velocity, and astrometric data. 

Building up to the joint fit, we have also made fits to smaller portions of the data.

1. [Done, TWA 3A RV only] Aa-Ab tight binary orbit fit with the double-lined RV dataset. Constrains `P_A`, `K_Aa`, `K_Ab`, `e_A`, `omega_Aa`, `gamma_A`, and `phi_A`. We pretty much reproduce Kellogg et al. results. Some different RV jitter results in general.
2. [Done, TWA 3A RV + astrometry] Aa-Ab tight binary orbit fit with double-lined RVs, the astrometric constraint from Anthonioz15, and the *Gaia* parallax. Constrains the above RV parameters plus `a_A`, `i_A` and `Omega_Aa`. Derive `M_A` and see what the posterior looks like. A little tricky to sample in `cos_incl`, since there are two families of < 90 and > 90. Omega, however, is constrained because of the RV (we know which way the secondary star is moving at the epoch of astrometric measurement). Technically, we should also explore a fit where the position angle + pi, since the primary and secondary could be swapped in the Anthonioz dataset. Looks like a good fit, though, with stellar masses indicative of mid-M dwarfs. A 5% dynamical mass will really help the inclination measurement.
3. [Done, TWA 3A RV+astro+disk] Aa-Ab tight binary orbit fit with double-lined RVs, astrometric constraint, parallax, and priors on `M_A` and `gamma_A` from millimeter. The mass constraint on `M_A` does in fact really tighten things up for inclination, but at least in this space, there are still two solutions of i < 90 and > 90. The value of Omega changes *slightly* for each one (just by like 5 degrees). So, comparing this with the disk measurement and calculating which is in fact the lower inclination solution would be interesting.
4. [Done] A-B wide binary orbit fit with the astrometric dataset (WDS) and *Gaia* parallax. Constrains `P_AB`, `a_AB`, `Omega_AB`, `omega_AB`, `e_AB`, `phi_AB`.
5. [Done] A-B wide binary orbit fit with astrometric dataset and RV information from Keck double-lined solution. Constrains the above astrometric parameters plus stellar masses. If we treat the Keck points as increasing in a statistically significant manner, then this breaks the Omega degeneracy in such a way that the outer triple is at least moderately aligned (if not coplanar) with the inner binary. If we don't assume this, then there is still a degeneracy.
6. [Done] Hierarchical triple orbit simultaneously fit with RV and astrometry for both tight inner binary and wide outer binary: inner: `parallax`, `P_A`, `a_A_ang`, `M_Ab`, `e_A`, `i_A`, `omega_Aa`, `Omega_Aa`,  outer: `P_B`, `a_B_ang`, `e_AB`, `i_AB`, `omega_A`, `Omega_A`, `gamma_AB`. `M_A` is derived from inner orbit and fed to outer orbit.  `gamma_A` is essentially the RV prediction of A, and is derived from outer orbit and fed to the inner orbit. This has 15 orbital parameters. Adding 4 RV offsets, 2 * 4 RV jitter terms, and 2 astrometric jitter terms makes it 30 parameters total.  
7. Hierarchical triple orbit simultaneously fit including dynamical mass prior on `M_A`.
8. Same, but now including disk orbit normal (evaluated w/ KDE from dynamical modeling) and directly calculating mutual inclinations for all angles.

## Remaining orbital ambiguities 
After completing the disk + rv + astro fits, I think there are still a few orbital ambiguities that we should consider addressing. If you want to stretch the data, I would say that we actually have leverage on all of these quantities. However, there are some tricky parts in interpreting the data that we need to be careful to consider.

### i_A (inner binary inclination) > 90 or < 90
From the inner orbit + Anthonioz point + disk-based mass on M_A, there are two degenerate solutions for i_A (below and above 90), each of which has a *slightly* different Omega value. These solutions are plotted in the ipython notebook for this fit.

    incl:48.75 +\- 0.89
    Omega:112.10 +\- 9.46

    incl:131.25 +\- 0.90
    Omega:104.44 +\- 9.41

### i_disk > 90 or < 90
In theory, we should be able to tell from the sub-mm emission alone which side of the disk is near. However, the CO emission is sufficiently faint that I'm not sure I can do this without error. There seems like there is the brightness asymmetry for the figure-8 and there is the brightness asymmetry in the C-shape. I should look at well-resolved disks like HD163296 to ascertain which side is near/far. One additional check we can do is that if the binary and disk are coplanar, then Omega_disk should match Omega_binary. 

From the Nuker fit, PA_disk = 207 +/- 1 degr. This would be Omega_disk = 117 +/- degr. If we assume that the disk and binary are coplanar, then this would seem to favor the Omega_binary=112 solution (i_binary < 90). Which would suggest that the inner binary and outer tertiary are retrograde. If we instead say that these things are not coplanar, then there are several other families of solutions.

### Omega_B (outer binary ascending node)
From the direction of stellar motion, we are confident that we have broken whether i_B is < or > 90 degrees.

Breaking the degeneracy of which node is the ascending (moving away from us, by our definition) rests upon the trend that the radial velocity of B is increasing over our baseline.

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
