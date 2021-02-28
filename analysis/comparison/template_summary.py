"""
Read both MCMC summaries from the Joint less and more runs, and compile them into a LaTeX table.
"""

import arviz as az
import corner
import exoplanet as xo
import matplotlib.pyplot as plt
import pymc3 as pm
import pandas as pd
import os

import twa.data as d
from twa.constants import *
from twa.plot_utils import efficient_autocorr, efficient_trace
from twa import joint
from pathlib import Path

p = Path(os.getenv("TWA_ANALYSIS_ROOT"))

df_less = pd.read_csv(p / "joint" / "rv_astro_disk_less" / "chains" / "current.csv")
df_more = pd.read_csv(p / "joint" / "rv_astro_disk_more" / "chains" / "current.csv")


# calculate rp for both cases
for df in [df_less, df_more]:
    df["rp"] = df["aOuter"] * (1 - df["eOuter"])  # au


# Assign table label with chain keyword and format precision for sampled and derived parameters.
# assemble into a block and then insert into template.


class Row:
    def __init__(self, key, label, precision=None, transformation=None, null=False):
        self.key = key  # mc label
        self.label = label  # row name in latex
        if precision is not None:
            self.precision = precision  # format string
        else:
            self.precision = "{:.2f}"

        if transformation is not None:
            self.transformation = transformation
        else:
            self.transformation = lambda x: x

        self.null = null

    def get_vals(self, df):
        """
        Calculate the mean and std off of the transformed variables.
        """
        samples = self.transformation(df[self.key])
        return (np.mean(samples), np.std(samples))

    def __call__(self, df1, df2):
        """
        Args:
            df : the DataFrame
        
        Output the LaTeX, applying the appropriate transformation.
        """

        if self.null:
            return (
                self.label
                + "& \\nodata\\tablenotemark{c} & \\nodata\\tablenotemark{c}\\\\"
            )
        else:
            return self.label + (
                " & $"
                + self.precision
                + " \pm "
                + self.precision
                + "$ & $"
                + self.precision
                + " \pm "
                + self.precision
                + "$\\\\"
            ).format(*self.get_vals(df1), *self.get_vals(df2))


# # sample pars
# [
#     "mparallax",
#     "aAngInner",
#     "logPInner",
#     "eInner",
#     "omegaInner",
#     "OmegaInner",
#     "cosInclInner",
#     "MAb",
#     "tPeriastronInner",
#     "logPOuter",
#     "omegaOuter",
#     "OmegaOuter",
#     "phiOuter",
#     "cosInclOuter",
#     "eOuter",
#     "gammaOuter",
#     "MB",
#     "offsetKeck",
#     "offsetFeros",
#     "offsetDupont",
#     "logjittercfa",
#     "logjitterkeck",
#     "logjitterferos",
#     "logjitterdupont",
#     "logRhoS",
#     "logThetaS",
#     "iDisk",
#     "OmegaDisk",
# ]

tdeg = lambda x: x / deg
tOout = lambda x: x / deg + 360.0
texp = lambda x: np.exp(x)
tyr = lambda x: x / yr
tjd = lambda x: x + jd0 - 2450000
sampled_rows = [
    Row("PInner", r"$P_\mathrm{inner}$ [days]", precision="{:.3f}"),
    Row("aAngInner", r"$a_\mathrm{inner}$ [mas]"),
    Row("MAb", r"$M_\mathrm{Ab}$ [$M_\odot$]"),
    Row("eInner", r"$e_\mathrm{inner}$"),
    Row(
        "inclInner",
        r"$i_\mathrm{inner}$ [\degr]",
        transformation=tdeg,
        precision="{:.1f}",
    ),
    Row(
        "omegaInner",
        r"$\omega_\mathrm{Aa}$\tablenotemark{a} [\degr]",
        transformation=tdeg,
        precision="{:.0f}",
    ),
    Row(
        "OmegaInner",
        r"$\Omega_\mathrm{inner}$\tablenotemark{b} [\degr]",
        transformation=tdeg,
        precision="{:.0f}",
    ),
    Row(
        "tPeriastronInner",
        r"$T_{0,\mathrm{inner}}$ [JD - 2,450,000]",
        transformation=tjd,
    ),
    Row("POuter", r"$P_\mathrm{outer}$ [yrs]", transformation=tyr, precision="{:.0f}"),
    Row("MB", r"$M_\mathrm{B}$ [$M_\odot$]"),
    Row("eOuter", r"$e_\mathrm{outer}$", precision="{:.1f}"),
    Row(
        "inclOuter",
        r"$i_\mathrm{outer}$ [\degr]",
        transformation=tdeg,
        precision="{:.0f}",
    ),
    Row(
        "omegaOuter",
        r"$\omega_\mathrm{A}$\tablenotemark{a} [\degr]",
        transformation=tOout,
        precision="{:.0f}",
        null=True,
    ),
    Row(
        "OmegaOuter",
        r"$\Omega_\mathrm{outer}$\tablenotemark{b} [\degr]",
        transformation=tdeg,
        precision="{:.0f}",
        null=True,
    ),
    Row(
        "tPeriastronOuter",
        r"$T_{0,\mathrm{outer}}$ [JD - 2,450,000]",
        transformation=tjd,
        precision="{:.0f}",
        null=True,
    ),
    Row("mparallax", r"$\varpi$ [$\arcsec$]"),
    Row(
        "logRhoS", r"$\sigma_\rho$ [$\arcsec$]", precision="{:.3f}", transformation=texp
    ),
    Row(
        "logThetaS",
        r"$\sigma_\theta$ [$\degr$]",
        precision="{:.3f}",
        transformation=texp,
    ),
    Row("jitCfa", r"$\sigma_\mathrm{CfA}$ [km s${}^{-1}$]", precision="{:.1f}"),
    Row("jitKeck", r"$\sigma_\mathrm{Keck}$ [km s${}^{-1}$]", precision="{:.1f}"),
    Row("jitFeros", r"$\sigma_\mathrm{FEROS}$ [km s${}^{-1}$]", precision="{:.1f}"),
    Row("jitDupont", r"$\sigma_\mathrm{du\;Pont}$ [km s${}^{-1}$]", precision="{:.1f}"),
    Row("offsetKeck", r"(Keck - CfA) [km s${}^{-1}$]", precision="{:.1f}"),
    Row("offsetFeros", r"(FEROS - CfA) [km s${}^{-1}$]", precision="{:.1f}"),
    Row("offsetDupont", r"(du Pont - CfA) [km s${}^{-1}$]", precision="{:.1f}"),
]

derived_rows = [
    Row("MAa", r"$M_\mathrm{Aa}$ [$M_\odot$]"),
    Row("MA", r"$M_\mathrm{A}$ [$M_\odot$]"),
    Row("aInner", r"$a_\mathrm{inner}$ [au]", precision="{:.3f}"),
    Row("aOuter", r"$a_\mathrm{outer}$ [au]", precision="{:.0f}"),
    Row("rp", r"$r_{p,\mathrm{outer}}$ [au]", precision="{:.0f}"),
]

sampled = "\n".join([row(df_more, df_less) for row in sampled_rows])
derived = "\n".join([row(df_more, df_less) for row in derived_rows])

template = """\\begin{{deluxetable}}{{lcc}}
\\tablecaption{{Orbital parameters \\label{{tab:orbits}}.}}
\\tablehead{{\\colhead{{Parameter}} & \\colhead{{primary solution}} & \\colhead{{alternate solution}}}}
\\startdata
\\cutinhead{{Sampled}}
{sampled:}
\\cutinhead{{Derived}}
{derived:}
\\enddata
\\tablenotetext{{a}}{{The argument of periastron of the primary. $\\omega_\\mathrm{{secondary}} = \\omega_\\mathrm{{primary}} + \\pi$.}}
\\tablenotetext{{b}}{{The ascending node is identified as the point where the secondary body crosses the sky plane \\emph{{receding}} from the observer.}}
\\tablenotetext{{c}}{{Posterior is non-Gaussian; see Figure~\\ref{{fig:corner}}}}
\\end{{deluxetable}}
"""

with open("table.tex", "w") as f:
    f.write(template.format(sampled=sampled, derived=derived))


# # all pars
# [
#     "mparallax",
#     "MAb",
#     "logPOuter",
#     "offsetKeck",
#     "offsetFeros",
#     "offsetDupont",
#     "logRhoS",
#     "logThetaS",
#     "parallax",
#     "aAngInner",
#     "aInner",
#     "logPInner",
#     "PInner",
#     "eInner",
#     "omegaInner",
#     "OmegaInner",
#     "cosInclInner",
#     "inclInner",
#     "tPeriastronInner",
#     "MA",
#     "MAa",
#     "POuter",
#     "omegaOuter",
#     "OmegaOuter",
#     "phiOuter",
#     "tPeriastronOuter",
#     "cosInclOuter",
#     "inclOuter",
#     "eOuter",
#     "gammaOuter",
#     "MB",
#     "Mtot",
#     "aOuter",
#     "aAngOuter",
#     "logjittercfa",
#     "logjitterkeck",
#     "logjitterferos",
#     "logjitterdupont",
#     "jitCfa",
#     "jitKeck",
#     "jitFeros",
#     "jitDupont",
#     "iDisk",
#     "OmegaDisk",
#     "thetaDiskInner",
#     "thetaInnerOuter",
#     "thetaDiskOuter",
# ]
