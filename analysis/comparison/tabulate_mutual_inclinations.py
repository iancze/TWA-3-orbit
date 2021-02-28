import matplotlib.pyplot as plt
import pymc3 as pm
import pandas as pd
import os
import corner

import twa.data as d
from twa.constants import *
from pathlib import Path
import arviz as az

p = Path(os.getenv("TWA_ANALYSIS_ROOT"))

df_less = pd.read_csv(p / "joint" / "rv_astro_disk_less" / "chains" / "current.csv")
df_more = pd.read_csv(p / "joint" / "rv_astro_disk_more" / "chains" / "current.csv")


# select where v_B is increasing
# df_inc = df.loc[df["increasing"] == True]
# df_dec = df.loc[df["increasing"] == False]


def get_val(samples, weights=None):

    # print("az", low, high)
    low, median, high = corner.quantile(samples, [0.16, 0.5, 0.84], weights)
    # print("corner", low, high)

    minus = median - low
    plus = high - median

    return (median, minus, plus)


def get_upper_limit(samples, weights=None):

    # bin the samples
    if weights is not None:
        density, bin_edges = np.histogram(
            samples, bins=40, density=True, weights=weights
        )
    else:
        density, bin_edges = np.histogram(samples, bins=40, density=True)

    # do this for the actual distribution
    dx = bin_edges[1] - bin_edges[0]

    tot_prob = np.cumsum(density * dx)
    assert np.allclose(tot_prob[-1], 1.0)

    ind = np.searchsorted(tot_prob / tot_prob[-1], 0.68)

    return bin_edges[1:][ind]


rows = []
# need to branch between chains for less / more, separated for inc / dec, and corrected for prior yes/no
for (chain, chain_label) in [
    (df_less, r"$< 90$"),
    (df_more, r"$> 90$"),
]:
    for (increasing, increasing_label) in [
        (True, r"$\uparrow$"),
        (False, r"$\downarrow$"),
    ]:
        for (reweight, prior_label) in [(False, r"$\sin(\theta)$"), (True, "flat")]:

            df = chain.copy()

            # convert units
            df["thetaDiskInner"] /= deg
            df["thetaInnerOuter"] /= deg
            df["thetaDiskOuter"] /= deg

            # filter the chain
            df = df.loc[df["increasing"] == increasing]

            if reweight:
                DiskInnerWeights = 1 / np.sin(df["thetaDiskInner"] * deg)
                InnerOuterWeights = 1 / np.sin(df["thetaInnerOuter"] * deg)
                DiskOuterWeights = 1 / np.sin(df["thetaDiskOuter"] * deg)
            else:
                DiskInnerWeights = np.ones_like(df["thetaDiskInner"])
                InnerOuterWeights = np.ones_like(df["thetaInnerOuter"])
                DiskOuterWeights = np.ones_like(df["thetaDiskOuter"])

            # thetaDiskInner
            DiskInner = "$<{:.0f}$".format(
                get_upper_limit(df["thetaDiskInner"].to_numpy(), DiskInnerWeights)
            )

            if chain_label == r"$> 90$" and increasing and reweight:
                InnerOuter = "$<{:.0f}$".format(
                    get_upper_limit(df["thetaInnerOuter"].to_numpy(), InnerOuterWeights)
                )

                DiskOuter = "$<{:.0f}$".format(
                    get_upper_limit(df["thetaDiskOuter"].to_numpy(), DiskOuterWeights)
                )

            else:
                # calculate asymmetric error bars
                InnerOuter = "${:.0f}_{{-{:.0f}}}^{{+{:.0f}}}$".format(
                    *get_val(df["thetaInnerOuter"].to_numpy(), InnerOuterWeights)
                )
                DiskOuter = "${:.0f}_{{-{:.0f}}}^{{+{:.0f}}}$".format(
                    *get_val(df["thetaDiskOuter"].to_numpy(), DiskOuterWeights)
                )

            row = (
                chain_label
                + " & "
                + increasing_label
                + " & "
                + prior_label
                + " & "
                + DiskInner
                + " & "
                + InnerOuter
                + " & "
                + DiskOuter
                + r"\\"
            )
            rows.append(row)

text = "\n".join(rows)

template = r"""\begin{{deluxetable}}{{cccccc}}
\tablecaption{{Inferred mutual inclinations. \label{{tab:mut}}}}
\tablehead{{\colhead{{$i_\mathrm{{disk}}$}} & \colhead{{$v_B$}} & \colhead{{$p(\theta)$}} & \colhead{{$\theta_\mathrm{{disk-inner}}$}} & \colhead{{$\theta_\mathrm{{inner-outer}}$}} & \colhead{{$\theta_\mathrm{{disk-outer}}$}} \\ 
\colhead{{$[\degr]$}} & \colhead{{}} & \colhead{{}} &  \colhead{{$[\degr]$}} & \colhead{{$[\degr]$}} & \colhead{{$[\degr]$}}}}
\startdata
{:}
\enddata
\tablecomments{{One $\sigma$ asymmetric error bars are reported for all unimodal distributions. Sixty eight percent confidence upper limits are reported for one-sided distributions.}}
\end{{deluxetable}}
"""

with open("mut.tex", "w") as f:
    f.write(template.format(text))
