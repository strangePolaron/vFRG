#!/usr/bin/python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy",
#     "tqdm",
# ]
# ///
"""BEC-side parameter sweep helpers for plotTcBEC / scripts/plot_tc_bec.py."""

import BECna as bec
import numpy as np

#eblst = np.arange(0.01, 2.5, 0.01)
eblst = np.exp(np.linspace(np.log(4.0) - 2.0 * np.euler_gamma + 8.0, np.log(4.0) - 2.0 * np.euler_gamma + 3.0, 101)) 

betalst = 1.0 / np.arange(1.0 / 10000.0, 10.01 / 100.0, 10.0 / 10000.0)
betaMulst = 1.0 / np.arange(1.0 / 10000.0, 1.0 / 40.0, 1.0 / 200.0)

ebgrid, betagrid = np.meshgrid(eblst, betalst)
ori_shape = ebgrid.shape
totLen = len(eblst) * len(betalst)
parLst = list(zip(ebgrid.reshape(totLen), betagrid.reshape(totLen)))

ebMugrid, betaMugrid = np.meshgrid(eblst, betaMulst)
ori_shape_mu = ebMugrid.shape
totMuLen = len(eblst) * len(betaMulst)
parMuLst = list(zip(ebMugrid.reshape(totMuLen), betaMugrid.reshape(totMuLen)))


def rhoSF(parpair, targetNum=1.0 / (4.0 * np.pi), mass=1.0):
    eb, beta = parpair
    mu = bec.findMu(targetNum, eb, beta, mass)
    becobj = bec.BECAction(eb, beta, mu, mass)
    sfFrac = becobj.FinalRhoSF()
    print(f"eb:\t{eb},\tbeta:\t{beta},\tmu:\t{mu},\tsfFrac:\t{sfFrac}")
    return np.nan_to_num(sfFrac, nan=0.0, posinf=0.0, neginf=0.0)


def eb_row_tasks(targetNum=1.0 / (4.0 * np.pi), mass=1.0):
    return [(float(eb), betalst.copy(), targetNum, mass) for eb in eblst]


def rhoSF_eb_row(task):
    eb, betas, targetNum, mass = task
    mu_hint = None
    row = []
    for beta in betas:
        mu = bec.findMu(targetNum, eb, float(beta), mass, mu_guess=mu_hint)
        mu_hint = mu
        becobj = bec.BECAction(eb, float(beta), mu, mass)
        sfFrac = becobj.FinalRhoSF()
        rhotot = becobj.FinalNum()
        print(f"eb:\t{eb},\tbeta:\t{float(beta)},\tmu:\t{mu},\tsfFrac:\t{sfFrac},\trho:{rhotot}")
        row.append(sfFrac)
    return row


def muBEC(parpair, targetNum=1.0 / (4.0 * np.pi), mass=1.0):
    eb, beta = parpair
    return bec.findMu(targetNum, eb, beta, mass)


if __name__ == "__main__":
    from scripts.plot_tc_bec import main

    main()
