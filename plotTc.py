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
"""BCS-side parameter sweep helpers for plotTc / scripts/plot_tc.py."""

import BCSna as bcs
import numpy as np

kF = 1.0
mf = 1.0


def _rho_at_mu(eb, beta, mu, targetNum, cutoff, mass):
    bcsobj = bcs.BCSAction(eb, beta, mu, cutoff, mass)
    if np.abs(np.sqrt(bcsobj.FinalNum() * 2.0 * np.pi) - kF) > 0.05:
        print(
            f"eb,\t{eb:.1f},\tbeta,\t{beta:.1f},\tmu\t{mu:.2f},\tkF,\t"
            f"{np.sqrt(bcsobj.FinalNum() * 2.0 * np.pi):.2f}"
        )
        return 0.0
    return bcsobj.FinalRhoSF()


def rhoSF(parpair, targetNum=(kF**2) / (2.0 * np.pi), cutoff=50.0, mass=mf):
    eb, beta = parpair
    mu = bcs.findMu(targetNum, eb, beta, cutoff, mass)
    return _rho_at_mu(eb, beta, mu, targetNum, cutoff, mass)


def eb_row_tasks(targetNum=(kF**2) / (2.0 * np.pi), cutoff=50.0, mass=mf):
    return [(float(eb), betalst.copy(), targetNum, cutoff, mass) for eb in eblst]


def rhoSF_eb_row(task):
    eb, betas, tnum, cut, m = task
    mu_hint = None
    row = []
    for beta in betas:
        mu = bcs.findMu(tnum, eb, float(beta), cut, m, mu_guess=mu_hint)
        mu_hint = mu
        row.append(_rho_at_mu(eb, float(beta), mu, tnum, cut, m))
    return row


def muiSF(parpair, targetNum=(kF**2) / (2.0 * np.pi), cutoff=20.0, mass=mf):
    eb, beta = parpair
    mu = bcs.findMu(targetNum, eb, beta, cutoff, mass)
    bcsobj = bcs.BCSAction(eb, beta, mu, cutoff, mass)
    print(
        f"eb,\t{eb:.1f},\tbeta,\t{beta:.1f},\tmu\t{mu:.2f},\tkF,\t"
        f"{np.sqrt(bcsobj.FinalNum() * 2.0 * np.pi):.2f}\t{bcsobj.becShift}\t{bcsobj.solBEC.status}"
    )
    return mu


def becbcsSeparate(parpair, targetNum=(kF**2) / (2.0 * np.pi), cutoff=20.0, mass=mf):
    eb, beta = parpair
    mu = bcs.findMu(targetNum, eb, beta, cutoff, mass)
    bcsobj = bcs.BCSAction(eb, beta, mu, cutoff, mass)
    if np.abs(np.sqrt(bcsobj.FinalNum() * 2.0 * np.pi) - kF) > 0.05:
        rhosfFrac = 0.0
    else:
        rhosfFrac = bcsobj.FinalRhoSF()
    if rhosfFrac > 0:
        return mu / eb
    return 0.0


eblst = np.exp(np.linspace(np.log(8.0) - 2.0 * np.euler_gamma + 4.0, np.log(8.0) - 2.0 * np.euler_gamma - 4.0, 151)) * (
    (kF**2) / (2.0 * mf)
)
betalst = 1.0 / np.arange(1.0 / 10000.0, 3.02 / 100.0, 0.2 / 1000.0) / ((kF**2) / (2.0 * mf))
betaMulst = 1.0 / np.arange(1.0 / 200000.0, 5.0 / 10000.0, 2.0 / 10000.0)

ebgrid, betagrid = np.meshgrid(eblst, betalst)
ori_shape = ebgrid.shape
totLen = len(eblst) * len(betalst)
parLst = list(zip(ebgrid.reshape(totLen), betagrid.reshape(totLen)))

ebMugrid, betaMugrid = np.meshgrid(eblst, betaMulst)
oriMu_shape = ebMugrid.shape
totMuLen = len(eblst) * len(betaMulst)
parMuLst = list(zip(ebMugrid.reshape(totMuLen), betaMugrid.reshape(totMuLen)))

if __name__ == "__main__":
    from scripts.plot_tc import main

    main()
