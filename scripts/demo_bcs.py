#!/usr/bin/python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# ///
"""Demo: BCSAction density map over (mu, T) — moved from BCSna.__main__."""

import matplotlib.pyplot as plt
import numpy as np

from BCSna import BCSAction


def plotcurveConcat(solythr, solybec, keythr, keyall):
    ythr = list()
    ybcs = list()
    yall = list()
    kthr = list()
    kbcs = list()
    kall = list()
    for idx, keyi in enumerate(keyall):
        if keyi in keythr:
            idxthr = keythr.index(keyi)
            yall.append(list(solythr[idxthr,:]) + list(solybec[idx,:]))
            kall.append(keyi)
        else:
            ybcs.append(list(solybec[idx, :]))
            kbcs.append(keyi)
    for idxthr, keyithr in enumerate(keythr):
        if (keyithr in keyall) is False:
            ythr.append(list(solythr[idxthr, :]))
            kthr.append(keyithr)
    return np.array(ythr), np.array(ybcs), np.array(yall), kthr, kbcs, kall


def plotAll(bcs:BCSAction):
    soltThr = bcs.solThr.t
    soltBCS = bcs.solBEC.t
    soltall = np.array(list(soltThr) + list(soltBCS))
    solyThr = bcs.solThr.y
    solyBCS = bcs.solBEC.y

    keythr = bcs.solThrKeys.copy()
    keybcs = None if bcs.ydata.keysUpd is None else bcs.ydata.keysUpd.copy()
    ythr, ybcs, yall, kthr, kbcs, kall = plotcurveConcat(solyThr, solyBCS, keythr, keybcs)
    
    print(bcs.solThrKeys)
    print(bcs.ydata.keysUpd)
    print(bcs.FinalNum())
    print(bcs.FinalRhoSF())
    print(bcs.becShift)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.figure()

    curvlst = list()
    for idx, _ in enumerate(kthr):
        ci, = plt.plot(soltThr, ythr[idx, :])
        curvlst.append(ci)
    for idx, _ in enumerate(kbcs):
        ci, = plt.plot(soltBCS, ybcs[idx, :])
        curvlst.append(ci)
    for idx, _ in enumerate(kall):
        ci, = plt.plot(soltall, yall[idx, :])
        curvlst.append(ci)
    plt.legend(curvlst, kthr+kbcs+kall)
    plt.show()
    return



def main():
    kF = 1.0
    mass = 1.0
    ef = (kF**2) / (2.0 * mass)
    eb = 20.
    cutoff = 50.0
    beta = 20.
    mu = -9.5

    bcs = BCSAction(eb, beta, mu, cutoff, mass)
    plotAll(bcs) 

    """
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.figure()
    c1, = plt.plot(solt, soly[0, :]/soly[0,0])
    c2, = plt.plot(solt, soly[1, :]/soly[1,0])
    c3, = plt.plot(solt, soly[2, :])
    c4, = plt.plot(solt, soly[3, :])
    c5, = plt.plot(solt, soly[5, :])
    c6, = plt.plot(solt, soly[4, :])
    plt.xlabel("$\\log(\\Lambda / k)$")
    assert bec.ydata.keysUpd is not None
    #plt.legend([c1, c2, c3, c4, c5], ["$g/g_0$", "$\\rho_{0,k}$", "$A_{v,k}$", "$A_{l,k}$", "$y_k$"])
    plt.legend([c1, c2, c3, c4, c5, c6], ["$g/g_0$", "$\\rho_{0,k}/\\rho_{0,k=0}$", "$A_{v,k}$", "$A_{l,k}$", "$y_k\\xi^2$", "$k\\xi/(2\\pi)$"])
    plt.show()
    """

    """
    mulst = np.arange(-0.35, 0.6, 0.006) * ef
    betalst = 1.0 / np.arange(0.1 / beta, 10.0 / beta, 0.1 / beta)

    mu2D, beta2D = np.meshgrid(mulst, betalst)
    nTot = np.zeros(mu2D.shape)
    lx, ly = mu2D.shape
    for idx in range(lx):
        for jdx in range(ly):
            nTot[idx, jdx] = BCSAction(eb, beta2D[idx, jdx], mu2D[idx, jdx], cutoff, mass).FinalNum()

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    plt.rcParams["mathtext.fontset"] = "cm"

    fig, ax = plt.subplots()
    ax.ticklabel_format(style="sci", scilimits=(-2, 2))
    c = ax.pcolormesh(
        mu2D / eb,
        1.0 / (beta2D * eb),
        nTot / eb / mass,
        shading="nearest",
        cmap="RdBu",
        vmin=np.min(nTot),
        vmax=np.max(nTot) / 6.0,
    )
    ax.set_xticks(np.arange(-0.5, 1.01, 0.5))
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("$\\mu_F/\\epsilon_B$")
    ax.set_ylabel("$k_B T /\\epsilon_B$")
    ax.set_title("$n_{tot}/(m_F\\epsilon_B)$")
    plt.show()
    """


if __name__ == "__main__":
    main()
