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

from typing import List
from BCSna import BCSAction, findMu


class labelDict:
    def __init__(self):
        self.labeldict = dict({
                "eb"   : ["$\\epsilon_B$", -1, False],
                "ef"   : ["$\\Sigma_F$", -1, False],
                "g"    : ["$g$", -1, False],
                "h"    : ["$h$", -1, False],
                "rhoF" : ["$n_F$", -1, False],
                "nthrm": ["$n_{B,\\text{th}}$", -1, False],
                "rho"  : ["$\\rho_{0,k}/n_{\\text{tot}}$", -1, False],
                "avv"  : ["$A_{v,k}$", -1, False],
                "all"  : ["$A_{l,k}$", -1, False],
                "lutK" : ["$K$", -1, False],
                "iota" : ["k\\xi$", -1, False],
                "dfac" : ["$\\Delta$", -1, False],
                "g1"   : ["$y_k$", -1, False]
                })

    def setShownLabel(self, key:str, keys_upd:List[str], initNorm=False):
        if key in keys_upd:
            idx = int(list(keys_upd).index(key))
            if key in self.labeldict.keys():
                self.labeldict[key][1] = idx
                self.labeldict[key][2] = initNorm
                if initNorm:
                    s = str(self.labeldict[key][0])
                    s = s[1:-1]
                    self.labeldict[key][0] = f"${s}/({{{s}}}(k=0))$"

    def getyi(self, y, key):
        if (self.labeldict[key][1]!=-1):
            if (self.labeldict[key][2]):
                return y[self.labeldict[key][1],:] / y[self.labeldict[key][1],0]
            else:
                return y[self.labeldict[key][1],:]
        else:
            return None


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


def regis(lbldict: labelDict, noninitkey, initkey, keys_upd):
    for key in keys_upd:
        if key in noninitkey:
            lbldict.setShownLabel(key, keys_upd, False)
        if key in initkey:
            lbldict.setShownLabel(key, keys_upd, True)


def plotAll(bcs:BCSAction, Shownkeylst_noninit=[], Shownkeylst_init=[]):
    if bcs.becShift:
        soltThr = bcs.solThr.t
        soltBCS = bcs.solBEC.t
        soltall = np.array(list(soltThr) + list(soltBCS))
        solyThr = bcs.solThr.y
        solyBCS = bcs.solBEC.y

        keythr = bcs.solThrKeys.copy()
        keybcs = None if bcs.ydata.keysUpd is None else bcs.ydata.keysUpd.copy()
        ythr, ybcs, yall, kthr, kbcs, kall = plotcurveConcat(solyThr, solyBCS, keythr, keybcs)
        
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 20
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.figure()

        lbldict = labelDict()
        regis(lbldict, Shownkeylst_noninit, Shownkeylst_init, kthr)
        regis(lbldict, Shownkeylst_noninit, Shownkeylst_init, kbcs)
        regis(lbldict, Shownkeylst_noninit, Shownkeylst_init, kall)

        curvlst = list()
        lgdlst = list()
        for ki in kthr:
            if ki in lbldict.labeldict.keys():
                if lbldict.labeldict[ki][1]!=-1:
                    yi = lbldict.getyi(ythr, ki)
                    if yi is not None:
                        ci, = plt.plot(soltThr, yi)
                        curvlst.append(ci)
                        lgdlst.append(lbldict.labeldict[ki][0])
        for ki in kbcs:
            if ki in lbldict.labeldict.keys():
                if lbldict.labeldict[ki][1]!=-1:
                    yi = lbldict.getyi(ybcs, ki)
                    if ki=="rho":
                        yi = np.array(yi) * 2.0 * np.pi
                    if yi is not None:
                        ci, = plt.plot(soltBCS, yi)
                        curvlst.append(ci)
                        lgdlst.append(lbldict.labeldict[ki][0])
        for ki in kall:
            if ki in lbldict.labeldict.keys():
                if lbldict.labeldict[ki][1]!=-1:
                    yi = lbldict.getyi(yall, ki)
                    if ki == "eb":
                        if yi is not None:
                            ci, = plt.plot(soltThr, yi[:len(soltThr)])
                            curvlst.append(ci)
                            lgdlst.append(lbldict.labeldict[ki][0])
                    else:
                        if yi is not None:
                            ci, = plt.plot(soltall, yi)
                            curvlst.append(ci)
                            lgdlst.append(lbldict.labeldict[ki][0])
        plt.legend(curvlst, lgdlst)
        plt.show()
    else:
        solt = bcs.solThr.t
        soly = bcs.solThr.y
        keythr = bcs.solThrKeys.copy()
        curvlst = list()
        plt.figure()
        for idx, _ in enumerate(keythr):
            ci, = plt.plot(solt, soly[idx,:])
            curvlst.append(ci)
        plt.legend(curvlst, keythr)
        plt.show()
    return



def main():
    kF = 1.0
    mass = 1.0
    ef = (kF**2) / (2.0 * mass)
    eb = 2.
    cutoff = 200.0
    beta = 60.0
    mu = -0.7

    mu = findMu(float(1.0/(2.*np.pi)), eb, beta, cutoff, mass, mu)
    print(mu)
    bcs = BCSAction(eb, beta, mu, cutoff, mass)
    print(f"Ntot={bcs.FinalNum()},\tSFratio={bcs.FinalRhoSF()}")

    a = np.sqrt((4.0 / np.exp(2.*np.euler_gamma) / eb / mass))
    print(f"log(kFa)={np.log(kF*a):.4f},\tk_{{B}}T/E_{{F}}={1.0/beta/ef:.4f}")
    plotAll(bcs, ["all", "avv","rho","g1"], ["eb"])

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
