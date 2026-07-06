#!/usr/bin/python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# ///
"""Demo: BECAction RG curves — moved from BECna.__main__."""

import matplotlib.pyplot as plt

from BECna import BECAction, findMu
from typing import List
import numpy as np
import sys

import BECna


class labelDict:
    def __init__(self):
        self.labeldict = dict({
                "g"    : ["$g$", -1, False],
                "rho"  : ["$\\rho_{0,k}$", -1, False],
                "avv"  : ["$A_{v,k}$", -1, False],
                "all"  : ["$A_{l,k}$", -1, False],
                "lutK" : ["$K$", -1, False],
                "iota" : ["k\\xi$", -1, False],
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


def regis(lbldict: labelDict, noninitkey, initkey, keys_upd):
    for key in noninitkey:
        lbldict.setShownLabel(key, keys_upd, False)
    for key in initkey:  
        lbldict.setShownLabel(key, keys_upd, True)


def plotAll(bec:BECAction, Shownkeylst_noninit=[], Shownkeylst_init=[]):
    solt = bec.sol.t
    soly = bec.sol.y

    key = None if bec.ydata.keysUpd is None else bec.ydata.keysUpd.copy()
    
    print(key)
    print(bec.FinalNum())
    print(bec.FinalRhoSF())

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.figure()

    lbldict = labelDict()
    regis(lbldict, Shownkeylst_noninit, Shownkeylst_init, key)

    curvlst = list()
    lgdlst = list()
    if key is  not None:
        for ki in key:
            if ki in lbldict.labeldict.keys():
                if lbldict.labeldict[ki][1]!=-1:
                    yi = lbldict.getyi(soly, ki)
                    if yi is not None:
                        ci, = plt.plot(solt, yi)
                        curvlst.append(ci)
                        lgdlst.append(lbldict.labeldict[ki][0])
    plt.legend(curvlst, lgdlst)
    plt.show()
    return


def main():
    ebBos = 300.0
    mu = 1.2-1e-2
    mass = 1.0
    beta = float(sys.argv[1])#0.242

    mu = findMu(float(1.0/(4.0*np.pi)), ebBos, beta, mass)
    print(f"mu={mu}")
    bec = BECAction(ebBos, beta, mu, mass)

    plotAll(bec, ["all", "avv", "g1"], ["g", "rho"])

if __name__ == "__main__":
    main()
