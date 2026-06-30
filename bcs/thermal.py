#!/usr/bin/python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "numpy",
# ]
# ///
"""Thermal boson RG flow before BEC transition."""

import numpy as np
from bcs.distributions import nB
from bcs.keys import Key
from bcs.sector import CouplingSpec, RGSector
from bcs.state import RGState

ydatakeysPrompt = ["g", "eb", "nthrm"]


class ThermalBoson(RGSector):
    def __init__(self, prsdata, m, cutoff, beta, g0, eb0, mu, lpar0=0.0):
        couplings = (
            CouplingSpec("g", g0, Key.G),
            CouplingSpec("eb", eb0 - mu, Key.EB),
            CouplingSpec("nthrm", 0.0, Key.NTHRM),
        )
        super().__init__(prsdata, couplings)
        self.m = m
        self.cutoff = cutoff
        self.beta = beta
        self.ydatakeys = self.coupling_names
        self.yval = lambda x: self.ydata.value(x)
        self.lpar = lpar0

    def isCondensing(self):
        return (self.yval("eb") < -1e-2 * self.yval("g")) and (self.yval("g") > 0)

    def reNorm(self, dZleg=0.0):
        return -1.0 * self.yval("eb") * dZleg, -2.0 * self.yval("g") * dZleg

    def ekCalc(self):
        self.ekval = pow(self.cutoff, 2) * np.exp(-2.0 * self.lpar) / (2.0 * self.m)

    def nbCalc(self):
        self.nbval = -1.0 * nB(self.ekval + self.yval("eb"), self.beta)

    def dosCalc(self):
        self.dosCoeff = self.ekval * self.m / np.pi

    def lp_g(self):
        return -1.0 * pow(self.yval("g"), 2) * (1.0 + 2.0 * self.nbval) / (2.0 * (self.ekval + self.yval("eb")))

    def lp_eb(self):
        return 2.0 * self.yval("g") * self.nbval

    def lp_dn(self):
        return self.nbval

    def upd(self, l):
        self.lpar = l
        self.ekCalc()
        self.nbCalc()
        self.dosCalc()

    def contribute(self, l, dy: RGState):
        self.upd(l)
        dy.data["g"] = self.lp_g() * self.dosCoeff
        dy.data["eb"] = self.lp_eb() * self.dosCoeff
        dy.data["nthrm"] = self.lp_dn() * self.dosCoeff

    def dylst_onlythrm(self, l, y):
        self.ydata.update(y)
        dy = self.ydata.zeroVecGen()
        self.dylst(l, dy)
        return dy.ylst()


class ThrterminFunc:
    def __init__(self, m, beta, gidx, ebidx):
        self.terminal = True
        self.direction = -1.0
        self.m = m
        self.beta = beta
        self.gidx = gidx
        self.ebidx = ebidx

    def __call__(self, _, y):
        return max(-1.0 * self.m * y[self.gidx], self.beta * y[self.ebidx] / (self.m * abs(y[self.gidx]))) + 1e-4
