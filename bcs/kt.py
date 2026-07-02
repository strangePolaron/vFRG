#!/usr/bin/python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# ///
"""Kosterlitz-Thouless RG for rigid-ball / vortex coupling sector."""

import numpy as np
from bcs.keys import Key
from bcs.sector import CouplingSpec, RGSector
from bcs.state import RGState

cutoff_convert_2_rspc = lambda cutoff: 2.0 * np.pi / cutoff


class KT(RGSector):
    def __init__(self, prsData: RGState, lutK, nMax: int, keysUpdRegis=True):
        self.lpar_add = 0.0
        self.g = 1.0 / (np.sqrt(3.0)) / pow(prsData.data["iota"], 2)
        self.g = 1.0 / (4.0 * np.sqrt(3.0)) 
        self.nMax = nMax
        vn = self.gn_init()
        couplings = (CouplingSpec("lutK", float(lutK), Key.LUTK),) + tuple(
            CouplingSpec(f"g{idx + 1}", float(vn[idx])) for idx in range(self.nMax)
        )
        super().__init__(prsData, couplings, append_keys=keysUpdRegis)
        self.ydatakeys = self.coupling_names

    def scaldim(self, n: int):
        return np.double((n**2 * np.pi) / (self.ydata.value("lutK")))

    def gn_init(self):
        vn = np.zeros(self.nMax, dtype=np.double)
        for idx in range(self.nMax):
            vn[idx] = self.g
        return vn

    def yGen(self):
        return self.ydata.ylst()

    def parUpd(self, ynew):
        self.ydata.update(ynew)

    def contribute(self, l, dy: RGState):
        self.eqRHS_ydata(l, dy)

    def contribute_post(self, l: float, dy: RGState) -> RGState | None:
        #return super().contribute_post(l, dy)
        dy_new = dy.zero_like()
        dy_new.data["iota"] = dy.data["iota"]
        self.eqRHS_ydata(l, dy_new)
        dy_new.data["iota"] = 0.0
        return dy_new

    def eqRHS_ydata(self, _, dy: RGState):
        gilen = len(self.ydatakeys) - 1
        y_lutK = 4.0 * pow(np.pi, 3) * sum(
            [(np.double(varName[1:]) ** 2) * np.pow(self.ydata.value(varName), 2) for varName in self.ydatakeys[1:]]
        )
        dy.data["lutK"] = y_lutK
        
        bareScl = -1.0 * dy.data["iota"] / self.ydata.value("iota")
        #print(bareScl)
        vorRegFactor = 1.0 / (1.0 + pow(self.ydata.value("iota"), 2))
        
        for varName in self.ydatakeys[1:]:
            dy.data[varName] = (2.0 * bareScl - self.scaldim(int(varName[1:])) * vorRegFactor) * self.ydata.value(varName)
        for idx in range(gilen):
            tmpi = 0.0
            for jdx in range(idx + 1, gilen):
                tmpi -= self.ydata.value(self.ydatakeys[jdx + 1]) * self.ydata.value(self.ydatakeys[jdx - idx])
            for jdx in range(idx):
                tmpi -= self.ydata.value(self.ydatakeys[jdx + 1]) * self.ydata.value(self.ydatakeys[idx - jdx]) / 2.0
            dy.data[self.ydatakeys[idx + 1]] += 2.0 * np.pi * tmpi

    def eqRHS_onlyKT(self, lpar, y):
        self.parUpd(y)
        dy = self.ydata.zeroVecGen()
        self.contribute(lpar, dy)
        return dy.ylst()
