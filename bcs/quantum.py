#!/usr/bin/python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# ///
"""Quantum BEC boson RG flow with optional Kosterlitz-Thouless sector."""

import numpy as np
from bcs.distributions import nB
from bcs.keys import Key
from bcs.kt import KT
from bcs.sector import CouplingSpec, RGSector
from bcs.state import RGState

ydatakeysPrompt = ["g", "rho", "avv", "all"]

_xi_div_healingLength = float(1.0)

class QuantumAction(RGSector):
    def __init__(self, prsdata: RGState, m, cutoff, lpar_init0, beta, g0, rho0, KTSwitch=True):
        iota0 = (cutoff * np.exp(-1.0*lpar_init0) /(np.sqrt(2.0*m*g0*rho0))) * _xi_div_healingLength
        couplings = (
            CouplingSpec("g", g0, Key.G),
            CouplingSpec("rho", rho0, Key.RHO),
            CouplingSpec("avv", 1.0, Key.AVV),
            CouplingSpec("all", 1.0, Key.ALL),
            CouplingSpec("iota", iota0, Key.IOTA),
        )
        #print(iota0)
        super().__init__(prsdata, couplings)
        self.KTSwitch = KTSwitch
        self.m = m
        self.cutoff = cutoff
        self.lpar_0 = lpar_init0
        self.lpar = self.lpar_0
        self.beta = beta
        self.k = self.cutoff * np.exp(-1.0 * self.lpar)
        self.cutoff_0 = cutoff * np.exp(-1.0 * self.lpar_0)
        self.ydatakeys = self.coupling_names
        self.yval = lambda x: self.ydata.value(x)
        #self.ktStart = False
        self.updInternalVar()
        self.gnMax = 1
        self.rbAct = KT(self.ydata, self.lutK(), self.gnMax, False)
        #self.ktStart = False
        if KTSwitch:
            tmp = self.rbAct.ydatakeys.copy()
            tmp.remove("lutK")
            self.ydata.keysUpdAppend(tmp)

    def updInternalVar(self):
        self.all_div_avv_sqrt = np.sqrt(self.yval("all") / self.yval("avv"))
        self.k2 = pow(self.k, 2)
        self.dosCoeff = self.k2 / (2.0 * np.pi)
        if (self.k * np.sqrt(self.beta) > 1e-2) and (self.beta * self.k2*self.all_div_avv_sqrt)>1e-5:
            self.ek = pow(self.k, 2) / (2.0 * self.m)
            self.k2 = pow(self.k, 2)
            self.Ek = self.Ek_pole()
            self.nbval = -1.0 * nB(self.Ek, self.beta)
            self.coth = 1.0 + 2.0 * self.nbval
            self.csch2 = 4.0 * self.nbval * (self.nbval + 1.0)
            #if self.yval("all") / self.yval("avv")<=0.0:
            #    print("Neg found", self.yval("all"), self.yval("avv"), self.ek, self.yval("rho"))
            #    a=input()
        else:
            self.updInfrared()

        #self.ktStart = self.ktStart or self.isKTstart()

    def updInfrared(self):
        self.ek = 1.0 / (2.0 * self.m)
        self.Ek = self.all_div_avv_sqrt * np.sqrt(self.k2 * self.ek + 2.0 * self.yval("g") * self.yval("rho") * self.yval("avv"))
        self.nbval = 1.0 / (self.beta * self.Ek)
        self.coth = self.k + 2.0*self.nbval
        self.csch2 = 4.0 * self.nbval * (self.k + self.nbval)
        self.dosCoeff = pow(self.k, 2) / (2.0 * np.pi)
        #self.ktStart = self.ktStart or self.isKTstart()

    def Ek_pole(self):
        #print(f"lpar:{self.lpar}, mu:{self.yval("g")*self.yval("rho")}")
        return np.sqrt(
            self.ek
            * (self.ek + 2.0 * self.yval("g") * self.yval("rho") * self.yval("avv"))
            * self.yval("all")
            / self.yval("avv")
        )

    def rho1_diag(self):
        return self.yval("all") * (
            -1.0 * self.ek * self.coth / self.Ek + 1.0 / self.all_div_avv_sqrt
        ) / 2.0

    def rho2_diag(self):
        coeff = -1.0 * pow(self.yval("g") * self.yval("all"), 2)
        term1 = pow(self.ek, 2) * self.coth / (2.0 * pow(self.Ek, 3))
        term2 = pow(self.ek / (2.0 * self.Ek), 2) * self.beta * self.csch2
        return coeff * (term1 + term2)

    def vll_diag(self):
        return -1.0 * pow(self.yval("all"), 2) * self.ek * self.beta * self.csch2 / 4.0

    def vvv_diag(self):
        #print(f"ek:{self.ek}, Ek:{self.Ek}, ek/Ek:{self.ek/self.Ek}, sqrt(avv/all):{1./self.all_div_avv_sqrt}")
        return (self.yval("all") * self.yval("avv")) * (
            self.ek * self.coth / self.Ek - 1.0 / self.all_div_avv_sqrt
        )

    def upd(self, l):
        """
        if self.yval("all")<=0:
            print (f"all:\t{self.yval("all")}")
        if self.yval("avv")<=0:
            print (f"avv:\t{self.yval("avv")}")
        if self.yval("g")<=0:
            print (f"avv:\t{self.yval("g")}")
        if self.yval("rho")<=0:
            print (f"rho:\t{self.yval("rho")}")
        """
        self.lpar = l
        self.k = self.cutoff * np.exp(-1.0 * self.lpar)
        self.updInternalVar()
        if self.KTSwitch:  #and self.ktStart:
            self.ydata.data["lutK"] = self.lutK()
            self.ydata.data["iota"] = self.healLength() * self.k * _xi_div_healingLength #/ (2.0 * np.pi)

    def lutK(self):
        return self.m / (self.yval("rho") * self.yval("all")) / self.beta

    def drhoKT(self, dK):
        Kkt = self.lutK()
        return -1.0 * self.m * dK / pow(Kkt, 2) / self.beta

    def contribute(self, l, dy: RGState):
        self.upd(l)
        drhoTot = self.rho1_diag() * self.dosCoeff
        dg = self.rho2_diag() * self.dosCoeff
        dall = self.vll_diag() * self.dosCoeff / self.yval("rho")
        davv = self.vvv_diag() * self.dosCoeff / self.yval("rho")

        dy.data["rho"] += drhoTot
        dy.data["g"] += dg
        dy.data["all"] += dall
        dy.data["avv"] += davv

    def contribute_post(self, l, dy: RGState) -> RGState | None:
        if self.KTSwitch:  #and self.ktStart:
            dypost = self.rbAct.contribute_post(l, dy)
            if dypost is not None:
                dall = self.drhoKT(float(dypost.data["lutK"])) / self.yval("rho")
                #print(dall)
                dypost.data["all"] += dall
            return dypost
        return None

    def dylst_onlyBos(self, l, y):
        self.ydata.update(y)
        dy = self.ydata.zeroVecGen()
        self.contribute(l, dy)
        return dy.ylst()

    def reNorm(self, dZleg=0.0):
        return -2.0 * self.yval("g") * dZleg, self.yval("rho") * dZleg

    def healLength(self):
        return np.sqrt(1.0 / (2.0 * self.m * self.yval("g") * self.yval("rho") * self.yval("avv")))

    def isKTstart(self):
        return self.healLength() <= ((2.0 * np.pi) / self.k)

    def meanfieldCrit(self):
        return self.lutK() < (np.pi / 2.0)


class BECterminFunc:
    def __init__(self, m, beta, rhoidx, allidx, avvidx):
        self.terminal = True
        self.direction = -1.0
        self.m = m
        self.beta = beta
        self.rhoidx = rhoidx
        self.allidx = allidx
        self.avvidx = avvidx

    def __call__(self, _, y):
        if y[self.avvidx]>1e3:
            return min(y[self.rhoidx], y[self.allidx], y[self.avvidx]) - 1e-3
        return min(y[self.avvidx], y[self.allidx]) - 1e-3
