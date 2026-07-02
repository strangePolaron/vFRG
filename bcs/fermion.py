#!/usr/bin/python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "numpy",
# ]
# ///
"""Outer-layer BCS fermion RG: ef diagnostics, pairing, and density-of-states flow."""

import numpy as np
from bcs.distributions import nB, nF
from bcs.keys import Key
from bcs.sector import CouplingSpec, RGSector
from bcs.state import RGState

thetaFunc = lambda x: (np.arctan(1000.0 * x)) / np.pi + 1.0 / 2.0


class efDiag:
    def __init__(self, prsdata: RGState, beta, muf, mf, isBEC=False):
        self.mf = mf
        self.muf = muf
        self.beta = beta
        self.iz0 = 1.0j * np.pi / self.beta
        self.ydata = prsdata
        self.yval = lambda x: self.ydata.value(x)
        self.isBEC = isBEC

    def upd(self, k2):
        self.ekb = k2 / (2.0 * (2.0 * self.mf))
        self.ekf = k2 / (2.0 * self.mf)
        self.ef_muf = self.yval("ef") - self.muf
        if self.isBEC:
            self.eBoson = np.sqrt(
                (self.yval("all") / self.yval("avv"))
                * self.ekb
                * (self.ekb + 2.0 * self.yval("avv") * self.yval("g") * self.yval("rho"))
            )
        self.eBosonThr = self.yval("eb") + self.ekb
        self.rho = 0.0
        if self.isBEC:
            self.rho = self.yval("rho") / self.yval("avv")
        self.eFermi = np.sqrt(
            pow(self.ekf + self.ef_muf, 2) + self.rho * pow(self.yval("h") * self.yval("dfac"), 2)
        )
        self.grho = self.yval("g") * self.yval("rho") * self.yval("avv") if self.isBEC else 0.0
        self.zfactor = np.sqrt(self.yval("all") / self.yval("avv")) if self.isBEC else 1.0

    def gBosonThr(self, z):
        return 1.0 / (z - self.eBosonThr)

    def gBosonBEC(self, z):
        return (self.zfactor * z + pow(self.zfactor, 2) * (self.ekb + self.grho)) / (
            pow(z, 2) - pow(self.eBoson, 2)
        )

    def ResBosBECp(self):
        return (self.zfactor * self.eBoson + pow(self.zfactor, 2) * (self.ekb + self.grho)) / (
            2.0 * self.eBoson
        )

    def ResBosBECm(self):
        return (-1.0 * self.zfactor * self.eBoson + pow(self.zfactor, 2) * (self.ekb + self.grho)) / (
            -2.0 * self.eBoson
        )

    def gFermion(self, z):
        return (z + self.ekf + self.ef_muf) / (pow(z, 2) - pow(self.eFermi, 2))

    def ResFerp(self):
        return (self.eFermi + self.ekf + self.ef_muf) / (2.0 * self.eFermi)

    def ResFerm(self):
        return (-1.0 * self.eFermi + self.ekf + self.ef_muf) / (-2.0 * self.eFermi)

    def efDiag(self, k2):
        self.upd(k2)
        if self.isBEC:
            r1 = self.gFermion(self.iz0 + self.eBoson) * self.ResBosBECp() * nB(self.eBoson, self.beta)
            r2 = self.gFermion(self.iz0 - self.eBoson) * self.ResBosBECm() * nB(-1.0 * self.eBoson, self.beta)
            r3 = self.gBosonBEC(self.eFermi - self.iz0) * self.ResFerp() * nF(self.eFermi, self.beta)
            r4 = self.gBosonBEC(-1.0 * self.eFermi - self.iz0) * self.ResFerm() * nF(-1.0 * self.eFermi, self.beta)
            return np.real(r1 + r2 + r3 + r4) * pow(self.yval("h"), 2)
        r1 = self.gFermion(self.iz0 + self.eBosonThr) * nB(self.eBosonThr, self.beta)
        r2 = self.gBosonThr(self.eFermi - self.iz0) * self.ResFerp() * nF(self.eFermi, self.beta)
        r3 = self.gBosonThr(-1.0 * self.eFermi - self.iz0) * self.ResFerm() * nF(-1.0 * self.eFermi, self.beta)
        return np.real(r1 + r2 + r3) * pow(self.yval("h"), 2)


ydatakeysPrompt = ["eb", "ef", "g", "h", "dfac", "rhoF"]


class OuterBCSFermion(RGSector):
    def __init__(self, prsdata: RGState, mf, beta, gFF, mu, cutoff, lpar=0.0, h=1.0):
        self.efSwitch = True
        self.muf = mu
        self.mf = mf
        self.efx2mf = 0.0
        self.kF2 = self.muf * 2.0 * self.mf
        self.mf_div_2pi = self.mf / (2.0 * np.pi)
        self.cutoff2 = pow(cutoff * np.exp(-1.0 * lpar), 2)
        self.beta = beta
        eb = -1.0 * pow(h, 2) / gFF - 2.0 * self.muf
        assert eb > 0, "already condensed"
        couplings = (
            CouplingSpec("eb", eb, Key.EB),
            CouplingSpec("ef", 0.0, Key.EF),
            CouplingSpec("g", 1e-4, Key.G),
            CouplingSpec("h", h, Key.H),
            CouplingSpec("dfac", 1.0, Key.DFAC),
            CouplingSpec("rhoF", 0.0, Key.RHO_F),
        )
        super().__init__(prsdata, couplings)
        self.ydatakeys = self.coupling_names
        self.yval = lambda x: self.ydata.value(x)
        self.lpar0 = lpar
        self.lpar = lpar
        self.isBEC = False
        self.dEfdiagObj = efDiag(self.ydata, self.beta, self.muf, self.mf, self.isBEC)
        self.dEfdiagfunc = self.dEfdiagObj.efDiag

    def BECcritUpd(self, isBEC):
        self.isBEC = isBEC
        self.dEfdiagObj.isBEC = isBEC

    def k_scale(self, lpar):
        self.efx2mf = self.yval("ef") * (2.0 * self.mf)
        self.kF2new = max(self.kF2 - self.efx2mf, 0.0)
        self.explpar = np.exp(-2.0 * (lpar - self.lpar0))
        self.k2p = (self.cutoff2 - self.kF2new) * self.explpar + self.kF2new
        self.holeinvolv = self.k2p < 2.0 * self.kF2new
        self.holetheta = thetaFunc(2.0 * self.kF2new - self.k2p)
        self.k2h = (2.0 * self.kF2new - self.k2p) if self.holeinvolv else 0.0
        self.k2Boson = self.cutoff2 * self.explpar
        self.bosonInvolv = self.k2Boson < self.k2h

    def dosCoeff(self):
        """Particle/hole DOS share the k_scale k^2 shell."""
        self.dosp0 = (self.k2p - self.kF2new) / (2.0 * np.pi)
        self.dosh0 = (self.k2p - self.kF2new) / (2.0 * np.pi) if self.holeinvolv else 0.0
        self.dosb = self.k2Boson / (2.0 * np.pi) if self.bosonInvolv else 0.0

    def dosCoeff_dEf(self):
        if self.efSwitch:
            diagp = self.dEfdiagfunc(self.k2p)
            diagh = self.dEfdiagfunc(self.k2h) if self.holeinvolv else 0.0
            numer = diagp * self.dosp0 + diagh * self.dosh0
            denom = 1.0 - self.mf_div_2pi * (diagp - diagh)
            dEf = np.real(numer / (denom + 1e-2j))
            self.dosp = self.dosp0 + dEf * self.mf_div_2pi
            self.dosh = (self.dosh0 - dEf * self.mf_div_2pi) if self.holeinvolv else 0.0
            return dEf
        self.dosp = self.dosp0
        self.dosh = self.dosh0
        return 0.0

    def upd(self, lpar):
        self.k_scale(lpar)
        self.dosCoeff()
        self.dEf = self.dosCoeff_dEf()
        self.ek0p = self.k2p / (2.0 * self.mf) - (self.muf - self.yval("ef"))
        self.ek0h = self.k2h / (2.0 * self.mf) - (self.muf - self.yval("ef"))
        self.rho = 0.0
        if self.isBEC:
            self.rho = self.yval("rho") / self.yval("avv")
        self.ekp_cp = np.sqrt(pow(self.ek0p, 2) + self.rho * pow(self.yval("h") * self.yval("dfac"), 2))
        self.ekh_cp = np.sqrt(pow(self.ek0h, 2) + self.rho * pow(self.yval("h") * self.yval("dfac"), 2))
        self.nf_p = nF(self.ekp_cp, self.beta)
        self.nf_h = nF(self.ekh_cp, self.beta)
        self.secsq_p = 4.0 * self.nf_p * (1.0 - self.nf_p)
        self.tanh_p = 1.0 - 2.0 * self.nf_p
        self.secsq_h = 4.0 * self.nf_h * (1.0 - self.nf_h)
        self.tanh_h = 1.0 - 2.0 * self.nf_h

    def ebDiag(self):
        if self.beta * self.ekp_cp > 1e-6:
            coeff_p = self.dosp * pow(self.yval("h"), 2)
            self.ebrs_p = -1.0 * coeff_p * self.tanh_p / (2.0 * self.ekp_cp)
        else:
            self.ebrs_p = -1.0 * pow(self.yval("h"), 2) * self.beta * self.dosp / 4.0
        if self.holeinvolv:
            if self.beta * self.ekh_cp > 1e-6:
                coeff_h = self.dosh * pow(self.yval("h"), 2)
                self.ebrs_h = -1.0 * coeff_h * self.tanh_h / (2.0 * self.ekh_cp)
                return self.ebrs_p + self.ebrs_h
            self.ebrs_h = -1.0 * self.beta * pow(self.yval("h"), 2) * self.dosh / 4.0
            return self.ebrs_p + self.ebrs_h
        self.ebrs_h = 0.0
        return self.ebrs_p

    def gDiag(self):
        if abs(self.beta * self.ekp_cp) > 1e-6:
            coeff_p = self.dosp * pow(self.yval("h"), 4)
            term1p = -1.0 * self.secsq_p * self.beta * self.ekp_cp
            term2p = 2.0 * self.tanh_p
            rs_p = coeff_p * (term1p + term2p) / (8.0 * pow(self.ekp_cp, 3))
        else:
            rs_p = self.dosp * pow(self.yval("h"), 4) * pow(self.beta, 3) / 48.0
        if self.holeinvolv:
            if abs(self.beta * self.ekh_cp) > 1e-6:
                coeff_h = self.dosh * pow(self.yval("h"), 4)
                term1h = -1.0 * self.secsq_h * self.beta * self.ekh_cp
                term2h = 2.0 * self.tanh_h
                rs_h = coeff_h * (term1h + term2h) / (8.0 * pow(self.ekh_cp, 3))
                return rs_h + rs_p
            rs_h = self.dosh * pow(self.yval("h"), 4) * pow(self.beta, 3) / 48.0
            return rs_h + rs_p
        return rs_p

    def dZDiag(self):
        dZp = self.ebrs_p * self.ek0p / (-2.0 * pow(self.ekp_cp, 2))
        if self.holeinvolv:
            dZh = self.ebrs_h * self.ek0h / (-2.0 * pow(self.ekh_cp, 2))
            return dZp + dZh
        return dZp

    def dhRen(self):
        dZ = self.dZDiag()
        return -(1.0 / 2.0) * dZ * self.yval("h")

    def dDfac(self):
        if self.isBEC:
            ek_bos = self.k2Boson / (4.0 * self.mf)
            ek_sf = np.sqrt(
                ek_bos * (ek_bos + 2.0 * self.yval("g") * self.yval("avv") * self.yval("rho"))
                * self.yval("all")
                / self.yval("avv")
            )
            coeff = self.yval("all") * self.mf / (4.0 * np.pi * self.yval("rho"))
            coth = -2.0 * nB(ek_sf, self.beta)
            return -1.0 * coeff * ek_sf * coth * self.yval("dfac")
        return 0.0

    def drhoF(self):
        integ = (1.0 / 2.0) * (1.0 - (1.0 - 2.0 * self.nf_p) * self.ek0p / self.ekp_cp) * self.dosp
        if self.holeinvolv:
            integ += (1.0 / 2.0) * (1.0 - (1.0 - 2.0 * self.nf_h) * self.ek0h / self.ekh_cp) * self.dosh
        return integ

    def contribute(self, l, dy: RGState):
        self.upd(l)
        dy.data["eb"] = self.ebDiag()
        dy.data["h"] = self.dhRen()
        dy.data["g"] = self.gDiag()
        dy.data["ef"] = self.dEf
        dy.data["dfac"] = self.dDfac()
        dy.data["rhoF"] = self.drhoF()

<<<<<<< HEAD
    def contribute_post(self, l: float, dy: RGState) -> RGState | None:
=======
    def contribute_post(self, l: float, dy: RGState) -> None:
>>>>>>> 819e16bd16788c3a8c3fad90a05952e5bb7afc60
        return super().contribute_post(l, dy)

    def dylst_BCSonly(self, l, ylst):
        self.ydata.update(ylst)
        dy = self.ydata.zeroVecGen()
        self.dylst(l, dy)
        return dy.ylst()


def dh2dZ(dh, h):
    return -2.0 * dh / h
