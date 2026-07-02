"""BCS-side RG orchestrator: thermal boson flow with optional BEC branch.

bareInt uses the BCS two-channel form with cutoff-dependent log argument:
  1 / [(m/2pi) * log(sqrt(m*eb) / sqrt(cutoff^2 + m*eb))]
Do not unify with bec_action.bareInt, which uses a different formula.
"""

import numpy as np
import scipy.integrate as itg

from bcs import fermion, quantum, thermal
from bcs.keys import Key, key_index
from bcs.merge_hooks import bec_clamp_hook, h_renorm_hook_bec, make_h_renorm_hook_thr, kt_hook_bec
from bcs.mu_root import bisect_with_guess
from bcs.sector import compose_sectors
from bcs.state import RGState

_bcs_mu_hint: dict[tuple[float, float, float, float], float] = {}

bareInt = lambda eb, m, cutoff: 1.0 / (
    (m / (2.0 * np.pi)) * (np.log(np.sqrt(m * eb) / np.sqrt(pow(cutoff, 2) + m * eb)))
)


class BCSAction:
    def __init__(self, eb0, beta, mu, cutoff, mf=1.0, h=40.0):
        self.efSwitch = False
        self.KTswitch = True

        self.lpar = 0.0
        self.mf = mf
        self.mb = 2.0 * self.mf
        self.muf = mu
        self.beta = beta
        self.cutoff = cutoff
        self.h0 = h
        self.h = h
        self.gFF0 = bareInt(eb0, self.mf, cutoff)
        self.gFF0 = 1.0 / (1.0 / self.gFF0 + eb0 / pow(self.h, 2))

        self.ydata = RGState()
        self.bcsFer = fermion.OuterBCSFermion(
            self.ydata, self.mf, self.beta, self.gFF0, self.muf, self.cutoff, self.lpar, self.h
        )
        self.bcsFer.efSwitch = self.efSwitch
        self.thrBos = thermal.ThermalBoson(
            self.ydata,
            self.mb,
            self.cutoff,
            self.beta,
            self.ydata.value(Key.G),
            self.ydata.value(Key.EB),
            0.0,
            self.lpar,
        )

        assert self.ydata.keysUpd is not None, "ydata.keysUpd is not updated"
        keys = self.ydata.keysUpd
        self.thrgidx = key_index(keys, Key.G)
        self.threbidx = key_index(keys, Key.EB)
        self.efidx = key_index(keys, Key.EF)
        self.hidx = key_index(keys, Key.H)

        self.terminFuncThr = thermal.ThrterminFunc(self.mb, self.beta, self.thrgidx, self.threbidx)
        self.y0Thr = self.ydata.ylst()

        self.solThr = itg.solve_ivp(
            self.thrEqn,
            (np.double(0.0), np.double(20.0)),
            self.y0Thr,
            method="LSODA",
            rtol=1e-7,
            atol=1e-7,
            min_step=1e-12,
            events=self.terminFuncThr,
        )

        self.ydata.update(self.solThr.y[:, -1])
        self.becShift = self.solThr.status == 1

        if self.becShift:
            self.rho_init = -1.0 * self.solThr.y[self.threbidx, -1] / self.solThr.y[self.thrgidx, -1]
            self.becBos = quantum.QuantumAction(
                self.ydata,
                self.mb,
                self.cutoff,
                self.solThr.t[-1],
                self.beta,
                self.solThr.y[self.thrgidx, -1],
                self.rho_init,
                self.KTswitch,
            )
            self.bcsFer.BECcritUpd(True)
            keys = self.ydata.keysUpd
            self.becrhoidx = key_index(keys, Key.RHO)
            self.becallidx = key_index(keys, Key.ALL)
            self.becavvidx = key_index(keys, Key.AVV)
            self.terminFuncBEC = quantum.BECterminFunc(self.mb, self.beta, self.becrhoidx, self.becallidx, self.becavvidx)
            self.y0BEC = self.ydata.ylst()
            self.solBEC = itg.solve_ivp(
                self.spfEqn,
                (np.double(self.solThr.t[-1]), np.double(20.0)),
                self.y0BEC,
                method="LSODA",
                rtol=1e-7,
                atol=1e-7,
                min_step=1e-12,
                events=self.terminFuncBEC,
            )

    def thrEqn(self, l, ylst):
        self.ydata.update(ylst)
        return compose_sectors(
            self.ydata,
            l,
            [self.bcsFer, self.thrBos],
            hooks=[make_h_renorm_hook_thr(self.h0)],
        )

    def spfEqn(self, l, ylst):
        self.ydata.update(ylst)
        bec_clamp_hook(self.ydata, self.ydata.zero_like())
        return compose_sectors(
            self.ydata,
            l,
            [self.becBos, self.bcsFer],
            hooks=[h_renorm_hook_bec, kt_hook_bec],
        )

    def FinalRhoSF(self):
        if self.becShift and self.solBEC.status == 0:
            return np.nan_to_num(float(self.solBEC.y[self.becallidx, -1]), nan=0.0, posinf=0.0, neginf=0.0)
        return float(0.0)

    def FinalNum(self):
        keys = self.ydata.keysUpd
        assert keys is not None, "Check ydata.keysUpd"
        rho_f_idx = key_index(keys, Key.RHO_F)
        nthrm_idx = key_index(keys, Key.NTHRM)
        if self.becShift:
            ferNum = self.solBEC.y[rho_f_idx, -1]
        else:
            ferNum = self.solThr.y[rho_f_idx, -1]
        if self.becShift:
            bosNum = self.solThr.y[nthrm_idx, -1]
            bosNum += max(self.solBEC.y[self.becrhoidx, -1], 0.0) #* pow(
            #    self.solBEC.y[self.hidx, -1] / self.h0, 2
            #)
        else:
            bosNum = self.solThr.y[nthrm_idx, -1]
        return np.nan_to_num(ferNum * 2.0 + 2.0 * bosNum, nan=0.0, posinf=0.0, neginf=0.0)


def findMu(targetNum, eb, beta, cutoff, mass, mu_guess=None, use_hint_cache=True):
    mu0 = targetNum * np.pi / mass
    lo = -1.0 * eb / 2.0 + 1e-7
    hi = mu0 * 3.0
    cache_key = (float(eb), float(cutoff), float(mass), float(targetNum))
    if mu_guess is None and use_hint_cache:
        mu_guess = _bcs_mu_hint.get(cache_key)

    def func(mui):
        return BCSAction(eb, beta, mui, cutoff, mass).FinalNum() - targetNum

    def on_bracket_fail(_, __, lft, rht):
        print(f"{eb:.2f},\t{beta:.2f},\t{lft:.2f},\t{rht:.2f}")

    root = bisect_with_guess(func, lo, hi, xtol=1e-6, mu_guess=mu_guess, on_bracket_fail=on_bracket_fail)
    if use_hint_cache:
        _bcs_mu_hint[cache_key] = root
    return root
