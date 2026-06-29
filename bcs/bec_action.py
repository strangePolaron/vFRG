"""BEC-side RG orchestrator: pure quantum boson flow via QuantumAction.

bareInt uses the BEC single-channel form:
  1 / [(m/2pi) * log(sqrt(m*eb) / cutoff)]
Do not unify with bcs_action.bareInt, which includes a cutoff-dependent correction.
"""

import numpy as np
import scipy.integrate as itg

from bcs import quantum
from bcs.keys import Key, key_index
from bcs.mu_root import bisect_with_guess
from bcs.state import RGState

_bec_mu_hint: dict[tuple[float, float, float], float] = {}

bareInt = lambda eb, m, cutoff: 1.0 / ((m / (2.0 * np.pi)) * (np.log(np.sqrt(m * eb) / cutoff)))


class BECAction:
    def __init__(self, eb2boson0, beta, mu, m=1.0):
        self.KTSwitch = True
        self.beta = beta
        self.lpar = 0.0
        self.mb = m

        self.cutoff = np.sqrt(m * eb2boson0)
        self.g0 = bareInt(eb2boson0 + mu, self.mb, self.cutoff)

        self.mub = mu
        self.rho_init = max(self.mub / self.g0, 0)

        self.ydata = RGState()
        self.quantumbec = quantum.QuantumAction(
            self.ydata, self.mb, self.cutoff, self.lpar, self.beta, self.g0, self.rho_init, self.KTSwitch
        )

        assert self.ydata.keysUpd is not None, "ydata.keysUpd is not updated"
        keys = self.ydata.keysUpd
        self.rhoidx = key_index(keys, Key.RHO)
        self.allidx = key_index(keys, Key.ALL)

        self.terminFunc = quantum.BECterminFunc(self.mb, self.beta, self.rhoidx, self.allidx)
        self.y0 = self.ydata.ylst()

        self.sol = itg.solve_ivp(
            self.eqn,
            (np.double(0.0), np.double(20.0)),
            self.y0,
            method="LSODA",
            rtol=1e-7,
            atol=1e-7,
            min_step=1e-12,
            events=self.terminFunc,
        )

    def eqn(self, l, ylst):
        self.ydata.update(ylst)
        dy = self.ydata.zeroVecGen()
        self.quantumbec.dylst(l, dy)
        return dy.ylst()

    def FinalRhoSF(self):
        if self.sol.status == 1 or self.sol.status == -1:
            return float(0.0)
        assert self.ydata.keysUpd is not None, "ydata.keysUpd is not updated"
        return float(self.sol.y[self.allidx, -1])

    def FinalNum(self):
        assert self.ydata.keysUpd is not None, "ydata.keysUpd is not updated (FinalNum)"
        return self.sol.y[self.rhoidx, -1]


def findMu(targetNum, ebBos, beta, mass, mu_guess=None, use_hint_cache=True):
    mu0 = 10.0 * targetNum * np.pi / mass
    lo = 1e-3
    hi = mu0
    cache_key = (float(ebBos), float(mass), float(targetNum))
    if mu_guess is None and use_hint_cache:
        mu_guess = _bec_mu_hint.get(cache_key)

    def func(mui):
        return BECAction(ebBos, beta, mui, mass).FinalNum() - targetNum

    root = bisect_with_guess(func, lo, hi, xtol=1e-5, mu_guess=mu_guess)
    if use_hint_cache:
        _bec_mu_hint[cache_key] = root
    return root
