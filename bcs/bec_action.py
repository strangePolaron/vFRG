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
from bcs.sector import compose_sectors
from bcs.state import RGState

_bec_mu_hint: dict[tuple[float, float, float], float] = {}

bareInt = lambda eb, m, cutoff: 1.0 / (
    (m / (2.0 * np.pi)) * (np.log(np.sqrt(m * eb) / cutoff))
)


class BECAction:
    def __init__(self, eb2boson0, beta, mu, m=1.0):
        self.KTSwitch = True
        self.beta = beta
        self.lpar = 0.0
        self.mb = m

        self.cutoff = np.sqrt(m * eb2boson0 + 0. * self.mb * mu)
        self.g0 = bareInt(eb2boson0, self.mb, np.sqrt(pow(self.cutoff, 2) - 2.0 * self.mb * mu))

        self.mub = mu
        self.rho_init = max(self.mub / self.g0, 0)

        self.ydata = RGState()
        self.quantumbec = quantum.QuantumAction(
            self.ydata, self.mb, self.cutoff, self.lpar, self.beta, self.g0, self.rho_init, self.KTSwitch
        )

        #print(f"healLength:\t{self.quantumbec.healLength()},\tRScutoff:\t{2.*np.pi/self.cutoff}")

        assert self.ydata.keysUpd is not None, "ydata.keysUpd is not updated"
        keys = self.ydata.keysUpd
        self.rhoidx = key_index(keys, Key.RHO)
        self.allidx = key_index(keys, Key.ALL)
        self.avvidx = key_index(keys, Key.AVV)

        self.terminFunc = quantum.BECterminFunc(self.mb, self.beta, self.rhoidx, self.allidx, self.avvidx)
        self.y0 = self.ydata.ylst()
        try:
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
        except ValueError:
            print(f"mu:\t{self.mub},\ty:\t{self.ydata.ylst()}")

    def eqn(self, l, ylst):
        """
        if ylst[self.allidx]<1e-3:
            ylst[self.allidx] = 1e-3 * np.exp(ylst[self.allidx])
        if ylst[self.avvidx]<1e-3:
            ylst[self.avvidx] = 1e-3 * np.exp(ylst[self.avvidx])
        if ylst[self.rhoidx]<1e-3:
            ylst[self.rhoidx] = 1e-3 * np.exp(ylst[self.rhoidx])
        """
        np.nan_to_num(ylst, nan=0.0, posinf=0.0, neginf=0.0)

        self.ydata.update(ylst)
        return compose_sectors(self.ydata, l, [self.quantumbec])

    def FinalRhoSF(self):
        if self.sol.status == 1 or self.sol.status == -1:
            return float(0.0)
        assert self.ydata.keysUpd is not None, "ydata.keysUpd is not updated"
        return float(self.sol.y[self.allidx, -1])

    def FinalNum(self):
        assert self.ydata.keysUpd is not None, "ydata.keysUpd is not updated (FinalNum)"
        return self.sol.y[self.rhoidx, -1]


def findMu(targetNum, ebBos, beta, mass, mu_guess=None, use_hint_cache=True):
    mu0 = min(20.0 * targetNum * np.pi / mass, ebBos/2. - 1e-3)
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
