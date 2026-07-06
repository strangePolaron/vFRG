"""BEC-side RG orchestrator: pure quantum boson flow via QuantumAction.

bareInt uses the BEC single-channel form:
  1 / [(m/2pi) * log(sqrt(m*eb) / cutoff)]
Do not unify with bcs_action.bareInt, which includes a cutoff-dependent correction.
"""

import numpy as np

from bcs.ode_integrate import DEFAULT_MAX_ODE_STEPS, solve_rg_ivp  # [ode-max-steps]
from bcs import quantum
from bcs.keys import Key, key_index
from bcs.merge_hooks import kt_hook_bec
from bcs.mu_root import bisect_with_guess
from bcs.ode_safe import clamp_bec_state, classify_bec_termination, guard_coupling_positive  # [mu-relax-ode-safe]
from bcs.ode_safe import TerminationKind  # [mu-relax-ode-safe]
from bcs.sector import compose_sectors
from bcs.state import RGState

_bec_mu_hint: dict[tuple[float, float, float], float] = {}

bareInt = lambda eb, m, cutoff: 1.0 / (
    (m / (2.0 * np.pi)) * (np.log(np.sqrt(m * eb) / cutoff))
)


class BECAction:
    def __init__(self, eb2boson0, beta, mu, m=1.0, max_ode_steps=None):
        self.KTSwitch = True
        self.beta = beta
        self.lpar = 0.0
        self.mb = m

        self.cutoff = 10.0 #np.sqrt(m * eb2boson0 - 1e-1)
        assert (self.mb * eb2boson0> pow(self.cutoff, 2)), "mb*eb should be larger than cutoff^2"
        self.g0 = bareInt(eb2boson0, self.mb, np.sqrt(pow(self.cutoff, 2)))
        self.eb2bosn_vac = eb2boson0
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
        self.step_limit_hit = False  # [ode-max-steps]
        try:
            # [ode-max-steps]
            result = solve_rg_ivp(
                self.eqn,
                (np.double(0.0), np.double(40.0)),
                self.y0,
                method="LSODA",
                rtol=1e-7,
                atol=1e-7,
                min_step=1e-12,
                events=self.terminFunc,
                max_ode_steps=max_ode_steps or DEFAULT_MAX_ODE_STEPS,
            )
            self.sol = result.sol
            self.step_limit_hit = result.step_limit_hit
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
        #if np.isnan(ylst).any():
        #    print(ylst)
        #    a = input()
        #np.nan_to_num(ylst, nan=0.0, posinf=0.0, neginf=0.0)

        # [mu-relax-ode-safe] clamp rho/avv for event overshoot; all only guarded if non-positive
        ylst = clamp_bec_state(ylst, (self.rhoidx, self.avvidx))
        ylst = guard_coupling_positive(ylst, (self.allidx,))
        self.ydata.update(ylst)
        return compose_sectors(self.ydata, l, [self.quantumbec], [kt_hook_bec],)

    def FinalRhoSF(self):
        # [mu-relax-ode-safe] all-floor stop: SF exhausted, report all as 0; completed: final all
        kind = classify_bec_termination(self)
        if kind in (
            TerminationKind.ALL_FLOOR,
            TerminationKind.RHO_FLOOR,
            TerminationKind.AVV_FLOOR,
            TerminationKind.FAILED,
        ):
            return float(0.0)
        if int(self.sol.status) == -1:
            return float(0.0)
        assert self.ydata.keysUpd is not None, "ydata.keysUpd is not updated"
        return np.nan_to_num((self.sol.y[self.allidx, -1]), nan=0.0, posinf=0.0, neginf=0.0)

    def FinalNum(self):
        assert self.ydata.keysUpd is not None, "ydata.keysUpd is not updated (FinalNum)"
        return np.nan_to_num(self.sol.y[self.rhoidx, -1], nan=0.0, posinf=0.0, neginf=0.0)


def findMu(targetNum, ebBos, beta, mass, mu_guess=None, use_hint_cache=True):
    #mu0 = ebBos/2.0 - 1e-7 #min(20.0 * targetNum * np.pi / mass, ebBos/2. - 1e-3)
    lo = ebBos * 1e-30
    hi = ebBos*1e-1
    cache_key = (float(ebBos), float(mass), float(targetNum))
    if mu_guess is None and use_hint_cache:
        mu_guess = _bec_mu_hint.get(cache_key)

    def func(mui):
        return BECAction(ebBos, beta, mui, mass).FinalNum() - targetNum
    #assert func(lo) * func(hi)<0.0, f"lo:{func(lo)},\thi:{func(hi)}"
    root = bisect_with_guess(func, lo, hi, xtol=1e-5, mu_guess=mu_guess)
    if use_hint_cache:
        _bec_mu_hint[cache_key] = root
    return root
