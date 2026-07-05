"""Relaxed secant root finding for chemical potential with branch recovery.

Each ODE-backed evaluation is expensive. Use ``RelaxLimits`` (default ``max_evals=25``)
to cap total work per ``findMu_relax_*`` call. Exceeding the budget raises ``MuRootError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from bcs.bec_action import BECAction
from bcs.bcs_action import BCSAction
from bcs.mu_root import bisect_with_guess
from bcs.ode_safe import (
    TerminationKind,
    classify_bec_termination,
    classify_bcs_bec_termination,
    is_bcs_integration_valid,
    is_bec_integration_valid,
)

@dataclass(frozen=True)
class RelaxLimits:
    max_evals: int = 25
    max_secant_iter: int = 10
    max_branch_steps: int = 15
    max_bracket_samples: int = 12
    enable_bisect_fallback: bool = True


DEFAULT_RELAX_LIMITS = RelaxLimits()


class NoValidBranchError(RuntimeError):
    """No valid integration branch found while decreasing mu."""


class MuRootError(RuntimeError):
    """Relaxed mu solver failed to converge or exceeded evaluation budget."""

    def __init__(
        self,
        message: str,
        *,
        phase: str | None = None,
        eval_used: int | None = None,
        eval_limit: int | None = None,
        last_mu: float | None = None,
        last_residual: float | None = None,
    ):
        super().__init__(message)
        self.phase = phase
        self.eval_used = eval_used
        self.eval_limit = eval_limit
        self.last_mu = last_mu
        self.last_residual = last_residual


@dataclass
class _EvalBudget:
    limit: int
    used: int = 0
    last_mu: float | None = None
    last_residual: float | None = None

    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def record(self, ev: MuEval) -> None:
        self.last_mu = ev.mu
        self.last_residual = ev.residual

    def check(self, phase: str) -> None:
        if self.used >= self.limit:
            raise self._exhausted(phase)

    def _exhausted(self, phase: str) -> MuRootError:
        return MuRootError(
            f"{phase}: max_evals={self.limit} exceeded ({self.used} used); "
            f"last_mu={self.last_mu}, last_residual={self.last_residual}",
            phase=phase,
            eval_used=self.used,
            eval_limit=self.limit,
            last_mu=self.last_mu,
            last_residual=self.last_residual,
        )

    def wrap(self, eval_fn: Callable[[float], MuEval]) -> Callable[[float], MuEval]:
        def wrapped(mu: float) -> MuEval:
            self.check("eval")
            self.used += 1
            ev = eval_fn(mu)
            self.record(ev)
            return ev

        return wrapped


@dataclass(frozen=True)
class MuEval:
    mu: float
    final_num: float
    residual: float
    ok: bool
    status: int | None
    reason: str | None
    termination: TerminationKind


def evaluate_mu_bec(eb: float, beta: float, mass: float, mu: float, target: float) -> MuEval:
    try:
        action = BECAction(eb, beta, mu, mass)
    except Exception:
        return MuEval(mu, 0.0, -target, False, None, "exception", TerminationKind.FAILED)

    if not hasattr(action, "sol"):
        return MuEval(mu, 0.0, -target, False, None, "exception", TerminationKind.FAILED)

    termination = classify_bec_termination(action)
    ok, reason = is_bec_integration_valid(action)
    final_num = float(action.FinalNum())
    return MuEval(
        mu=mu,
        final_num=final_num,
        residual=final_num - target,
        ok=ok,
        status=int(action.sol.status),
        reason=reason,
        termination=termination,
    )


def evaluate_mu_bcs(
    eb: float, beta: float, cutoff: float, mass: float, mu: float, target: float
) -> MuEval:
    try:
        action = BCSAction(eb, beta, mu, cutoff, mass)
    except Exception:
        return MuEval(mu, 0.0, -target, False, None, "exception", TerminationKind.FAILED)

    if not hasattr(action, "solThr"):
        return MuEval(mu, 0.0, -target, False, None, "exception", TerminationKind.FAILED)

    termination = classify_bcs_bec_termination(action)
    ok, reason = is_bcs_integration_valid(action)
    final_num = float(action.FinalNum())
    status = int(action.solBEC.status) if getattr(action, "becShift", False) else int(action.solThr.status)
    return MuEval(
        mu=mu,
        final_num=final_num,
        residual=final_num - target,
        ok=ok,
        status=status,
        reason=reason,
        termination=termination,
    )


def find_valid_mu(
    eval_fn: Callable[[float], MuEval],
    mu_init: float,
    lo: float,
    hi: float,
    *,
    step: float = 0.01,
    max_steps: int = 15,
) -> float:
    """Decrease mu until eval_fn reports a valid integration."""
    mu = min(max(float(mu_init), lo), hi)
    for _ in range(max_steps):
        if eval_fn(mu).ok:
            return mu
        mu *= 1.0 - step
        if mu <= lo:
            break
    mu = max(min(float(mu_init), hi), lo)
    for _ in range(max_steps):
        if eval_fn(mu).ok:
            return mu
        mu *= 1.0 + step
        if mu >= hi:
            break
    return mu_init
    #raise NoValidBranchError(f"No valid mu branch found in [{lo}, {hi}]")


def _second_mu_point(
    eval_fn: Callable[[float], MuEval],
    mu0: float,
    lo: float,
    hi: float,
    *,
    max_branch_steps: int,
) -> tuple[float, MuEval, float, MuEval]:
    span = hi - lo
    for delta in (
        max(abs(mu0) * 0.01, span / 20.0, 1e-6),
        max(abs(mu0) * 0.005, span / 40.0, 1e-6),
        -max(abs(mu0) * 0.01, span / 20.0, 1e-6),
    ):
        mu1 = min(max(mu0 + delta, lo), hi)
        if abs(mu1 - mu0) <= 1e-15:
            continue
        ev0 = eval_fn(mu0)
        ev1 = eval_fn(mu1)
        if ev0.ok and ev1.ok:
            return mu0, ev0, mu1, ev1
    mu1 = find_valid_mu(eval_fn, mu0 * (1.0 - 0.02), lo, hi, max_steps=max_branch_steps)
    ev0 = eval_fn(mu0)
    ev1 = eval_fn(mu1)
    if not ev0.ok or not ev1.ok:
        raise NoValidBranchError("Could not seed secant with two valid evaluations")
    return mu0, ev0, mu1, ev1


def _find_valid_bracket(
    eval_fn: Callable[[float], MuEval],
    lo: float,
    hi: float,
    samples: int,
) -> tuple[float, float]:
    points: list[MuEval] = []
    for mu in _linspace(lo, hi, samples):
        ev = eval_fn(mu)
        if ev.ok:
            points.append(ev)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if points[i].residual * points[j].residual <= 0.0:
                return points[i].mu, points[j].mu
    raise MuRootError(
        "bracket: no valid bracket with opposite-sign residuals",
        phase="bracket",
    )


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    if n <= 1:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


def _residual_for_bisect(
    eval_fn: Callable[[float], MuEval],
    lo: float,
    hi: float,
    *,
    max_branch_steps: int,
) -> Callable[[float], float]:
    def func(mu: float) -> float:
        ev = eval_fn(mu)
        if ev.ok:
            return ev.residual
        safe_mu = find_valid_mu(eval_fn, mu, lo, hi, max_steps=max_branch_steps)
        return eval_fn(safe_mu).residual

    return func


def _build_limits(
    limits: RelaxLimits | None,
    *,
    max_evals: int | None,
    max_secant_iter: int | None,
    max_branch_steps: int | None,
) -> RelaxLimits:
    base = limits or DEFAULT_RELAX_LIMITS
    return RelaxLimits(
        max_evals=max_evals if max_evals is not None else base.max_evals,
        max_secant_iter=max_secant_iter if max_secant_iter is not None else base.max_secant_iter,
        max_branch_steps=max_branch_steps if max_branch_steps is not None else base.max_branch_steps,
        max_bracket_samples=base.max_bracket_samples,
        enable_bisect_fallback=base.enable_bisect_fallback,
    )


def relaxed_find_mu(
    eval_fn: Callable[[float], MuEval],
    lo: float,
    hi: float,
    target: float,
    *,
    mu_guess: float | None = None,
    xtol: float = 1e-5,
    limits: RelaxLimits | None = None,
    max_evals: int | None = None,
    max_secant_iter: int | None = None,
    max_branch_steps: int | None = None,
) -> float:
    """Damped secant on valid evaluations with optional bisection fallback."""
    del target  # encoded in eval_fn residuals

    lim = _build_limits(
        limits,
        max_evals=max_evals,
        max_secant_iter=max_secant_iter,
        max_branch_steps=max_branch_steps,
    )
    budget = _EvalBudget(lim.max_evals)
    eval_fn = budget.wrap(eval_fn)

    seed = mu_guess if mu_guess is not None else hi
    try:
        mu0 = find_valid_mu(
            eval_fn, seed, lo, hi, max_steps=lim.max_branch_steps
        )
    except NoValidBranchError as exc:
        raise MuRootError(
            f"branch_recovery: {exc}",
            phase="branch_recovery",
            eval_used=budget.used,
            eval_limit=lim.max_evals,
            last_mu=budget.last_mu,
            last_residual=budget.last_residual,
        ) from exc

    ev0 = eval_fn(mu0)
    if abs(ev0.residual) <= xtol:
        return mu0

    try:
        mu0, ev0, mu1, ev1 = _second_mu_point(
            eval_fn, mu0, lo, hi, max_branch_steps=lim.max_branch_steps
        )
    except NoValidBranchError as exc:
        raise MuRootError(
            f"secant_seed: {exc}",
            phase="secant_seed",
            eval_used=budget.used,
            eval_limit=lim.max_evals,
            last_mu=budget.last_mu,
            last_residual=budget.last_residual,
        ) from exc

    if abs(ev1.residual) <= xtol:
        return mu1

    last_good = mu1
    stagnation = 0
    for _ in range(lim.max_secant_iter):
        denom = ev1.residual - ev0.residual
        if abs(denom) <= 1e-15:
            break
        mu_new = mu1 - ev1.residual * (mu1 - mu0) / denom
        mu_new = min(max(mu_new, lo), hi)

        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.125):
            trial = mu1 + alpha * (mu_new - mu1) if alpha < 1.0 else mu_new
            trial = min(max(trial, lo), hi)
            try:
                ev_trial = eval_fn(trial)
            except MuRootError:
                raise
            if ev_trial.ok and abs(ev_trial.residual) <= xtol:
                return trial
            if ev_trial.ok:
                mu0, ev0, mu1, ev1 = mu1, ev1, trial, ev_trial
                last_good = trial
                accepted = True
                stagnation = 0
                break
            damped = min(max(last_good - alpha * (trial - last_good), lo), hi)
            try:
                ev_damped = eval_fn(damped)
            except MuRootError:
                raise
            if ev_damped.ok and abs(ev_damped.residual) <= xtol:
                return damped
            if ev_damped.ok:
                mu0, ev0, mu1, ev1 = mu1, ev1, damped, ev_damped
                last_good = damped
                accepted = True
                stagnation = 0
                break

        if not accepted:
            stagnation += 1
            if stagnation >= 3:
                break

    remaining = budget.remaining()
    if not lim.enable_bisect_fallback or remaining < 4:
        raise MuRootError(
            f"secant: stopped after {budget.used}/{lim.max_evals} evals "
            f"(remaining={remaining}, stagnation={stagnation})",
            phase="secant",
            eval_used=budget.used,
            eval_limit=lim.max_evals,
            last_mu=budget.last_mu,
            last_residual=budget.last_residual,
        )

    samples = min(lim.max_bracket_samples, remaining - 2)
    try:
        bracket_lo, bracket_hi = _find_valid_bracket(eval_fn, lo, hi, samples=samples)
    except MuRootError as exc:
        raise MuRootError(
            str(exc),
            phase="bracket",
            eval_used=budget.used,
            eval_limit=lim.max_evals,
            last_mu=budget.last_mu,
            last_residual=budget.last_residual,
        ) from exc

    if budget.remaining() < 2:
        raise budget._exhausted("bisect")

    func = _residual_for_bisect(
        eval_fn, lo, hi, max_branch_steps=lim.max_branch_steps
    )
    try:
        return bisect_with_guess(func, bracket_lo, bracket_hi, xtol=xtol)
    except (ValueError, MuRootError) as exc:
        raise MuRootError(
            f"bisect: {exc}",
            phase="bisect",
            eval_used=budget.used,
            eval_limit=lim.max_evals,
            last_mu=budget.last_mu,
            last_residual=budget.last_residual,
        ) from exc


def findMu_relax_bec(
    targetNum: float,
    ebBos: float,
    beta: float,
    mass: float,
    mu_guess: float | None = None,
    *,
    max_evals: int = 25,
    max_secant_iter: int = 10,
    max_branch_steps: int = 15,
) -> float:
    mu0 = ebBos / 2.0 - 1e-5
    lo = ebBos * 0.0002
    hi = mu0

    def eval_fn(mu: float) -> MuEval:
        return evaluate_mu_bec(ebBos, beta, mass, mu, targetNum)

    return relaxed_find_mu(
        eval_fn,
        lo,
        hi,
        targetNum,
        mu_guess=mu_guess,
        xtol=1e-5,
        max_evals=max_evals,
        max_secant_iter=max_secant_iter,
        max_branch_steps=max_branch_steps,
    )


def findMu_relax_bcs(
    targetNum: float,
    eb: float,
    beta: float,
    cutoff: float,
    mass: float,
    mu_guess: float | None = None,
    *,
    max_evals: int = 100,
    max_secant_iter: int = 10,
    max_branch_steps: int = 15,
) -> float:
    mu0 = targetNum * np.pi / mass
    lo = -1.0 * eb / 2.0 #+ 1e-3
    
    hi = mu0 * 3.0

    def eval_fn(mu: float) -> MuEval:
        return evaluate_mu_bcs(eb, beta, cutoff, mass, mu, targetNum)

    #assert (eval_fn(lo).residual*eval_fn(hi).residual<0.0), f"func(lo)={eval_fn(lo).residual},\tfunc(hi)={eval_fn(hi).residual}"

    return relaxed_find_mu(
        eval_fn,
        lo,
        hi,
        targetNum,
        mu_guess=mu_guess,
        xtol=1e-6,
        max_evals=max_evals,
        max_secant_iter=max_secant_iter,
        max_branch_steps=max_branch_steps,
    )
