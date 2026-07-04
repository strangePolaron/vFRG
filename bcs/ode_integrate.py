"""RG flow ODE integration with a configurable RHS evaluation budget.

Each ``solve_ivp`` call in ``BECAction`` / ``BCSAction`` routes through
``solve_rg_ivp``, which stops integration when ``max_ode_steps`` RHS
evaluations are exceeded (default ``DEFAULT_MAX_ODE_STEPS`` = 5000).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import scipy.integrate as itg

DEFAULT_MAX_ODE_STEPS = 5000


class OdeStepLimitError(Exception):
    """RHS evaluation budget for one ``solve_ivp`` call exceeded."""

    def __init__(self, count: int):
        self.count = count
        super().__init__(f"ODE RHS limit exceeded ({count} evaluations)")


@dataclass(frozen=True)
class RgOdeResult:
    sol: Any
    step_limit_hit: bool
    nfev: int


class _CountingRhs:
    def __init__(self, fun: Callable, limit: int):
        self.fun = fun
        self.limit = limit
        self.count = 0
        self.last_t: float | None = None
        self.last_y: np.ndarray | None = None

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        self.count += 1
        if self.count > self.limit:
            raise OdeStepLimitError(self.count)
        out = self.fun(t, y)
        self.last_t = float(t)
        self.last_y = np.asarray(y, dtype=float).copy()
        return out


def _failed_solution(
    y0: np.ndarray,
    t0: float,
    counter: _CountingRhs,
    *,
    message: str = "step limit",
) -> SimpleNamespace:
    if counter.last_y is not None:
        y = counter.last_y.reshape(-1, 1)
        t = np.array([counter.last_t if counter.last_t is not None else t0], dtype=float)
    else:
        y = np.asarray(y0, dtype=float).reshape(-1, 1)
        t = np.array([t0], dtype=float)
    return SimpleNamespace(
        t=t,
        y=y,
        sol=None,
        t_events=None,
        y_events=None,
        nfev=counter.count,
        njev=0,
        nlu=0,
        status=-1,
        message=message,
        success=False,
    )


def solve_rg_ivp(
    fun: Callable,
    t_span: tuple[float, float],
    y0: np.ndarray | list[float],
    *,
    events=None,
    max_ode_steps: int = DEFAULT_MAX_ODE_STEPS,
    method: str = "LSODA",
    rtol: float = 1e-7,
    atol: float = 1e-7,
    min_step: float = 1e-12,
    **kwargs: Any,
) -> RgOdeResult:
    """Integrate an RG flow with a hard cap on RHS evaluations."""
    counter = _CountingRhs(fun, max_ode_steps)
    t_span = (float(t_span[0]), float(t_span[1]))
    y0_arr = np.asarray(y0, dtype=float)
    try:
        sol = itg.solve_ivp(
            counter,
            t_span,
            y0_arr,
            method=method,
            rtol=rtol,
            atol=atol,
            min_step=min_step,
            events=events,
            **kwargs,
        )
        nfev = int(getattr(sol, "nfev", counter.count))
        return RgOdeResult(sol=sol, step_limit_hit=False, nfev=nfev)
    except OdeStepLimitError:
        sol = _failed_solution(y0_arr, t_span[0], counter)
        return RgOdeResult(sol=sol, step_limit_hit=True, nfev=counter.count)
