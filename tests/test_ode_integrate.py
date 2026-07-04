"""Tests for bcs.ode_integrate step-limit wrapper."""

import numpy as np

from bcs.ode_integrate import DEFAULT_MAX_ODE_STEPS, solve_rg_ivp


def test_solve_rg_ivp_completes_within_budget():
    def fun(_t, y):
        return -0.01 * y

    result = solve_rg_ivp(fun, (0.0, 1.0), np.array([1.0]), max_ode_steps=DEFAULT_MAX_ODE_STEPS)
    assert not result.step_limit_hit
    assert result.sol.success
    assert result.nfev <= DEFAULT_MAX_ODE_STEPS


def test_solve_rg_ivp_stops_at_step_limit():
    def fun(_t, y):
        return np.ones_like(y)

    result = solve_rg_ivp(
        fun,
        (0.0, 100.0),
        np.ones(3),
        max_ode_steps=10,
        max_step=0.5,
    )
    assert result.step_limit_hit
    assert result.sol.status == -1
    assert result.nfev <= 11
