"""Tests for bcs.ode_safe clamp, classification, and validity helpers."""

import numpy as np

from bcs.ode_safe import (
    COUPLING_FLOOR,
    TerminationKind,
    clamp_bec_state,
    clamp_rg_state,
    classify_bec_termination_from_sol,
    guard_coupling_positive,
    is_bec_integration_valid,
    is_bcs_integration_valid,
    step_limit_hit,
    termination_is_valid,
)
from bcs.state import RGState


class _FakeSol:
    def __init__(self, status: int, y_final: list[float]):
        self.status = status
        self.y = np.array(y_final, dtype=float).reshape(-1, 1)


def test_clamp_bec_state_exp_keeps_values_positive():
    y = np.array([1.0, -0.5, 1e-5, 2.0, 1.0, 1.0])
    out = clamp_bec_state(y, (1, 2), mode="exp")
    assert out[1] > 0.0
    assert out[2] >= COUPLING_FLOOR
    assert np.isfinite(out).all()


def test_guard_coupling_positive_leaves_all_above_floor():
    y = np.array([0.5, 0.5, 5e-4])
    out = guard_coupling_positive(y, (2,))
    assert out[2] == 5e-4


def test_guard_coupling_positive_fixes_nonpositive_all():
    y = np.array([0.5, 0.5, -0.1])
    out = guard_coupling_positive(y, (2,))
    assert out[2] > 0.0


def test_clamp_bec_state_floor_mode():
    y = np.array([0.0, 1e-6, 1.0])
    out = clamp_bec_state(y, (0, 1), mode="floor")
    assert out[0] == COUPLING_FLOOR
    assert out[1] == COUPLING_FLOOR


def test_clamp_rg_state_in_place():
    state = RGState({"rho": 1e-6, "avv": 1.0, "all": 2.0})
    clamp_rg_state(state)
    assert state.data["rho"] >= COUPLING_FLOOR


def test_classify_all_floor():
    sol = _FakeSol(1, [0.05, COUPLING_FLOOR, 0.2])
    kind = classify_bec_termination_from_sol(sol, 0, 1, 2)
    assert kind is TerminationKind.ALL_FLOOR
    assert termination_is_valid(kind)


def test_classify_rho_floor():
    sol = _FakeSol(1, [COUPLING_FLOOR, 0.2, 0.05])
    kind = classify_bec_termination_from_sol(sol, 0, 2, 1)
    assert kind is TerminationKind.RHO_FLOOR
    assert not termination_is_valid(kind)


def test_classify_avv_floor():
    sol = _FakeSol(1, [0.05, COUPLING_FLOOR, 0.2])
    kind = classify_bec_termination_from_sol(sol, 0, 2, 1)
    assert kind is TerminationKind.AVV_FLOOR
    assert not termination_is_valid(kind)


def test_classify_completed():
    sol = _FakeSol(0, [0.05, 0.2, 0.3])
    kind = classify_bec_termination_from_sol(sol, 0, 2, 1)
    assert kind is TerminationKind.COMPLETED


def test_is_bec_integration_valid_accepts_completed_flow():
    from bcs.bec_action import BECAction
    from bcs.mu_relax import findMu_relax_bec

    p = {"eb": 1.0, "beta": 500.0, "mass": 1.0, "targetNum": 1.0 / (4.0 * np.pi)}
    mu = findMu_relax_bec(p["targetNum"], p["eb"], p["beta"], p["mass"])
    action = BECAction(p["eb"], p["beta"], mu, p["mass"])
    ok, reason = is_bec_integration_valid(action)
    assert ok is True
    assert reason is None


def test_step_limit_hit_marks_bec_invalid():
    from bcs.bec_action import BECAction

    p = {"eb": 1.0, "beta": 500.0, "mass": 1.0, "targetNum": 1.0 / (4.0 * np.pi)}
    action = BECAction(p["eb"], p["beta"], p["eb"] / 2.0 - 1e-5, p["mass"], max_ode_steps=5)
    assert step_limit_hit(action)
    ok, reason = is_bec_integration_valid(action)
    assert ok is False
    assert reason == "step_limit"


def test_step_limit_hit_marks_bcs_invalid():
    from bcs.bcs_action import BCSAction

    action = BCSAction(8.0, 5000.0, 10.0, 50.0, 1.0, max_ode_steps=5)
    assert step_limit_hit(action)
    ok, reason = is_bcs_integration_valid(action)
    assert ok is False
    assert reason == "step_limit"
