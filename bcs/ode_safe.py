"""Safe BEC coupling state handling for ODE integration."""

from __future__ import annotations

from enum import StrEnum
from typing import Iterable, Sequence

import numpy as np

from bcs.state import RGState

COUPLING_FLOOR = 1e-3
CLAMP_MODE = "exp"
_FLOOR_TOL_RATIO = 0.05


class TerminationKind(StrEnum):
    COMPLETED = "completed"
    ALL_FLOOR = "all_floor"
    RHO_FLOOR = "rho_floor"
    AVV_FLOOR = "avv_floor"
    FAILED = "failed"
    UNKNOWN = "unknown"


def _clamp_scalar(value: float, floor: float, mode: str) -> float:
    if value >= floor:
        return float(value)
    if mode == "floor":
        return floor
    if mode == "exp":
        return floor * float(np.exp(value / floor))
    raise ValueError(f"Unknown clamp mode: {mode}")


def clamp_bec_state(
    y: Sequence[float] | np.ndarray,
    indices: Iterable[int],
    *,
    floor: float = COUPLING_FLOOR,
    mode: str = CLAMP_MODE,
    copy: bool = True,
) -> np.ndarray:
    """Remap or floor BEC couplings so RHS evaluation stays finite."""
    out = np.array(y, dtype=float, copy=copy)
    for idx in indices:
        out[idx] = _clamp_scalar(float(out[idx]), floor, mode)
    return out


def guard_coupling_positive(
    y: Sequence[float] | np.ndarray,
    indices: Iterable[int],
    *,
    floor: float = COUPLING_FLOOR,
    copy: bool = True,
) -> np.ndarray:
    """Keep couplings finite and strictly positive without remapping physical floor approach."""
    out = np.array(y, dtype=float, copy=copy)
    for idx in indices:
        value = float(out[idx])
        if not np.isfinite(value):
            out[idx] = floor
        elif value <= 0.0:
            out[idx] = floor * float(np.exp(value / floor))
    return out


def clamp_rg_state(
    state: RGState,
    names: Sequence[str] = ("rho", "avv", "all"),
    *,
    floor: float = COUPLING_FLOOR,
    mode: str = CLAMP_MODE,
) -> None:
    """In-place clamp on RGState coupling dict entries."""
    for name in names:
        if name in state.data:
            state.data[name] = _clamp_scalar(float(state.data[name]), floor, mode)


def classify_bec_termination_from_sol(
    sol,
    rhoidx: int,
    allidx: int,
    avvidx: int,
    floor: float = COUPLING_FLOOR,
) -> TerminationKind:
    """Classify which coupling limited a BEC-phase integration."""
    if sol is None:
        return TerminationKind.FAILED

    status = int(sol.status)
    if status == -1:
        return TerminationKind.FAILED

    y_final = sol.y[:, -1]
    if not np.all(np.isfinite(y_final)):
        return TerminationKind.FAILED

    if status == 0:
        return TerminationKind.COMPLETED

    if status != 1:
        return TerminationKind.UNKNOWN

    tol = floor * _FLOOR_TOL_RATIO
    rho = float(y_final[rhoidx])
    avv = float(y_final[avvidx])
    all_val = float(y_final[allidx])

    at_rho = rho <= floor + tol
    at_avv = avv <= floor + tol
    at_all = all_val <= floor + tol

    if at_rho:
        return TerminationKind.RHO_FLOOR
    if at_avv:
        return TerminationKind.AVV_FLOOR
    if at_all:
        return TerminationKind.ALL_FLOOR
    return TerminationKind.UNKNOWN


def step_limit_hit(action) -> bool:
    """Return True if any RG integration on this action hit the RHS step budget."""
    if getattr(action, "step_limit_hit", False):
        return True
    if getattr(action, "step_limit_hit_thr", False):
        return True
    if getattr(action, "step_limit_hit_bec", False):
        return True
    return False


def classify_bec_termination(action, floor: float = COUPLING_FLOOR) -> TerminationKind:
    """Classify termination for a BECAction instance."""
    if step_limit_hit(action):
        return TerminationKind.FAILED
    return classify_bec_termination_from_sol(
        getattr(action, "sol", None),
        action.rhoidx,
        action.allidx,
        action.avvidx,
        floor,
    )


def classify_bcs_bec_termination(action, floor: float = COUPLING_FLOOR) -> TerminationKind:
    """Classify BEC-branch termination for a BCSAction instance."""
    if step_limit_hit(action):
        return TerminationKind.FAILED
    if not getattr(action, "becShift", False):
        sol_thr = getattr(action, "solThr", None)
        if sol_thr is None:
            return TerminationKind.FAILED
        status = int(sol_thr.status)
        if status == -1 or not np.all(np.isfinite(sol_thr.y[:, -1])):
            return TerminationKind.FAILED
        if status == 0:
            return TerminationKind.COMPLETED
        return TerminationKind.UNKNOWN

    return classify_bec_termination_from_sol(
        action.solBEC,
        action.becrhoidx,
        action.becallidx,
        action.becavvidx,
        floor,
    )


def termination_is_valid(kind: TerminationKind) -> bool:
    return kind in (TerminationKind.COMPLETED, TerminationKind.ALL_FLOOR)


def invalid_reason_for_kind(kind: TerminationKind) -> str | None:
    if termination_is_valid(kind):
        return None
    if kind == TerminationKind.RHO_FLOOR:
        return "rho_floor"
    if kind == TerminationKind.AVV_FLOOR:
        return "avv_floor"
    if kind == TerminationKind.FAILED:
        return "integrator_failed"
    if kind == TerminationKind.UNKNOWN:
        return "unknown_termination"
    return str(kind)


def is_bec_integration_valid(action, floor: float = COUPLING_FLOOR) -> tuple[bool, str | None]:
    """Return whether a BECAction integration is usable for mu finding."""
    if step_limit_hit(action):
        return False, "step_limit"
    kind = classify_bec_termination(action, floor)
    if termination_is_valid(kind):
        return True, None
    return False, invalid_reason_for_kind(kind)


def is_bcs_integration_valid(action, floor: float = COUPLING_FLOOR) -> tuple[bool, str | None]:
    """Return whether a BCSAction integration is usable for mu finding."""
    if step_limit_hit(action):
        return False, "step_limit"
    sol_thr = getattr(action, "solThr", None)
    if sol_thr is None:
        return False, "exception"

    if int(sol_thr.status) == -1 or not np.all(np.isfinite(sol_thr.y[:, -1])):
        return False, "integrator_failed"

    if getattr(action, "becShift", False):
        kind = classify_bcs_bec_termination(action, floor)
        if termination_is_valid(kind):
            return True, None
        return False, invalid_reason_for_kind(kind)

    if int(sol_thr.status) == 0:
        return True, None
    return False, "unknown_termination"
