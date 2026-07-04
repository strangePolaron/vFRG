"""Tests for relaxed chemical-potential solvers."""

import numpy as np
import pytest

from bcs import bec_action, bcs_action
from bcs.mu_relax import (
    MuEval,
    MuRootError,
    evaluate_mu_bec,
    findMu_relax_bec,
    findMu_relax_bcs,
    relaxed_find_mu,
)
from bcs.ode_safe import TerminationKind
from tests.test_regression_bec import BEC_FINDMU
from tests.test_regression_bcs import BCS_FINDMU


def test_mu_relax_bec_matches_bisect():
    p = BEC_FINDMU
    mu_bisect = bec_action.findMu(p["targetNum"], p["eb"], p["beta"], p["mass"], use_hint_cache=False)
    mu_relax = findMu_relax_bec(p["targetNum"], p["eb"], p["beta"], p["mass"])
    assert mu_relax == pytest.approx(mu_bisect, rel=1e-4, abs=1e-5)


def test_mu_relax_bcs_matches_bisect():
    p = BCS_FINDMU
    target = 0.001
    mu_bisect = bcs_action.findMu(
        target, p["eb0"], p["beta"], p["cutoff"], p["mf"], use_hint_cache=False
    )
    mu_relax = findMu_relax_bcs(target, p["eb0"], p["beta"], p["cutoff"], p["mf"])
    assert mu_relax == pytest.approx(mu_bisect, rel=1e-3, abs=1e-3)


def test_mu_relax_rejects_rho_floor_for_high_mu():
    p = BEC_FINDMU
    hi = p["eb"] / 2.0 - 1e-5
    ev = evaluate_mu_bec(p["eb"], p["beta"], p["mass"], hi, p["targetNum"])
    if ev.termination is TerminationKind.RHO_FLOOR:
        assert ev.ok is False
        assert ev.reason == "rho_floor"

    mu_bisect = bec_action.findMu(p["targetNum"], p["eb"], p["beta"], p["mass"], use_hint_cache=False)
    mu_relax = findMu_relax_bec(
        p["targetNum"],
        p["eb"],
        p["beta"],
        p["mass"],
        mu_guess=hi,
    )
    assert mu_relax == pytest.approx(mu_bisect, rel=1e-4, abs=1e-5)
    ev = evaluate_mu_bec(p["eb"], p["beta"], p["mass"], mu_relax, p["targetNum"])
    assert ev.ok is True


def test_relaxed_find_mu_stops_at_max_evals():
    calls = {"n": 0}

    def eval_fn(mu: float) -> MuEval:
        calls["n"] += 1
        return MuEval(
            mu=mu,
            final_num=1.0,
            residual=1.0,
            ok=True,
            status=0,
            reason=None,
            termination=TerminationKind.COMPLETED,
        )

    with pytest.raises(MuRootError) as exc_info:
        relaxed_find_mu(eval_fn, 0.1, 1.0, 0.0, max_evals=5, max_secant_iter=20)

    assert calls["n"] <= 5
    assert exc_info.value.eval_limit == 5
    assert exc_info.value.eval_used == calls["n"]
