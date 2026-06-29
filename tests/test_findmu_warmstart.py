"""findMu warm-start returns same roots as direct bisection."""

import numpy as np
import pytest

from bcs import bcs_action, bec_action


def test_bcs_findmu_mu_guess_matches_baseline():
    target = 1.0 / np.pi
    eb, beta, cutoff, mass = 8.0, 5000.0, 50.0, 1.0
    bcs_action._bcs_mu_hint.clear()
    mu0 = bcs_action.findMu(target, eb, beta, cutoff, mass, use_hint_cache=False)
    bcs_action._bcs_mu_hint.clear()
    mu1 = bcs_action.findMu(target, eb, beta, cutoff, mass, mu_guess=mu0 * 0.95, use_hint_cache=False)
    assert mu1 == pytest.approx(mu0, rel=1e-6, abs=1e-8)


def test_bec_findmu_mu_guess_matches_baseline():
    target = 1.0 / (4.0 * np.pi)
    eb, beta, mass = 1.0, 500.0, 1.0
    bec_action._bec_mu_hint.clear()
    mu0 = bec_action.findMu(target, eb, beta, mass, use_hint_cache=False)
    bec_action._bec_mu_hint.clear()
    mu1 = bec_action.findMu(target, eb, beta, mass, mu_guess=mu0 * 0.999, use_hint_cache=False)
    assert mu1 == pytest.approx(mu0, rel=1e-4, abs=1e-5)
