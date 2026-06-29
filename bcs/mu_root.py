"""Root-finding helpers for chemical potential bisection."""

from __future__ import annotations

import scipy.optimize as optm


def bisect_with_guess(
    func,
    lo: float,
    hi: float,
    xtol: float,
    mu_guess: float | None = None,
    on_bracket_fail=None,
) -> float:
    """Bisect on [lo, hi], optionally narrowing the bracket around mu_guess."""
    lo_eval = func(lo)
    hi_eval = func(hi)

    if mu_guess is not None:
        mu_guess = min(max(float(mu_guess), lo), hi)
        mid_eval = func(mu_guess)
        if abs(mid_eval) <= xtol:
            return mu_guess
        if lo_eval * mid_eval <= 0.0:
            hi = mu_guess
            hi_eval = mid_eval
        elif mid_eval * hi_eval <= 0.0:
            lo = mu_guess
            lo_eval = mid_eval

    if lo_eval * hi_eval > 0.0:
        if on_bracket_fail is not None:
            on_bracket_fail(lo, hi, lo_eval, hi_eval)
        raise ValueError("Bisection bracket does not straddle a root")

    return optm.root_scalar(func, method="bisect", bracket=[lo, hi], xtol=xtol).root
