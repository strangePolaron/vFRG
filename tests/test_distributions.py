import numpy as np
import pytest

from bcs.distributions import nB, nF


def test_nF_midrange():
    result = float(nF(1.0 + 0.0j, 100.0))
    assert 0.0 <= result <= 1.0


def test_nF_low_temperature_limit():
    assert float(nF(5.0 + 0.0j, 100.0)) == pytest.approx(0.0, abs=1e-10)


def test_nF_high_temperature_limit():
    assert float(nF(-5.0 + 0.0j, 100.0)) == pytest.approx(1.0, abs=1e-10)


def test_nB_low_temperature_limit():
    assert float(nB(5.0 + 0.0j, 100.0)) == pytest.approx(0.0, abs=1e-10)


def test_nB_high_temperature_limit():
    assert float(nB(-5.0 + 0.0j, 100.0)) == pytest.approx(1.0, abs=1e-10)


def test_nB_midrange_complex():
    z = 0.5 + 0.2j
    result = float(nB(z, 10.0))
    assert 0.0 <= result <= 1.0
    assert np.isfinite(result)
