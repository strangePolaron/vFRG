import numpy as np
import pytest

import BECna as bec


BEC_DIRECT = {
    "eb": 3.0,
    "beta": 1000.0,
    "mu": 0.3,
    "mass": 1.0,
    "sol_status": 0,
    "FinalNum": 0.007634233785371707,
    "FinalRhoSF": 0.5786279878375121,
}

BEC_FINDMU = {
    "eb": 1.0,
    "beta": 500.0,
    "mass": 1.0,
    "targetNum": 1.0 / (4.0 * np.pi),
    "mu": 1.0572961387634277,
    "sol_status": 0,
    "FinalNum": 0.07957851638824644,
    "FinalRhoSF": 0.6556034723924156,
}


def test_bec_direct_mu():
    p = BEC_DIRECT
    a = bec.BECAction(p["eb"], p["beta"], p["mu"], p["mass"])
    assert int(a.sol.status) == p["sol_status"]
    assert a.FinalNum() == pytest.approx(p["FinalNum"], rel=1e-9, abs=1e-12)
    assert a.FinalRhoSF() == pytest.approx(p["FinalRhoSF"], rel=1e-9, abs=1e-12)


def test_bec_find_mu():
    p = BEC_FINDMU
    mu = bec.findMu(p["targetNum"], p["eb"], p["beta"], p["mass"])
    assert mu == pytest.approx(p["mu"], rel=1e-5, abs=1e-7)
    a = bec.BECAction(p["eb"], p["beta"], mu, p["mass"])
    assert int(a.sol.status) == p["sol_status"]
    assert a.FinalNum() == pytest.approx(p["FinalNum"], rel=1e-9, abs=1e-12)
    assert a.FinalRhoSF() == pytest.approx(p["FinalRhoSF"], rel=1e-9, abs=1e-12)
