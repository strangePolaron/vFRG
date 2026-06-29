import numpy as np
import pytest

import BCSna as bcs


BCS_DIRECT = {
    "eb0": 1.2,
    "beta": 200.0,
    "mu": -0.22,
    "cutoff": 3.0,
    "mf": 1.0,
    "becShift": True,
    "solThr_status": 1,
    "solBEC_status": 0,
    "FinalNum": 0.02970812011736881,
    "FinalRhoSF": 0.5101547003053655,
}

BCS_FINDMU = {
    "eb0": 8.0,
    "beta": 5000.0,
    "cutoff": 50.0,
    "mf": 1.0,
    "targetNum": 1.0 / np.pi,
    "mu": -1.184701025309193,
    "becShift": True,
    "solThr_status": 1,
    "solBEC_status": 0,
    "FinalNum": 0.31830976224529,
    "FinalRhoSF": 0.7647437881854254,
}

BCS_BEC_BRANCH = {
    "eb0": 2.0,
    "beta": 800.0,
    "mu": 0.1,
    "cutoff": 3.0,
    "mf": 1.0,
    "becShift": True,
    "solThr_status": 1,
    "solBEC_status": 0,
    "FinalNum": 0.10655505947234784,
    "FinalRhoSF": 0.7315075982586725,
}


def test_bcs_direct_mu():
    a = bcs.BCSAction(
        BCS_DIRECT["eb0"],
        BCS_DIRECT["beta"],
        BCS_DIRECT["mu"],
        BCS_DIRECT["cutoff"],
        BCS_DIRECT["mf"],
    )
    assert a.becShift is BCS_DIRECT["becShift"]
    assert int(a.solThr.status) == BCS_DIRECT["solThr_status"]
    assert a.FinalNum() == pytest.approx(BCS_DIRECT["FinalNum"], rel=1e-9, abs=1e-12)
    assert a.FinalRhoSF() == pytest.approx(BCS_DIRECT["FinalRhoSF"], rel=1e-9, abs=1e-12)
    assert int(a.solBEC.status) == BCS_DIRECT["solBEC_status"]


def test_bcs_find_mu():
    mu = bcs.findMu(
        BCS_FINDMU["targetNum"],
        BCS_FINDMU["eb0"],
        BCS_FINDMU["beta"],
        BCS_FINDMU["cutoff"],
        BCS_FINDMU["mf"],
    )
    assert mu == pytest.approx(BCS_FINDMU["mu"], rel=1e-6, abs=1e-8)
    a = bcs.BCSAction(
        BCS_FINDMU["eb0"],
        BCS_FINDMU["beta"],
        mu,
        BCS_FINDMU["cutoff"],
        BCS_FINDMU["mf"],
    )
    assert a.becShift is BCS_FINDMU["becShift"]
    assert int(a.solThr.status) == BCS_FINDMU["solThr_status"]
    assert a.FinalNum() == pytest.approx(BCS_FINDMU["FinalNum"], rel=1e-9, abs=1e-12)
    assert a.FinalRhoSF() == pytest.approx(BCS_FINDMU["FinalRhoSF"], rel=1e-9, abs=1e-12)
    assert int(a.solBEC.status) == BCS_FINDMU["solBEC_status"]


def test_bcs_bec_branch():
    p = BCS_BEC_BRANCH
    a = bcs.BCSAction(p["eb0"], p["beta"], p["mu"], p["cutoff"], p["mf"])
    assert a.becShift is True
    assert int(a.solThr.status) == p["solThr_status"]
    assert int(a.solBEC.status) == p["solBEC_status"]
    assert a.FinalNum() == pytest.approx(p["FinalNum"], rel=1e-9, abs=1e-12)
    assert a.FinalRhoSF() == pytest.approx(p["FinalRhoSF"], rel=1e-9, abs=1e-12)
