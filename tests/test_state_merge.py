"""Tests for RGState key-based merge (add_by_key)."""

import pytest

import parsey as prs
from bcs.state import RGState


def test_add_by_key_ignores_keys_upd_order():
    a = RGState({"g": 1.0, "eb": 2.0}, keysUpd=["g", "eb"])
    b = RGState({"g": 3.0, "eb": 4.0}, keysUpd=["eb", "g"])
    c = a.add_by_key(b)
    assert c.value("g") == 4.0
    assert c.value("eb") == 6.0
    assert c.keysUpd == ["g", "eb"]


def test_add_by_key_preserves_export_order():
    a = RGState({"g": 1.0, "eb": 2.0}, keysUpd=["g", "eb"])
    b = RGState({"g": 3.0, "eb": 4.0}, keysUpd=["eb", "g"])
    c = a.add_by_key(b)
    assert list(c.ylst()) == [4.0, 6.0]


def test_add_by_key_strict_rejects_missing_keys():
    a = RGState({"g": 1.0, "eb": 2.0})
    b = RGState({"g": 3.0})
    with pytest.raises(KeyError):
        a.add_by_key(b, strict=True)


def test_sum_other_shim_matches_add_by_key():
    a = prs.parseData({"g": 1.0, "eb": 2.0}, keysUpd=["g", "eb"])
    b = prs.parseData({"g": 3.0, "eb": 4.0}, keysUpd=["eb", "g"])
    a.sum_other(b)
    assert a.value("g") == 4.0
    assert a.value("eb") == 6.0


def test_sum_other_strict_rejects_mismatched_keys():
    a = prs.parseData({"g": 1.0, "eb": 2.0})
    b = prs.parseData({"g": 3.0})
    with pytest.raises(KeyError):
        a.sum_other(b)
