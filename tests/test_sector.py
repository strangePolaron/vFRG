"""Tests for RGSector framework."""

import numpy as np
import pytest

from bcs.keys import Key
from bcs.sector import CouplingSpec, RGSector, compose_sectors
from bcs.state import RGState


class _DummySector(RGSector):
    def __init__(self, state: RGState, name: str, initial: float):
        super().__init__(state, (CouplingSpec(name, initial),))
        self._scale = 1.0

    def contribute(self, l: float, dy: RGState) -> None:
        dy.data[self.coupling_names[0]] = self._scale * l

    def contribute_post(self, l: float, dy: RGState) -> RGState | None:
        return super().contribute_post(l, dy)


def test_coupling_spec_registers_data_and_keys_upd():
    state = RGState()
    _DummySector(state, "zeta", 2.5)
    assert state.value("zeta") == 2.5
    assert state.keysUpd == ["zeta"]


def test_coupling_spec_key_must_match_name():
    with pytest.raises(ValueError, match="must match"):
        CouplingSpec("wrong", 1.0, Key.G)


def test_compose_sectors_merges_by_key():
    state = RGState()
    left = _DummySector(state, "a", 0.0)
    right = _DummySector(state, "b", 0.0)
    right._scale = 2.0
    out = compose_sectors(state, 3.0, [left, right])
    assert out[0] == pytest.approx(3.0)
    assert out[1] == pytest.approx(6.0)


def test_compose_sectors_applies_hooks():
    state = RGState()
    sector = _DummySector(state, "a", 0.0)

    def double_hook(_st, dy):
        dy.data["a"] *= 2.0

    out = compose_sectors(state, 2.0, [sector], hooks=[double_hook])
    assert out[0] == pytest.approx(4.0)


def test_compose_sectors_uses_strict_add_by_key():
    """Merged buffers share state keys; mismatched manual buffers still raise."""
    a = RGState({"x": 1.0}, keys_upd=["x"])
    b = RGState({"y": 1.0}, keys_upd=["y"])
    with pytest.raises(KeyError, match="Key mismatch"):
        a.add_by_key(b)


def test_add_coupling_via_coupling_spec_only():
    """Adding a dummy coupling requires only CouplingSpec + contribute."""
    state = RGState()

    class ZetaSector(RGSector):
        def __init__(self, st):
            super().__init__(st, (CouplingSpec("zeta", 1.0),))

        def contribute(self, l, dy):
            dy.data["zeta"] = l

        def contribute_post(self, l: float, dy: RGState) -> RGState | None:
            return super().contribute_post(l, dy)

    sector = ZetaSector(state)
    dy = state.zero_like()
    sector.contribute(5.0, dy)
    assert dy.value("zeta") == pytest.approx(5.0)
    assert state.value("zeta") == pytest.approx(1.0)
