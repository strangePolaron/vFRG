"""RG sector framework: declarative couplings and composed flow contributions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from bcs.keys import Key
from bcs.state import RGState

MergeHook = Callable[[RGState, RGState], None]


@dataclass(frozen=True)
class CouplingSpec:
    """Declares one RG coupling: string name, initial value, optional Key enum link."""

    name: str
    initial: float
    key: Key | None = None

    def __post_init__(self) -> None:
        if self.key is not None and self.name != self.key.value:
            raise ValueError(f"CouplingSpec name {self.name!r} must match key {self.key.value!r}")


class RGSector(ABC):
    """Physics subsystem that registers couplings and contributes ∂coupling/∂l."""

    def __init__(
        self,
        state: RGState,
        couplings: tuple[CouplingSpec, ...],
        *,
        append_keys: bool = True,
    ) -> None:
        self.state = state
        self.ydata = state
        self._couplings = couplings
        self._register(append_keys)

    def _register(self, append_keys: bool) -> None:
        values = {c.name: c.initial for c in self._couplings}
        key_names = [c.name for c in self._couplings]
        self.state.register(values, key_names if append_keys else None)

    @property
    def coupling_names(self) -> list[str]:
        return [c.name for c in self._couplings]

    @abstractmethod
    def contribute(self, l: float, dy: RGState) -> None:
        """Write this sector's derivatives into dy (zeroed buffer)."""

    @abstractmethod
    def contribute_post(self, l: float, dy: RGState) -> RGState | None:
        """Write this sector's derivatives into dy (zeroed buffer)."""
        """This function is designed for the case you need information of previous dy calculation"""


    def dylst(self, l: float, dy: RGState) -> None:
        self.contribute(l, dy)


def compose_sectors(
    state: RGState,
    l: float,
    sectors: Sequence[RGSector],
    hooks: Sequence[MergeHook] = (),
) -> np.ndarray:
    """Merge sector contributions by key, apply hooks, return ODE derivative vector."""
    dy_accum: RGState | None = None
    for sector in sectors:
        dy_sector = state.zero_like()
        sector.contribute(l, dy_sector)
        dy_accum = dy_sector if dy_accum is None else dy_accum.add_by_key(dy_sector)
    if dy_accum is None:
        raise ValueError("compose_sectors requires at least one sector")
    #print(f"1:{dy_accum.ylst()}")
    for hook in hooks:
        hook(state, dy_accum)
    #print(f"2:{dy_accum.ylst()}")
    for sector in sectors:
        #dy_sector = state.zero_like()
        dy_sector = sector.contribute_post(l, dy_accum)
        if dy_sector is not None:
            dy_accum = dy_sector if dy_accum is None else dy_accum.add_by_key(dy_sector)
    #print(f"3:{dy_accum.ylst()}")
    return dy_accum.ylst()
