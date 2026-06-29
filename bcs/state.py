"""Dynamic RG state container with key-based merge."""

from typing import Dict, Union
import numpy as np

from bcs.keys import Key


class RGState:
    """Keyed RG couplings with optional ODE vector ordering."""

    def __init__(self, data: dict[str, float] | None = None, keys_upd: list[str] | None = None, *, keysUpd=None):
        if keysUpd is not None:
            keys_upd = keysUpd
        self._data: dict[str, float] = dict(data or {})
        self._keys_upd: list[str] | None = list(keys_upd) if keys_upd is not None else None

    @property
    def data(self) -> dict[str, float]:
        return self._data

    @data.setter
    def data(self, value: dict[str, float]) -> None:
        self._data = value

    @property
    def keysUpd(self) -> list[str] | None:
        return self._keys_upd

    @keysUpd.setter
    def keysUpd(self, value: list[str] | None) -> None:
        self._keys_upd = list(value) if value is not None else None

    def _key_name(self, key: Union[Key, str]) -> str:
        return key.value if isinstance(key, Key) else key

    def value(self, key: Union[Key, str]) -> float:
        return self._data[self._key_name(key)]

    def register(self, values: dict[str, float], new_keys: list[str] | None = None) -> None:
        self._data.update(values)
        if new_keys is not None:
            self.keysUpdAppend(new_keys)

    def keysUpdFunc(self, keys_upd: list[str]) -> None:
        self._keys_upd = list(keys_upd)

    def keysUpdAppend(self, keys_new: list[str]) -> None:
        if self._keys_upd is None:
            self._keys_upd = keys_new.copy()
        else:
            for keyi in keys_new:
                if keyi not in self._keys_upd:
                    self._keys_upd.append(keyi)

    def ylst(self) -> np.ndarray:
        return self.to_array()

    def to_array(self) -> np.ndarray:
        if self._keys_upd is None:
            return np.array(list(self._data.values()), dtype=np.double)
        return np.array([self._data[k] for k in self._keys_upd], dtype=np.double)

    def update(self, arr) -> None:
        self.from_array(arr)

    def from_array(self, arr) -> None:
        if self._keys_upd is None:
            for idx, k in enumerate(self._data.keys()):
                self._data[k] = arr[idx]
        else:
            for idx, k in enumerate(self._keys_upd):
                self._data[k] = arr[idx]

    def additem(self, kargs: str, values: float) -> None:
        self._data[kargs] = values

    def dataCpy(self, data_dict: Dict) -> None:
        self._data = data_dict.copy()

    def dataAppend(self, data_dict: Dict, keys_new=None) -> None:
        self.register(data_dict, keys_new)

    def dataConcat(self, other: "RGState") -> None:
        self.dataAppend(other._data, other._keys_upd)

    def subylst_keys(self, key_lst: list[str]) -> np.ndarray:
        return np.array([self._data[k] for k in key_lst], dtype=np.double)

    def copy(self) -> "RGState":
        tmp = RGState()
        tmp._keys_upd = None if self._keys_upd is None else self._keys_upd.copy()
        tmp._data = self._data.copy()
        return tmp

    def zeroVecGen(self) -> "RGState":
        return self.zero_like()

    def zero_like(self) -> "RGState":
        tmp = RGState()
        tmp._keys_upd = self._keys_upd
        tmp._data = {k: 0.0 for k in self._data}
        return tmp

    def add_by_key(self, other: "RGState", *, strict: bool = True) -> "RGState":
        if strict and set(self._data) != set(other._data):
            raise KeyError(f"Key mismatch: {set(self._data)} vs {set(other._data)}")
        result = self.zero_like()
        for k in self._data:
            if k in other._data:
                result._data[k] = self._data[k] + other._data[k]
            else:
                result._data[k] = self._data[k]
        return result

    def sum_other(self, other: "RGState") -> "RGState":
        merged = self.add_by_key(other, strict=True)
        self._data.update(merged._data)
        return self


parseData = RGState
