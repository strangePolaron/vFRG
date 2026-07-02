"""RG coupling key names."""

from enum import StrEnum


class Key(StrEnum):
    EB = "eb"
    EF = "ef"
    G = "g"
    H = "h"
    DFAC = "dfac"
    RHO_F = "rhoF"
    NTHRM = "nthrm"
    RHO = "rho"
    AVV = "avv"
    ALL = "all"
    LUTK = "lutK"
    IOTA = "iota"


def key_index(keys_upd: list[str], key: Key | str) -> int:
    name = key.value if isinstance(key, Key) else key
    return keys_upd.index(name)
