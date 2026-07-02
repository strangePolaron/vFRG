"""Post-merge hooks for BCS orchestrator sector composition."""

from bcs import fermion
from bcs.keys import Key
from bcs.state import RGState


def make_h_renorm_hook_thr(h0: float):
    """h-renormalization corrections for thermal phase (was inline in thrEqn)."""

    def hook(state: RGState, dy: RGState) -> None:
        dZ = fermion.dh2dZ(dy.value(Key.H), state.value(Key.H))
        dy.data["eb"] += -1.0 * state.value(Key.EB) * dZ
        dy.data["g"] += -2.0 * state.value(Key.G) * dZ
        #dy.data["nthrm"] = dy.data["nthrm"] * pow(state.value(Key.H) / h0, 2)

    return hook


def bec_clamp_hook(state: RGState, _dy: RGState) -> None:
    """Floor rho, avv, all before BEC-phase sector merge."""
    state.data["rho"] = max(state.data["rho"], 1e-5)
    state.data["avv"] = max(state.data["avv"], 1e-5)
    state.data["all"] = max(state.data["all"], 1e-5)


def h_renorm_hook_bec(state: RGState, dy: RGState) -> None:
    """h-renormalization and BCS→BEC condensate corrections (was inline in spfEqn)."""
    dZ = fermion.dh2dZ(dy.value(Key.H), state.value(Key.H))
    dy.data["g"] += -2.0 * state.value(Key.G) * dZ
    dy.data["rho"] += state.value(Key.RHO) * dZ + (-1.0 * dy.value(Key.EB) / state.value(Key.G))
    dy.data["eb"] = 0.0
    dy.data["all"] += state.value(Key.ALL) * dZ
    dy.data["avv"] += state.value(Key.AVV) * dZ 
    dy.data["nthrm"] = 0.0

def kt_hook_bec(state: RGState, dy: RGState) -> None:
    """diota"""
    dy.data["iota"] = -1.0 * state.value(Key.IOTA) * (1.0 + (1.0/2.0)*(dy.data["avv"]/state.value(Key.AVV) + dy.data["g"]/state.value(Key.G) + dy.data["rho"]/state.value(Key.RHO)))

