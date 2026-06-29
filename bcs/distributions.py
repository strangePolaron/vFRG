"""Thermal distribution functions for fermions and bosons (Matsubara)."""

import numpy as np


def nF(z, beta):
    rl: np.double = np.real(z)
    exprl: np.double = beta * rl
    if exprl < -30.0:
        return np.double(1.0)
    if exprl > 30.0:
        return np.double(0.0)
    im: np.double = np.imag(z)
    phs_log: np.double = beta * im
    phs: np.complex128 = np.cos(phs_log) + 1.0j * np.sin(phs_log)
    return np.real(1.0 / (1.0 + phs * np.exp(exprl)))


def nB(z, beta):
    rl: np.double = np.real(z)
    exprl: np.double = beta * rl
    if exprl < -30.0:
        return np.double(1.0)
    if exprl > 30.0:
        return np.double(0.0)
    im: np.double = np.imag(z)
    phs_log: np.double = beta * im
    phs: np.complex128 = np.cos(phs_log) + 1.0j * np.sin(phs_log)
    return np.real(1.0 / (1.0 - phs * np.exp(exprl)))
