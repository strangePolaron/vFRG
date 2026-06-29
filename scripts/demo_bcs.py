#!/usr/bin/python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# ///
"""Demo: BCSAction density map over (mu, T) — moved from BCSna.__main__."""

import matplotlib.pyplot as plt
import numpy as np

from BCSna import BCSAction


def main():
    kF = 2.0
    mass = 1.0
    ef = (kF**2) / (2.0 * mass)
    eb = 1.2
    cutoff = 3.0
    beta = 200.0

    mulst = np.arange(-0.35, 0.6, 0.006) * ef
    betalst = 1.0 / np.arange(0.1 / beta, 10.0 / beta, 0.1 / beta)

    mu2D, beta2D = np.meshgrid(mulst, betalst)
    nTot = np.zeros(mu2D.shape)
    lx, ly = mu2D.shape
    for idx in range(lx):
        for jdx in range(ly):
            nTot[idx, jdx] = BCSAction(eb, beta2D[idx, jdx], mu2D[idx, jdx], cutoff, mass).FinalNum()

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    plt.rcParams["mathtext.fontset"] = "cm"

    fig, ax = plt.subplots()
    ax.ticklabel_format(style="sci", scilimits=(-2, 2))
    c = ax.pcolormesh(
        mu2D / eb,
        1.0 / (beta2D * eb),
        nTot / eb / mass,
        shading="nearest",
        cmap="RdBu",
        vmin=np.min(nTot),
        vmax=np.max(nTot) / 6.0,
    )
    ax.set_xticks(np.arange(-0.5, 1.01, 0.5))
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("$\\mu_F/\\epsilon_B$")
    ax.set_ylabel("$k_B T /\\epsilon_B$")
    ax.set_title("$n_{tot}/(m_F\\epsilon_B)$")
    plt.show()


if __name__ == "__main__":
    main()
