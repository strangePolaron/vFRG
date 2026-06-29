#!/usr/bin/python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# ///
"""Demo: BECAction RG curves — moved from BECna.__main__."""

import matplotlib.pyplot as plt

from BECna import BECAction


def main():
    ebBos = 3.0
    mu = 0.3
    mass = 1.0
    beta = 1000.0

    bec = BECAction(ebBos, beta, mu, mass)
    solt = bec.sol.t
    soly = bec.sol.y
    print(bec.FinalRhoSF())

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.figure()
    c2, = plt.plot(solt, soly[1, :])
    c3, = plt.plot(solt, soly[2, :])
    c4, = plt.plot(solt, soly[3, :])
    c5, = plt.plot(solt, soly[4, :])
    plt.xlabel("$\\log(\\Lambda / k)$")
    assert bec.ydata.keysUpd is not None
    plt.legend([c2, c3, c4, c5], ["$\\rho_{0,k}$", "$A_{v,k}$", "$A_{l,k}$", "$y_k$"])
    plt.show()


if __name__ == "__main__":
    main()
