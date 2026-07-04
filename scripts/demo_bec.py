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

import sys


def main():
    ebBos = 3.0
    mu = 1.2-1e-2
    mass = 1.0
    beta = float(sys.argv[1])#0.242

    bec = BECAction(ebBos, beta, mu, mass)
    solt = bec.sol.t
    soly = bec.sol.y

    print(bec.ydata.keysUpd)
    print(bec.FinalNum())
    print(bec.FinalRhoSF())

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.figure()
    c1, = plt.plot(solt, soly[0, :]/soly[0,0])
    c2, = plt.plot(solt, soly[1, :]/soly[1,0])
    c3, = plt.plot(solt, soly[2, :])
    c4, = plt.plot(solt, soly[3, :])
    c5, = plt.plot(solt, soly[5, :])
    c6, = plt.plot(solt, soly[4, :])
    plt.xlabel("$\\log(\\Lambda / k)$")
    assert bec.ydata.keysUpd is not None
    #plt.legend([c1, c2, c3, c4, c5], ["$g/g_0$", "$\\rho_{0,k}$", "$A_{v,k}$", "$A_{l,k}$", "$y_k$"])
    plt.legend([c1, c2, c3, c4, c5, c6], ["$g/g_0$", "$\\rho_{0,k}/\\rho_{0,k=0}$", "$A_{v,k}$", "$A_{l,k}$", "$y_k\\xi^2$", "$k\\xi/(2\\pi)$"])
    plt.show()


if __name__ == "__main__":
    main()
