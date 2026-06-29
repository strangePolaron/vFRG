#!/usr/bin/python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# ///
"""Demo: standalone KT flow — moved from RigidBallRG.__main__."""

import numpy as np
import scipy.integrate as itg
from matplotlib import pyplot as plt

import parsey as prs
from RigidBallRG import KT, cutoff_convert_2_rspc


def main():
    cutoffLambda = 12.0
    m = np.double(1.0)
    rho = np.double(12.0)
    nMax = 1
    prsdat = prs.parseData()
    kt_obj = KT(prsdat, m / rho, nMax)
    y0 = kt_obj.yGen()
    sol = itg.solve_ivp(
        kt_obj.eqRHS_onlyKT,
        (np.double(0.0), np.double(10.0)),
        y0,
        method="LSODA",
        rtol=1e-6,
        atol=1e-8,
        min_step=1e-12,
    )
    plt.figure()
    plt.plot(1.0 / sol.y[0, :], sol.y[1, :])
    plt.show()


if __name__ == "__main__":
    main()
