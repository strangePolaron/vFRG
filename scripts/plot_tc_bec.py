#!/usr/bin/python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy",
#     "tqdm",
# ]
# ///
"""BEC-side Tc sweep and plotting — moved from plotTcBEC.__main__."""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import tqdm

from plotTcBEC import eb_row_tasks, ebgrid, betagrid, ori_shape, rhoSF_eb_row


def main():
    tasks = eb_row_tasks()
    with Pool(10) as p:
        rows = list(tqdm.tqdm(p.imap(rhoSF_eb_row, tasks), total=len(tasks)))
    rhogrid = np.array(rows).T.reshape(ori_shape)

    dat = {"rhosf": rhogrid, "eb": ebgrid.reshape(ori_shape), "Tc": (1.0 / (betagrid.reshape(ori_shape)))}
    try:
        with open("Results/bec.pickle", "wb") as f:
            pickle.dump(dat, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

    with open("Results/bec.pickle", "rb") as f:
        dat = pickle.load(f)
    rhogrid = dat["rhosf"]
    rhogrid = np.nan_to_num(rhogrid, nan=0.0, posinf=0.0, neginf=0.0)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, ax = plt.subplots()
    ax.ticklabel_format(style="sci", scilimits=(-2, 2))
    c = ax.pcolormesh(
        -1.0 * np.log(ebgrid.reshape(ori_shape)) / 2.0 - np.euler_gamma + np.log(2.0),
        1.0 / (betagrid.reshape(ori_shape)),
        rhogrid,
        shading="nearest",
        cmap="RdBu",
        vmin=np.min(rhogrid),
        vmax=np.max(rhogrid),
    )
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("$\\log(k_n a)$")
    ax.set_ylabel("$k_B T m/k_n^2$")
    ax.set_title("$A_{l,k}=\\rho_s/\\rho_{0,k}$")
    results_dir = Path("Results")
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_path = results_dir / "bec.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {fig_path}")
    # plt.show()


if __name__ == "__main__":
    main()
