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
"""BCS-side Tc sweep and plotting — moved from plotTc.__main__."""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import tqdm

from plotTc import eb_row_tasks, kF, mf, ori_shape, ebgrid, betagrid, muiSF_eb_row


def main():
    falseSet = {'', 'False', '0', 'false', 'None', 'none'}
    if len(sys.argv)==1:
        recalc = True
    else:
        recalc = not (sys.argv[-1] in falseSet)
    if recalc:
        tasks = eb_row_tasks()
        with Pool(10) as p:
            rows = list(tqdm.tqdm(p.imap(muiSF_eb_row, tasks), total=len(tasks)))
        mugrid = np.array(rows).T.reshape(ori_shape)
        
        dat = {"mui": mugrid, "eb": ebgrid.reshape(ori_shape), "Tc": (1.0 / (betagrid.reshape(ori_shape)))}
        try:
            with open("Results/bcs-mu-effixed.pickle", "wb") as f:
                pickle.dump(dat, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            print("Error during pickling object (Possibly unsupported):", ex)

    with open("Results/bcs-mu-effixed.pickle", "rb") as f:
        dat = pickle.load(f)

    mugrid = dat["mui"]
    mugrid = np.nan_to_num(mugrid, nan=0.0, posinf=0.0, neginf=0.0)/ ((kF**2) / (2.0 * mf))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, ax = plt.subplots()
    ax.ticklabel_format(style="sci", scilimits=(-2, 2))

    ebm = np.min(-1.0 * np.log(ebgrid.reshape(ori_shape) * (mf / (kF**2))) / 2.0 - np.euler_gamma + np.log(2.0))
    ebM = np.max(-1.0 * np.log(ebgrid.reshape(ori_shape) * (mf / (kF**2))) / 2.0 - np.euler_gamma + np.log(2.0))
    ax.set_xticks(np.arange(ebm, ebM+1e-10, 0.5))

    murange = max(abs(np.min(mugrid)), abs(np.max(mugrid)))
    
    c = ax.pcolormesh(
        -1.0 * np.log(ebgrid.reshape(ori_shape) * (mf / (kF**2))) / 2.0 - np.euler_gamma + np.log(2.0),
        (2.0 * mf / (kF**2)) / (betagrid.reshape(ori_shape)),
        mugrid,
        shading="nearest",
        cmap="RdBu",
        vmin=-1.0*murange,
        vmax=murange,
    )
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.set_yticks([-0.5, 0., 0.5],["-0.5","0 or\nnonSF","$\\geq 0.5$"])
    ax.set_xlabel("$\\log(k_F a)$")
    ax.set_ylabel("$k_B T/E_F$")
    ax.set_title("$\\mu/E_F$")
    results_dir = Path("Results")
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_path = results_dir / "bcs-mu-effixed.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {fig_path}")
    # plt.show()


if __name__ == "__main__":
    main()
