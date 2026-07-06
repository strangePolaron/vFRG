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

from plotTc import mu_row_tasks, kF, mf, ntot_ori_shape, Ntotmugrid, Ntotbetagrid, ntot_row


def main():
    falseSet = {'', 'False', '0', 'false', 'None', 'none'}
    if len(sys.argv)==1:
        recalc = True
    else:
        recalc = not (sys.argv[-1] in falseSet)
    if recalc:
        tasks = mu_row_tasks()
        with Pool(10) as p:
            rows = list(tqdm.tqdm(p.imap(ntot_row, tasks), total=len(tasks)))
        ntotgrid = np.array(rows).T.reshape(ntot_ori_shape)
        
        dat = {"Ntot_i": ntotgrid, "mu": Ntotmugrid.reshape(ntot_ori_shape), "Tc": (1.0 / (Ntotbetagrid.reshape(ntot_ori_shape)))}
        try:
            with open("Results/bcs-ntot-effixed.pickle", "wb") as f:
                pickle.dump(dat, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            print("Error during pickling object (Possibly unsupported):", ex)

    with open("Results/bcs-ntot-effixed.pickle", "rb") as f:
        dat = pickle.load(f)

    ntotgrid = dat["Ntot_i"]
    ntotgrid = np.nan_to_num(ntotgrid, nan=0.0, posinf=0.0, neginf=0.0)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, ax = plt.subplots()
    ax.ticklabel_format(style="sci", scilimits=(-2, 2))

    mum = np.min(Ntotmugrid)
    muM = np.max(Ntotmugrid)
    ax.set_xticks(np.arange(mum, muM+1e-10, 0.5))

    

    c = ax.pcolormesh(
        Ntotmugrid,
        (mf / (kF**2)) / (Ntotbetagrid.reshape(ntot_ori_shape)),
        ntotgrid.reshape(ntot_ori_shape)/(kF**2),
        shading="nearest",
        cmap="RdBu",
        vmin=np.min(ntotgrid/(kF**2)),
        vmax=np.max(ntotgrid/(kF**2)),
    )
    cbar = fig.colorbar(c, ax=ax)
    #cbar.ax.set_yticks([-0.5, 0., 0.5],["$-$0.5","0 or\nnonSF","$\\geq$0.5"])
    ax.set_xlabel("$\\mu/\\epsilon_B$")
    ax.set_ylabel("$T/\\epsilon_B$")
    #ax.set_title("$\\mu/\\epsilon_B$")
    ax.set_title("$n_\\text{tot} / (m \\epsilon_B)$")
    results_dir = Path("Results")
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_path = results_dir / "bcs-ntot-effixed.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {fig_path}")
    # plt.show()


if __name__ == "__main__":
    main()
