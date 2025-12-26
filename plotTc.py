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
import BCSna as bcs
#import BECna as bec
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import tqdm
import pickle

kF=1.
mf = 1.

def rhoSF(parpair, targetNum=(kF**2)/(2. * np.pi), cutoff=50., mass=mf):
    eb, beta = parpair
    #print(f"eb,\t{eb:.2f},\tbeta,\t{beta:.2f}")
    mu = bcs.findMu(targetNum, eb, beta, cutoff, mass)
    bcsobj = bcs.BCSAction(eb, beta, mu, cutoff, mass)
    if np.abs(np.sqrt(bcsobj.FinalNum() * 2. * np.pi) - (kF))>0.05:
        print(f"eb,\t{eb:.1f},\tbeta,\t{beta:.1f},\tmu\t{mu:.2f},\tkF,\t{np.sqrt(bcsobj.FinalNum() * 2. * np.pi):.2f}")
        return 0.
    return bcsobj.FinalRhoSF()

def muiSF(parpair, targetNum=(kF**2)/(2. * np.pi), cutoff=20., mass=mf):
    eb, beta = parpair
    #print(f"eb,\t{eb:.2f},\tbeta,\t{beta:.2f}")
    mu = bcs.findMu(targetNum, eb, beta, cutoff, mass)
    bcsobj = bcs.BCSAction(eb, beta, mu, cutoff, mass)
    if True:  #np.abs(np.sqrt(bcsobj.FinalNum() * 2. * np.pi) - 1.)>0.1:
        print(f"eb,\t{eb:.1f},\tbeta,\t{beta:.1f},\tmu\t{mu:.2f},\tkF,\t{np.sqrt(bcsobj.FinalNum() * 2. * np.pi):.2f}\t{bcsobj.becShift}\t{bcsobj.solBEC.status}")
    return mu   #bcsobj.FinalRhoSF()


def becbcsSeparate(parpair, targetNum=(kF**2)/(2. * np.pi), cutoff=20., mass=mf):
    eb, beta = parpair
    #print(f"eb,\t{eb:.2f},\tbeta,\t{beta:.2f}")
    mu = bcs.findMu(targetNum, eb, beta, cutoff, mass)
    bcsobj = bcs.BCSAction(eb, beta, mu, cutoff, mass)
    if np.abs(np.sqrt(bcsobj.FinalNum() * 2. * np.pi) - (kF))>0.05:
        #print(f"eb,\t{eb:.1f},\tbeta,\t{beta:.1f},\tmu\t{mu:.2f},\tkF,\t{np.sqrt(bcsobj.FinalNum() * 2. * np.pi):.2f}")
        rhosfFrac = 0.
    else:
        rhosfFrac = bcsobj.FinalRhoSF()
    #if np.abs(np.sqrt(bcsobj.FinalNum() * 2. * np.pi) - 1.)>0.1:
    #    print(f"eb,\t{eb:.1f},\tbeta,\t{beta:.1f},\tmu\t{mu:.2f},\tkF,\t{np.sqrt(bcsobj.FinalNum() * 2. * np.pi):.2f}")
    if rhosfFrac>0:
        if mu>0:
            return 1.
        else:
            return -1.
    return 0.  #bcsobj.FinalRhoSF()


#rhoSF((2.75, 800))
#"""
eblst = np.arange(0.1, 10., 0.2) * ((kF**2)/(2.*mf))
#betalst = 1. / np.arange(1./10000., 1./400., 1./10000.)
betalst = 1. / np.arange(1./10000., 20./100., 2./1000.) / ((kF**2)/(2.*mf))
betaMulst = 1. / np.arange(1./200000., 5./10000., 2./10000.)


ebgrid, betagrid = np.meshgrid(eblst, betalst)
ori_shape = ebgrid.shape
totLen = len(eblst) * len(betalst)
parLst = list(zip(ebgrid.reshape(totLen), betagrid.reshape(totLen)))
print(ori_shape)

#"""
ebMugrid, betaMugrid = np.meshgrid(eblst, betaMulst)
oriMu_shape = ebMugrid.shape
totMuLen = len(eblst) * len(betaMulst)
parMuLst = list(zip(ebMugrid.reshape(totMuLen), betaMugrid.reshape(totMuLen)))
print(oriMu_shape)
#"""



if __name__=="__main__":
    with Pool(12) as p:
        rhoi = list(tqdm.tqdm(p.imap_unordered(rhoSF, parLst), total=totLen))
    rhogrid = np.array(rhoi).reshape(ori_shape)

    dat = {"rhosf":rhogrid, "eb":ebgrid.reshape(ori_shape), "Tc":(1./(betagrid.reshape(ori_shape)))}
    try:
        with open("Results/bcs-effixed-KToff.pickle", "wb") as f:
            pickle.dump(dat, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)
    """
    with open("Results/bcs-effixed-KToff.pickle", "rb") as f:
        dat = pickle.load(f)
    """
    rhogrid = dat["rhosf"]
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    plt.rcParams['mathtext.fontset'] = "cm"
    fig, ax = plt.subplots()
    ax.ticklabel_format(style="sci",scilimits=(-2,2))
    c = ax.pcolormesh(ebgrid.reshape(ori_shape)*(2.*mf/(kF**2)), ((2.*mf/(kF**2))/(betagrid.reshape(ori_shape))), rhogrid, shading="nearest", cmap="RdBu", vmin=np.min(rhogrid), vmax=np.max(rhogrid))
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("$\\epsilon_B/E_F$")
    ax.set_ylabel("$k_B T/E_F$")
    ax.set_title("$A_{l,k}=\\rho_s/\\rho_{0,k}$")
    plt.show()
    #"""


    """
    with Pool(12) as p:
        mui = list(tqdm.tqdm(p.imap_unordered(muiSF, parMuLst), total=totMuLen))
    mugrid = np.array(mui).reshape(oriMu_shape)

    dat = {"mu":mugrid, "eb":ebMugrid.reshape(oriMu_shape), "Tc":(1./(betaMugrid.reshape(oriMu_shape)))}
    try:
        with open("Results/bcs-effixed-mu.pickle", "wb") as f:
            pickle.dump(dat, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    plt.rcParams['mathtext.fontset'] = "cm"

    plt.figure()
    crvLst = list()
    for idx in range(len(betaMulst)):
        ci, = plt.plot(ebMugrid.reshape(oriMu_shape)[idx, 5:-2], mugrid[idx, 5:-2])
        crvLst.append(ci)
    plt.legend(crvLst, [f"$T_c m_f/ k_n^2$={1./bi:.2e}" for bi in betaMulst])
    plt.xlabel("$\\epsilon_B m_f/k_n^2$")
    plt.ylabel("$\\mu m_f/k_n^2$")
    plt.show()
    """


    """
    with Pool(12) as p:
        becbcsi = list(tqdm.tqdm(p.imap_unordered(becbcsSeparate, parLst), total=totLen))
    becbcsgrid = np.array(becbcsi).reshape(ori_shape)

    #isbcsMat = (rhogrid==1.)
    #isbecMat = (rhogrid==2.)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    plt.rcParams['mathtext.fontset'] = "cm"
    fig, ax = plt.subplots()
    ax.ticklabel_format(style="sci",scilimits=(-2,2))
    ax.pcolormesh(ebgrid.reshape(ori_shape)*(2.*mf/(kF**2)), ((2.*mf/(kF**2))/(betagrid.reshape(ori_shape))), becbcsgrid, shading="nearest", cmap="RdBu", vmin=np.min(becbcsgrid), vmax=np.max(becbcsgrid))
    ax.set_xticks(np.arange(0., 10.1, 2.5))
    #fig.colorbar(c, ax=ax)
    ax.set_xlabel("$\\epsilon_B/E_F$")
    ax.set_ylabel("$k_B T/E_F$")
    #ax.set_title("$A_{l,k}=\\rho_s/\\rho_{0,k}$")
    plt.show()
    """
