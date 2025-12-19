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
import BECna as bec
#import BECna as bec
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import tqdm
import pickle

def rhoSF(parpair, targetNum=1./(4. * np.pi), mass=1.):
    eb, beta = parpair
    #print(f"eb,\t{eb:.2f},\tbeta,\t{beta:.2f}")
    mu = bec.findMu(targetNum, eb, beta, mass)
    becobj = bec.BECAction(eb, beta, mu, mass)
    #print(f"eb,\t{eb:.1f},\tbeta,\t{beta:.1f},\tmu\t{mu:.2f},\tkn,\t{np.sqrt(becobj.FinalNum() * 2. * np.pi):.2f}")
    return becobj.FinalRhoSF()

def muBEC(parpair, targetNum=1./(4. * np.pi), mass=1.):
    eb, beta = parpair
    #print(f"eb,\t{eb:.2f},\tbeta,\t{beta:.2f}")
    mu = bec.findMu(targetNum, eb, beta, mass)
    #becobj = bec.BECAction(eb, beta, mu, mass)
    #print(f"eb,\t{eb:.1f},\tbeta,\t{beta:.1f},\tmu\t{mu:.2f},\tkn,\t{np.sqrt(becobj.FinalNum() * 2. * np.pi):.2f}")
    return mu   #becobj.FinalRhoSF()


#rhoSF((2.75, 800))
#"""
eblst = np.arange(0.01, 2.5, 0.01)
betalst = 1. / np.arange(1./10000., 7.5/100., 5./10000.)
betaMulst = 1. / np.arange(1./10000., 1./40., 1./200.)

ebgrid, betagrid = np.meshgrid(eblst, betalst)
ori_shape = ebgrid.shape
totLen = len(eblst) * len(betalst)
parLst = list(zip(ebgrid.reshape(totLen), betagrid.reshape(totLen)))

ebMugrid, betaMugrid = np.meshgrid(eblst, betaMulst)
ori_shape_mu = ebMugrid.shape
totMuLen = len(eblst) * len(betaMulst)
parMuLst = list(zip(ebMugrid.reshape(totMuLen), betaMugrid.reshape(totMuLen)))
print(ori_shape_mu)

if __name__=="__main__":
    #"""
    with Pool(12) as p:
        rhoi = list(tqdm.tqdm(p.imap_unordered(rhoSF, parLst), total=totLen))
    rhogrid = np.array(rhoi).reshape(ori_shape)

    dat = {"rhosf":rhogrid, "eb":ebgrid.reshape(ori_shape), "Tc":(1./(betagrid.reshape(ori_shape)))}
    try:
        with open("Results/bec.pickle", "wb") as f:
            pickle.dump(dat, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)
    """    
    
    with open("Results/bec.pickle", "rb") as f:
        dat = pickle.load(f)
    rhogrid = dat["rhosf"]
    """

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    plt.rcParams['mathtext.fontset'] = "cm"
    fig, ax = plt.subplots()

    ax.ticklabel_format(style="sci",scilimits=(-2,2))
    c = ax.pcolormesh(ebgrid.reshape(ori_shape), (1./(betagrid.reshape(ori_shape))), rhogrid, shading="nearest", cmap="RdBu", vmin=np.min(rhogrid), vmax=np.max(rhogrid))
    fig.colorbar(c, ax=ax)
    #plt.plot(eblst, 0.0281/(2.*np.log(3.738*np.log(4.*np.pi * eblst))))
    ax.set_xlabel("$1/(k_n a)^2$")
    ax.set_ylabel("$k_B T m/k_n^2$")
    ax.set_title("$A_{l,k}=\\rho_s/\\rho_{0,k}$")
    plt.show()
    """
    with Pool(12) as p:
        mui = list(tqdm.tqdm(p.imap_unordered(muBEC, parMuLst), total=totMuLen))
    mugrid = np.array(mui).reshape(ori_shape_mu)

    dat = {"mu":mugrid, "eb":ebMugrid.reshape(ori_shape_mu), "Tc":(1./(betaMugrid.reshape(ori_shape_mu)))}
    try:
        with open("Results/becMu.pickle", "wb") as f:
            pickle.dump(dat, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

    with open("Results/becMu.pickle", "rb") as f:
        dat = pickle.load(f)
    mugrid = dat["mu"]
        
    plt.figure()
    crvLst = list()
    for idx in range(len(betaMulst)):
        ci, = plt.plot(ebMugrid.reshape(ori_shape_mu)[idx, 5:-2], mugrid[idx, 5:-2])
        crvLst.append(ci)
    plt.legend(crvLst, [f"$T_c m/ k_n^2$={1./bi:.2e}" for bi in betaMulst])
    plt.xlabel("$\\epsilon_B m/k_n^2$")
    plt.ylabel("$\\mu m/k_n^2$")
    plt.show()

    """
