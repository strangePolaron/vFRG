#!/usr/bin/python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# ///
import numpy as np
import scipy.integrate as itg
import quantumBos as qbos
import matplotlib.pyplot as plt
import parsey as prsy
import scipy.optimize as optm

bareInt = lambda eb, m, cutoff: 1. / ((m/(2.*np.pi))*(np.log(np.sqrt(m * eb)/cutoff)))

class BECAction:
    def __init__(self, eb2boson0, beta, mu, m=1.):
        self.KTSwitch = True
        self.beta = beta
        self.lpar = 0.
        self.mb = m 

        self.cutoff = np.sqrt(m * eb2boson0) #cutoff
        self.g0 = bareInt(eb2boson0 + mu, self.mb, self.cutoff)  #bareInt(eb2boson0, self.mb, self.cutoff)

        #print("g0",self.g0)
        self.mub = mu
        self.rho_init = max(self.mub / self.g0, 0)

        self.ydata = prsy.parseData()
        self.quantumbec = qbos.QuantumAction(self.ydata, self.mb, self.cutoff, self.lpar, self.beta, self.g0, self.rho_init, self.KTSwitch)
        
        assert self.ydata.keysUpd is not None, "ydata.keysUpd is not updated"
        rhoidx = self.ydata.keysUpd.index("rho")
        allidx = self.ydata.keysUpd.index("all")
        
        self.terminFunc = qbos.BECterminFunc(self.mb, self.beta, rhoidx, allidx)
        self.y0 = self.ydata.ylst()
    
        self.sol = itg.solve_ivp(self.eqn, (np.double(0.), np.double(20.)), self.y0, method="LSODA", rtol=1e-7, atol=1e-7, min_step=1e-12, events=self.terminFunc)


    def eqn(self, l, ylst):
        self.ydata.update(ylst)
        dy = self.ydata.zeroVecGen()
        self.quantumbec.dylst(l, dy)
        return dy.ylst()

    def FinalRhoSF(self):
        if self.sol.status==1 or self.sol.status==-1:
            return float(0.)
        assert self.ydata.keysUpd is not None, "ydata.keysUpd is not updated"
        allidx = self.ydata.keysUpd.index("all")
        return float(self.sol.y[allidx, -1])  #float(self.solBEC.y[1,-1]/self.solBEC.y[0,-1])

    def FinalNum(self):
        assert self.ydata.keysUpd is not None, "ydata.keysUpd is not updated (FinalNum)"
        return self.sol.y[self.ydata.keysUpd.index("rho"), -1] #/self.sol.y[self.ydata.keysUpd.index("avv"), -1]

def findMu(targetNum, ebBos, beta, mass):
    mu0 = 10. * targetNum  * np.pi / mass
    func = lambda mui: BECAction(ebBos, beta, mui, mass).FinalNum() - targetNum
    #print(ebBos, beta, func(1e-5), func(mu0))
    return optm.root_scalar(func, method="bisect", bracket=[1e-3,mu0], xtol=1e-5).root


if __name__=="__main__":
    #"""
    #l0 = 2.
    l0 = 0.
    ebBos = 3.
    mu = 0.3
    mass = 1.
    beta = 1000.
    #cutoff = 20.
    bec = BECAction(ebBos, beta, mu, mass)
    solt = bec.sol.t
    soly = bec.sol.y
    print(bec.FinalRhoSF())

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    plt.rcParams['mathtext.fontset'] = "cm"
    plt.figure()
    #c1, = plt.plot(solt, soly[0,:])
    c2, = plt.plot(solt, soly[1,:])
    c3, = plt.plot(solt, soly[2,:])
    c4, = plt.plot(solt, soly[3,:])
    c5, = plt.plot(solt, soly[4,:])
    plt.xlabel("$\\log(\\Lambda / k)$")
    #plt.title(f"$m \\epsilon_B/\\Lambda^2={(mass*ebBos/pow(cutoff, 2)):.2e}$, $2m\\mu/\\Lambda^2={(2.*mass*mu/pow(cutoff,2)):.2e}$, $2mk_BT/\\Lambda^2={(2.*mass/(beta*pow(cutoff, 2))):.2e}$")
    #plt.legend([c1, c2, c3, c4, c5], bec.ydata.keysUpd)
    assert bec.ydata.keysUpd is not None, "ydata.keysUpd is not updated (main)"
    plt.legend([c2, c3, c4, c5], ["$\\rho_{0,k}$","$A_{v,k}$", "$A_{l,k}$","$y_k$"])
    plt.show()

