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
import thermBos as tbos
import parsey as prsy
import outerBCS as bcs
import matplotlib.pyplot as plt
import scipy.optimize as optm

bareInt = lambda eb, m, cutoff: 1. / ((m/(2.*np.pi))*(np.log(np.sqrt(m * eb)/np.sqrt(pow(cutoff, 2) + m * eb))))

class BCSAction:
    def __init__(self, eb0, beta, mu, cutoff, mf=1., h=10.):
        self.efSwitch = False
        self.KTswitch = True

        self.lpar = 0.
        self.mf = mf
        self.mb = 2. * self.mf
        self.muf = mu
        self.beta = beta
        self.cutoff = cutoff
        self.h0 = h
        self.h = h
        self.gFF0 = bareInt(eb0, self.mf, cutoff)
        self.gFF0 = 1. / (1./self.gFF0 + eb0/pow(self.h, 2))

        self.ydata = prsy.parseData()
        self.bcsFer = bcs.OuterBCSFermion(self.ydata, self.mf, self.beta, self.gFF0, self.muf, self.cutoff, self.lpar, self.h)
        self.bcsFer.efSwitch = self.efSwitch
        #print(self.ydata.data)
        self.thrBos = tbos.ThermalBoson(self.ydata, self.mb, self.cutoff, self.beta, self.ydata.value("g"), self.ydata.value("eb"), 0., self.lpar)

        assert self.ydata.keysUpd is not None, "ydata.keysUpd is not updated"
        self.thrgidx = self.ydata.keysUpd.index("g")
        self.threbidx = self.ydata.keysUpd.index("eb")
        self.efidx = self.ydata.keysUpd.index("ef")
        self.hidx = self.ydata.keysUpd.index("h")

        self.terminFuncThr = tbos.ThrterminFunc(self.mb, self.beta, self.thrgidx, self.threbidx)
        self.y0Thr = self.ydata.ylst()
        
        self.solThr = itg.solve_ivp(self.thrEqn, (np.double(0.), np.double(20.)), self.y0Thr, method="LSODA", rtol=1e-7, atol=1e-7, min_step=1e-12, events=self.terminFuncThr)
        
        self.ydata.update(self.solThr.y[:,-1])
        self.becShift = (self.solThr.status==1)
        
        if self.becShift:
            #print("Shifted to BEC")
            self.rho_init = -1. * self.solThr.y[self.threbidx, -1] / self.solThr.y[self.thrgidx, -1]
            self.becBos = qbos.QuantumAction(self.ydata, self.mb, self.cutoff, self.solThr.t[-1], self.beta, self.solThr.y[self.thrgidx, -1], self.rho_init, self.KTswitch)
            self.bcsFer.BECcritUpd(True)
            self.becrhoidx = self.ydata.keysUpd.index("rho")
            self.becallidx = self.ydata.keysUpd.index("all")
            self.becavvidx = self.ydata.keysUpd.index("avv")
            #self.becvkidx = self.ydata.keysUpd.index("vk")
            self.terminFuncBEC = qbos.BECterminFunc(self.mb, self.beta, self.becrhoidx, self.becallidx)
            self.y0BEC = self.ydata.ylst()
            self.solBEC = itg.solve_ivp(self.spfEqn, (np.double(self.solThr.t[-1]), np.double(20.)), self.y0BEC, method="LSODA", rtol=1e-7, atol=1e-7, min_step=1e-12, events=self.terminFuncBEC)
        return
        
    def thrEqn(self, l, ylst):
        self.ydata.update(ylst)
        dybcs = self.ydata.zeroVecGen()
        dyThr = self.ydata.zeroVecGen()
        self.bcsFer.dylst(l, dybcs)
        self.thrBos.dylst(l, dyThr)
        dZ = bcs.dh2dZ(dybcs.value("h"), self.ydata.value("h"))
        debZ = -1. * (self.ydata.value("eb")) * dZ
        dgZ = -2. * self.ydata.value("g") * dZ
        
        dy = dybcs.sum_other(dyThr)
        dy.data["eb"] += debZ
        dy.data["g"] += dgZ
        
        return dy.ylst()

    def spfEqn(self, l, ylst):
        #print(l)
        self.ydata.update(ylst)
        self.ydata.data["rho"] = max(self.ydata.data["rho"], 1e-5)
        self.ydata.data["avv"] = max(self.ydata.data["avv"], 1e-5)
        self.ydata.data["all"] = max(self.ydata.data["all"], 1e-5)

        dybcs = self.ydata.zeroVecGen()
        dybec = self.ydata.zeroVecGen()
        self.bcsFer.dylst(l, dybcs)
        self.becBos.dylst(l, dybec)
        dZ = bcs.dh2dZ(dybcs.value("h"), self.ydata.value("h"))
        dgZ = -2. * self.ydata.value("g") * dZ
        drhoZ = self.ydata.value("rho") * dZ

        dallZ = self.ydata.value("all") * dZ
        davvZ = self.ydata.value("avv") * dZ

        drhoBCS = -1. * dybcs.value("eb") / self.ydata.value("g")
        dy = dybec.sum_other(dybcs)
        dy.data["g"] += dgZ 
        dy.data["rho"] += drhoZ + drhoBCS
        dy.data["eb"] = 0.
        dy.data["all"] += dallZ
        dy.data["avv"] += davvZ
        #dy.data["vk"] = self.ydata.value("vk") * (2. *  dZ)
        return dy.ylst()

    def FinalRhoSF(self):
        if self.becShift :
            if self.solBEC.status==0:
                return float(self.solBEC.y[self.becallidx, -1])  #/np.sqrt(self.solBEC.y[self.becvkidx, -1]))
        return float(0.)

    def FinalNum(self):
        """
        ef = self.solBEC.y[self.efidx, -1] if self.becShift else self.solThr.y[self.efidx, -1]
        if self.becShift:
            h = self.solBEC.y[self.hidx, -1] 
            assert self.ydata.keysUpd is not None, "Check ydata.keysUpd"
            dfact = self.solBEC.y[self.ydata.keysUpd.index("dfac"), -1]
            rho = max(self.solBEC.y[self.becrhoidx, -1] * self.solBEC.y[self.becavvidx, -1], 0.)
            #if rho<0:
            #    print(f"rho\t{self.solBEC.y[self.becrhoidx, -1]}\tavv\t{self.solBEC.y[self.becavvidx, -1]}")
            #x = np.sqrt(pow(abs(self.muf-ef), 2) + rho*pow(h * dfact,2))
            x = np.sqrt(pow(abs(self.muf-ef), 2))
            x0 = (x + self.muf - ef)/2.
        else:
            x = self.muf - ef
            x0 = self.muf - ef
        x = self.muf - ef
        x0 = self.muf - ef

        if x*self.beta>50.:
            y = x
        elif x*self.beta<-50.:
            y = 0.
        else:
            y = x0 + np.log(1 + np.exp(-1.* self.beta*x))/self.beta
        """
        if self.becShift:
            assert self.ydata.keysUpd is not None, "Check ydata.keysUpd"
            ferNum = self.solBEC.y[self.ydata.keysUpd.index("rhoF"), -1]
        else:
            assert self.ydata.keysUpd is not None, "Check ydata.keysUpd"
            ferNum = self.solThr.y[self.ydata.keysUpd.index("rhoF"), -1]
        if self.becShift:
            yb = max(self.solBEC.y[self.becrhoidx, -1], 0.) #/ self.solBEC.y[self.becavvidx, -1]
        else:
            xb = self.solThr.y[self.threbidx, -1]
            if xb*self.beta>50.:
                yb = 0.
            else:
                yb = -1. * (self.mb / (2. * np.pi)) * np.log(np.abs(1. - np.exp(-1.* self.beta*xb)))/self.beta
        #ferNum = 2. * (y)* (self.mf / (2.*np.pi))
        bosNum = yb * pow(self.solBEC.y[self.hidx, -1] / self.h0, 2) if self.becShift else 0.
        totNum = ferNum + 2. * bosNum
        #if self.becShift:
        #    print(self.muf, self.solBEC.status, totNum, ferNum, bosNum)
        return totNum

def findMu(targetNum, eb, beta, cutoff, mass):
    mu0 = targetNum  * np.pi / mass
    func = lambda mui: BCSAction(eb, beta, mui, cutoff, mass).FinalNum() - targetNum
    lft = func(-1.*eb)
    rht = func(5.*mu0)
    if lft*rht>0:
        print(f"{eb:.2f},\t{beta:.2f},\t{lft:.2f},\t{rht:.2f}")
    return optm.root_scalar(func, method="bisect", bracket=[-1.*eb/2.+1e-4,5.*mu0], xtol=1e-6).root
    
            
if __name__=="__main__":
    #l0 = 2.
    kF = 2.
    mass = 1.
    ef = (kF**2)/(2.*mass)
    l0 = 0.
    eb = 1.2 #* ef
    mu = -0.22 #* ef #0.31835222466
    mass = 1.
    beta = 200. #/ ef  #1. / 0.000405   #1000.
    beta1 = 1000. / ef #1. / 0.000405   #1000.
    beta2 = (1./eb) / 7.5e-3  / ef #1000.
    beta3 = (1./eb) / 2e-2 / ef  #1000.
    cutoff = 3.
    #mu = findMu(1.*mass/np.pi, eb, beta, cutoff, mass)
    #print("mu", mu)
    mulst = np.arange(-0.35, 0.6, 0.006) * ef
    betalst = 1./np.arange(0.1/beta, 10./beta, 0.1/beta)

    #"""
    mu2D, beta2D = np.meshgrid(mulst, betalst)
    nTot = np.zeros(mu2D.shape)
    lx, ly = mu2D.shape
    for idx in range(lx):
        for jdx in range(ly):
            nTot[idx, jdx] = BCSAction(eb , beta2D[idx, jdx], mu2D[idx, jdx], cutoff, mass).FinalNum()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    plt.rcParams['mathtext.fontset'] = "cm"

    fig, ax = plt.subplots()   
    ax.ticklabel_format(style="sci",scilimits=(-2,2))
    c = ax.pcolormesh(mu2D/eb, 1./(beta2D*eb), nTot/eb/mass, shading="nearest", cmap="RdBu", vmin=np.min(nTot), vmax=np.max(nTot)/6.)
    ax.set_xticks(np.arange(-0.5,1.01,0.5))
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("$\\mu_F/\\epsilon_B$")
    ax.set_ylabel("$k_B T /\\epsilon_B$")
    ax.set_title("$n_{tot}/(m_F\\epsilon_B)$")
    plt.show()
 
    #"""


    """
    numlst = [BCSAction(eb, beta, mui, cutoff, mass).FinalNum() for mui in mulst]
    numlst1 = [BCSAction(eb, beta1, mui, cutoff, mass).FinalNum() for mui in mulst]
    print("0.75")
    numlst2 = [BCSAction(eb, beta2, mui, cutoff, mass).FinalNum() for mui in mulst]
    numlst3 = [BCSAction(eb, beta3, mui, cutoff, mass).FinalNum() for mui in mulst]
    plt.figure()
    curv1, = plt.plot(mulst/eb, np.array(numlst)/eb/mass)
    curv2, = plt.plot(mulst/eb, np.array(numlst1)/eb/mass)
    curv3, = plt.plot(mulst/eb, np.array(numlst2)/eb/mass)
    curv4, = plt.plot(mulst/eb, np.array(numlst3)/eb/mass)
    plt.xlabel("$\\mu/\\epsilon_b$")
    plt.ylabel("$n_{tot}/(m_F\\epsilon_b)$")
    plt.legend([curv1, curv2, curv3, curv4], [f"T={1./beta:.2e}", f"T={1./beta1:.2e}", f"T={1./beta2:.2e}", f"T={1./beta3:.2e}"])
    plt.show()
    """

    """
    bcsAct = BCSAction(eb, beta, mu, cutoff, mass)
    tThr = bcsAct.solThr.t
    print(bcsAct.becShift)
    print(bcsAct.FinalNum(), 1.*mass/np.pi)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    plt.rcParams['mathtext.fontset'] = "cm"


    plt.figure()
    eb_curv, = plt.plot(tThr, bcsAct.solThr.y[bcsAct.threbidx, :])
    plt.xlabel("$\\log(\\Lambda/k)$")
    if bcsAct.becShift:
        print(bcsAct.solBEC.status)
        tBEC = list(bcsAct.solBEC.t)
        tTot = list(tThr) + list(tBEC)
        gTot = list(bcsAct.solThr.y[bcsAct.thrgidx, :]) + list(bcsAct.solBEC.y[bcsAct.thrgidx, :])
        assert bcsAct.ydata.keysUpd is not None, "keysUpd is none, ef index not found"
        efidx = bcsAct.ydata.keysUpd.index("ef")
        hidx = bcsAct.ydata.keysUpd.index("h")
        efTot = list(bcsAct.solThr.y[efidx, :]) + list(bcsAct.solBEC.y[efidx, :])
        hTot = list(bcsAct.solThr.y[hidx, :]) + list(bcsAct.solBEC.y[hidx, :])
        avvidx = bcsAct.ydata.keysUpd.index("avv")
        g1idx = bcsAct.ydata.keysUpd.index("g1")
        avvBEC = bcsAct.solBEC.y[avvidx, :]
        g1BEC = bcsAct.solBEC.y[g1idx, :]
        #dfacBEC = bcsAct.solBEC.y[bcsAct.ydata.keysUpd.index("dfac"),:]
        #print(f"{bcsAct.solBEC.y[bcsAct.becallidx, -1]*bcsAct.solBEC.y[bcsAct.becrhoidx, -1]*np.pi/(2.*mass)}")

        rhocurv, = plt.plot(tBEC, bcsAct.solBEC.y[bcsAct.becrhoidx,:])
        allcurv, = plt.plot(tBEC, bcsAct.solBEC.y[bcsAct.becallidx,:])
        avvcurv, = plt.plot(tBEC, avvBEC)
        g1_curv, = plt.plot(tBEC, g1BEC)
        #dfaccurv, = plt.plot(tBEC, dfacBEC)

        gbbcurv, = plt.plot(tTot, gTot)
        #ef_curv, = plt.plot(tTot, efTot)
        h__curv, = plt.plot(tTot, hTot)

        plt.legend([eb_curv, rhocurv, allcurv, avvcurv, g1_curv, gbbcurv, h__curv], ["$\\epsilon_b$", "$\\rho_{0,k}$", "$A_{l,k}$", "$A_{v,k}$", "$y_k$", "g", "h"])
        #plt.legend([eb_curv, rhocurv, allcurv, avvcurv, g1_curv, gbbcurv, h__curv, dfaccurv], ["$\\epsilon_b$", "$\\rho_{0,k}$", "$A_{l,k}$", "$A_{v,k}$", "$y_k$", "g", "h", "$|\\langle d\\rangle|\\sqrt{\\frac{A_{v,k}}{\\rho_{0,k}}}$"])
        #plt.legend([eb_curv, rhocurv, allcurv, avvcurv, gbbcurv, h__curv, dfaccurv], ["$\\epsilon_b$", "$\\rho_{0,k}$", "$A_{l,k}$", "$A_{v,k}$", "g", "h", "$|\\langle d\\rangle|\\sqrt{\\frac{A_{v,k}}{\\rho_{0,k}}}$"])
    else:
        assert bcsAct.ydata.keysUpd is not None, "keysUpd is none, ef index not found"
        efidx = bcsAct.ydata.keysUpd.index("ef")
        hidx = bcsAct.ydata.keysUpd.index("h")
        #efcurv, = plt.plot(bcsAct.solThr.t, bcsAct.solThr.y[efidx, :])
        hcurv, = plt.plot(bcsAct.solThr.t, bcsAct.solThr.y[hidx, :])
        plt.legend([eb_curv,hcurv], ["eb", "h"])
    #plt.plot(tThr, bcsAct.solThr.y[1, :])
    #plt.plot(tThr, bcsAct.solThr.y[2, :])
    #plt.plot(tThr, bcsAct.solThr.y[3, :])
    plt.show()
    """
