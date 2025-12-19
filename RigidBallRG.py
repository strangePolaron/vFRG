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
from matplotlib import pyplot as plt
import scipy.special as spc
import scipy.optimize as optm
import parsey as prs

g0_iv = optm.root(lambda x: spc.iv(1, x) - 1, 1).x[0]


cutoff_convert_2_rspc = lambda cutoff: 2. * np.pi / cutoff

class KT:
    def __init__(self, prsData:prs.parseData, lutK, nMax:int, keysUpdRegis=True):
        self.lpar_add = 0.
        #print(f"g0_iv:{g0_iv}")
        #self.g = 1. / np.pi  #g0_iv / 2. / np.pi #* np.pow(self.rspc_cutoff_a0, 2)
        self.g = 1. / (4.*np.sqrt(3.))  #g0_iv / 2. / np.pi #* np.pow(self.rspc_cutoff_a0, 2)
        self.nMax = nMax
        self.ydatakeys = ["lutK"] + [f"g{idx+1}" for idx in range(self.nMax)]
        self.ydata = prsData  #prs.parseData({"lutK": lutK})
        self.ydata.additem("lutK", lutK)
        vn = self.gn_init()
        for idx in range(self.nMax):
            self.ydata.additem(f"g{idx+1}", vn[idx])
        if keysUpdRegis:
            self.ydata.keysUpdAppend(self.ydatakeys)
        return

    def scaldim(self, n:int):
        return np.double((n**2 * np.pi)/(self.ydata.value("lutK")))
        
    def gn_init(self):
        vn = np.zeros(self.nMax, dtype=np.double)
        for idx in range(self.nMax):
            #self.vn[idx] = self.g * np.exp((2. - self.scaldim(idx+1))*self.lpar_0)
            #vn[idx] = self.g #* np.exp((2.)*self.lpar_0)
            vn[idx] = self.g #* np.exp(-1.*self.scaldim(idx+1)*np.log(2.)/2.)
        return vn

    def yGen(self):
        return self.ydata.ylst()

    def parUpd(self, ynew):
        self.ydata.update(ynew)
        return
    
    def eqRHS_ydata(self, _, dy:prs.parseData):
        gilen = len(self.ydatakeys) - 1
        y_lutK = 4. * pow(np.pi, 3) * sum([(np.double(varName[1:]) ** 2) * np.pow(self.ydata.value(varName), 2) for varName in self.ydatakeys[1:]])
        #dy = self.ydata.zeroVecGen()
        dy.data["lutK"] = y_lutK
        for idx, varName in enumerate(self.ydatakeys[1:]):
            dy.data[varName] = (2. - self.scaldim(int(varName[1:]))) * self.ydata.value(varName)#*(1-dhl)
        for idx in range(gilen):
            tmpi = 0.
            for jdx in range(idx + 1, gilen):
                tmpi -= self.ydata.value(self.ydatakeys[jdx+1]) * self.ydata.value(self.ydatakeys[jdx-idx])  #self.vn[jdx]*self.vn[jdx - idx - 1]
            for jdx in range(idx):
                tmpi -= self.ydata.value(self.ydatakeys[jdx+1]) * self.ydata.value(self.ydatakeys[idx-jdx]) / 2.
            dy.data[self.ydatakeys[idx+1]] += 2. * np.pi * tmpi
        return

    def eqRHS_onlyKT(self, lpar, y):
        self.parUpd(y)
        dy = self.ydata.zeroVecGen()
        self.eqRHS_ydata(lpar, dy)
        return dy.ylst()


if __name__=="__main__":
    cutoffLambda = 12.
    a_0 = np.double(cutoff_convert_2_rspc(cutoffLambda))
    k_0 = np.double(0.1 * cutoffLambda)
    m = np.double(1.)
    rho = np.double(12.)
    nMax = 1
    prsdat = prs.parseData()
    kt_obj = KT(prsdat, m/rho, nMax)
    y0 = kt_obj.yGen()
    sol = itg.solve_ivp(kt_obj.eqRHS_onlyKT, (np.double(0.), np.double(10.)), y0, method="LSODA", rtol=1e-6, atol=1e-8, min_step=1e-12)
    plt.figure()
    plt.plot(1./sol.y[0,:], sol.y[1,:])
    #plt.plot(1./sol.y[0,:], sol.y[2,:])
    plt.show()
