#!/usr/bin/python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "numpy",
# ]
# ///
import numpy as np
import parsey as prs

def nB(z, beta):
    rl:np.double = np.real(z)  
    exprl:np.double = (beta*rl)
    if exprl<-50.:
        return np.double(1.)   
    elif exprl>50.:
        return np.double(0.)
    else:
        im:np.double = np.imag(z) 
        phs_log:np.double = beta*im
        phs:np.complex128 = np.cos(phs_log) + 1.j*np.sin(phs_log)
        rs = np.real(1./(1.-phs*np.exp(exprl)))
        return rs

ydatakeysPrompt = ["g", "eb"]

class ThermalBoson:
    def __init__(self, prsdata, m, cutoff, beta, g0, eb0, mu, lpar0=0.):
        self.m = m
        self.cutoff = cutoff
        self.beta = beta
        self.ydatakeys = ydatakeysPrompt.copy()
        self.ydata = prsdata
        self.ydata.dataAppend({"g": g0, "eb":eb0 - mu}, self.ydatakeys)
        #print(self.ydata.data)
        self.yval = lambda x: self.ydata.value(x)
        self.lpar = lpar0 

    def isCondensing(self):
        if (self.yval("eb") < -1e-2 * self.yval("g")) and (self.yval("g")>0):
            return True
        else:
            return False

    def reNorm(self, dZleg=0.):
        return -1.*self.yval("eb")*dZleg, -2.*self.yval("g")*dZleg

    def ekCalc(self):
        self.ekval = pow(self.cutoff, 2) * np.exp(-2.*self.lpar) / (2.*self.m)
        return

    def nbCalc(self):
        self.nbval = -1. * nB(self.ekval + self.yval("eb"), self.beta)
        return

    def dosCalc(self):
        self.dosCoeff = self.ekval * self.m / np.pi

    def lp_g(self):
        return -1. * pow(self.yval("g"), 2)*(1. + 2.*self.nbval) / (2.*(self.ekval + self.yval("eb")))

    def lp_eb(self):
        return 2. * self.yval("g") * self.nbval

    def upd(self, l):
        self.lpar = l
        #self.ydata.update(y)
        self.ekCalc()
        self.nbCalc()
        self.dosCalc()
        return

    def dylst(self, l, dy:prs.parseData):
        self.upd(l)
        dg = self.lp_g() * self.dosCoeff
        deb = self.lp_eb() * self.dosCoeff
        #dy = self.ydata.zeroVecGen()
        dy.data["g"] = dg
        dy.data["eb"] = deb
        return  #dy.ylst()  #np.array([deb, dg], dtype=np.double)

    def dylst_onlythrm(self, l, y):
        self.ydata.update(y)
        dy = self.ydata.zeroVecGen()
        self.dylst(l, dy)
        return dy.ylst()


class ThrterminFunc:
    def __init__(self, m, beta, gidx, ebidx):
        self.terminal = True
        self.direction = -1.
        self.m = m
        self.beta = beta
        self.gidx = gidx  #ydatakeysPrompt.index("g")
        self.ebidx = ebidx  #ydatakeysPrompt.index("eb")

    def __call__(self, _, y):
        return max(-1.*self.m*y[self.gidx], self.beta*y[self.ebidx]/(self.m*abs(y[self.gidx])))+1e-4
