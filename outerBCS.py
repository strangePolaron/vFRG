#!/usr/bin/python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "numpy",
# ]
# ///
import numpy as np
import parsey as prs

thetaFunc = lambda x: (np.arctan(1000.*x))/np.pi + 1./2.

def nF(z, beta):
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
        rs = np.real(1./(1.+phs*np.exp(exprl)))
        #print(f"rs:{rs}\tphs:{phs}")
        return rs

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
        #print(f"rs:{rs}\tphs:{phs}")
        return rs


class efDiag:
    def __init__(self, prsdata:prs.parseData, beta, muf, mf, isBEC=False):
        self.mf = mf
        self.muf = muf
        self.beta = beta
        self.iz0 = 1.j * np.pi / self.beta
        self.ydata = prsdata
        self.yval = lambda x: self.ydata.value(x)
        self.isBEC = isBEC
        return

    def upd(self, k2):
        self.ekb = k2 / (2. * (2. * self.mf))
        self.ekf = k2 / (2. * self.mf)
        self.ef_muf = self.yval("ef") - self.muf
        if self.isBEC:
            self.eBoson = np.sqrt((self.yval("all")/self.yval("avv")) * self.ekb * (self.ekb + 2. * self.yval("avv") * self.yval("g") * self.yval("rho"))) 
        self.eBosonThr = self.yval("eb") + self.ekb
        self.rho = 0.
        if self.isBEC:
            self.rho = self.yval("rho") / self.yval("avv")
        self.eFermi = np.sqrt(pow(self.ekf + self.ef_muf, 2) + self.rho * pow(self.yval("h")*self.yval("dfac"), 2))
        self.grho = self.yval("g") * self.yval("rho") * self.yval("avv") if self.isBEC else 0.
        self.zfactor = np.sqrt(self.yval("all")/self.yval("avv")) if self.isBEC else 1.
        return

    def gBosonThr(self, z):
        return 1. / (z - self.eBosonThr)  #self.eb

    def gBosonBEC(self, z):
        return (self.zfactor * z + pow(self.zfactor, 2)*(self.ekb + self.grho))/(pow(z,2) - pow(self.eBoson, 2))
        #return (z + (1./self.yval("avv"))*(self.ekb/2. + self.yval("avv") * self.yval("g") * self.yval("rho")) + self.yval("all") * self.ekb/2.)/(pow(z,2) - pow(self.eBoson, 2))
        #return (z + self.yval("all") * self.ekb/2.)/(pow(z,2) - pow(self.eBoson, 2))

    def ResBosBECp(self):
        return (self.zfactor * self.eBoson + pow(self.zfactor, 2) * (self.ekb + self.grho))/(2.*self.eBoson)
        #return (self.eBoson + (1./self.yval("avv"))*(self.ekb/2. + self.yval("avv") * self.yval("g") * self.yval("rho")) + self.yval("all") * self.ekb/2.)/(2.*self.eBoson)
        #return (self.eBoson + self.yval("all") * self.ekb/2.)/(2.*self.eBoson)

    def ResBosBECm(self):
        return (-1. * self.zfactor * self.eBoson + pow(self.zfactor, 2) * (self.ekb + self.grho))/(-2.*self.eBoson)
        #return (-1.*self.eBoson + (1./self.yval("avv"))*(self.ekb/2. + self.yval("avv") * self.yval("g") * self.yval("rho")) + self.yval("all") * self.ekb/2.)/(-2.*self.eBoson)
        #return (-1.*self.eBoson + self.yval("all") * self.ekb/2.)/(-2.*self.eBoson)

    def gFermion(self, z):
        return (z + self.ekf + self.ef_muf)/(pow(z, 2) - pow(self.eFermi, 2))

    def ResFerp(self):
        return (self.eFermi + self.ekf + self.ef_muf)/(2. * self.eFermi)

    def ResFerm(self):
        return (-1.*self.eFermi + self.ekf + self.ef_muf)/(-2. * self.eFermi)

    def efDiag(self, k2):
        self.upd(k2)
        if self.isBEC:
            r1 = (self.gFermion(self.iz0 + self.eBoson))* self.ResBosBECp() * nB(self.eBoson, self.beta)
            r2 = (self.gFermion(self.iz0 - self.eBoson))* self.ResBosBECm() * nB(-1.*self.eBoson, self.beta)
            r3 = self.gBosonBEC(self.eFermi - self.iz0) * self.ResFerp() * nF(self.eFermi, self.beta)
            r4 = self.gBosonBEC(-1.*self.eFermi - self.iz0) * self.ResFerm() * nF(-1.*self.eFermi, self.beta)
            #print("efdiag", self.ResBosBECp(), self.ResBosBECm(), self.gFermion(self.iz0 - self.eBoson), self.gFermion(self.iz0 + self.eBoson))
            #print("efdiag", np.real(r1), np.real(r2), np.real(r3), np.real(r4))
            return np.real(r1 + r2 + r3 + r4) * pow(self.yval("h"), 2)
        else:
            r1 = self.gFermion(self.iz0 + self.eBosonThr) * nB(self.eBosonThr, self.beta)
            r2 = self.gBosonThr(self.eFermi - self.iz0) * self.ResFerp() * nF(self.eFermi, self.beta)
            r3 = self.gBosonThr(-1.*self.eFermi - self.iz0) * self.ResFerm() * nF(-1.*self.eFermi, self.beta)
            rs = np.real(r1 + r2 + r3) * pow(self.yval("h"), 2)
            #print("efdiag", self.ResFerm(), self.gBosonThr(-1.*self.eFermi - self.iz0))
            #self.upd(0.)
            #print("efdiag", self.gBosonThr(-1.*self.eFermi - self.iz0) * self.ResFerm() * nF(-1.*self.eFermi, self.beta))
            return rs


ydatakeysPrompt = ["eb", "ef", "g", "h", "dfac", "rhoF"]

class OuterBCSFermion:
    def __init__(self, prsdata:prs.parseData, mf, beta, gFF, mu, cutoff, lpar=0., h = 1.):
        self.efSwitch = True

        self.muf = mu
        self.mf = mf
        self.efx2mf = 0.
        self.kF2 = (self.muf * 2. * self.mf)
        self.mf_div_2pi = self.mf / (2. * np.pi)
        self.cutoff2 = pow(cutoff*np.exp(-1.*lpar), 2)
        self.beta = beta
        eb = -1. * pow(h, 2) / gFF - 2. * self.muf
        #print("eb0", eb)
        assert eb>0, "already condensed"
        self.ydatakeys = ydatakeysPrompt.copy()
        self.ydata = prsdata
        self.ydata.dataAppend({"eb":eb, "ef":0., "g":1e-4, "h":h, "dfac":1., "rhoF":0.}, self.ydatakeys)
        self.yval = lambda x: self.ydata.value(x)
        #print(f"gFF: {gFF},\t h: {h},\t eb:{self.eb}")
        self.lpar0 = lpar
        self.lpar = lpar
        self.isBEC = False
        self.dEfdiagObj = efDiag(self.ydata, self.beta, self.muf, self.mf, self.isBEC)
        self.dEfdiagfunc = self.dEfdiagObj.efDiag
        return

    def BECcritUpd(self, isBEC):
        self.isBEC = isBEC
        self.dEfdiagObj.isBEC = isBEC
        return
    
    def k_scale(self, lpar):
        self.efx2mf = self.yval("ef") * (2. * self.mf)
        #if self.kF2<self.efx2mf:
        #    print("Fermi sea vanished")
        self.kF2new = max(self.kF2 - self.efx2mf, 0.)
        self.explpar = np.exp(-2.*(lpar-self.lpar0))
        self.k2p = (self.cutoff2 - self.kF2new) * self.explpar + self.kF2new
        #print("k2p", self.k2p)
        #print("kF2new", self.kF2new)
        #print("Cutoff2",self.cutoff2)
        self.holeinvolv = (self.k2p < 2.*self.kF2new)
        self.holetheta = thetaFunc(2.*self.kF2new - self.k2p)
        self.k2h = (2.*self.kF2new - self.k2p) if self.holeinvolv else 0.
        #if self.holeinvolv:
        #    print(self.kF2new, self.k2h, self.k2p)
        self.k2Boson = self.cutoff2 * self.explpar
        self.bosonInvolv = (self.k2Boson < self.k2h)
        #if self.bosonInvolv:
        #    print("k2boson", self.k2Boson)
        return

    def dosCoeff(self):
        """
        By taking this k^2 as those in k_scale function, the dos for particles and holes are the same.
        dosp=dosh=(cutoff^2 - kf^2)exp(-2.*l)
        """
        #self.pefdDOS = (1. - self.explpar) * (2. * self.mf) / (4. * np.pi)
        #self.hefdDOS = (-1. - self.explpar) * (2. * self.mf) / (4. * np.pi)
        self.dosp0 = (self.k2p - self.kF2new) / (2. * np.pi)
        self.dosh0 = (self.k2p - self.kF2new) / (2. * np.pi) if self.holeinvolv else 0.
        self.dosb = self.k2Boson / (2. * np.pi) if self.bosonInvolv else 0.
        return

    def dosCeoff_dEf(self):
        if self.efSwitch:
            diagp = self.dEfdiagfunc(self.k2p)
            diagh = self.dEfdiagfunc(self.k2h) if (self.holeinvolv) else 0.
            #diagh = self.dEfdiagfunc(self.k2h) if (self.holeinvolv and self.bosonInvolv) else 0.
            #diagb = self.dEfdiagfunc(self.k2Boson) if (self.holeinvolv and self.bosonInvolv) else 0.
            numer = diagp * self.dosp0 + diagh * self.dosh0 #+ diagb * self.dosb
            denom = 1. - self.mf_div_2pi * (diagp - diagh)
            dEf = np.real(numer / (denom +1e-2j))
            self.dosp = self.dosp0 + dEf * self.mf_div_2pi
            self.dosh = (self.dosh0 - dEf * self.mf_div_2pi) if self.holeinvolv else 0.
            #print("dos", self.dosp0, self.dosh0)
            return dEf
        else:
            self.dosp = self.dosp0
            self.dosh = self.dosh0
            return 0.
    
    def upd(self, lpar):

        #if y is not None:
        #    self.eb, self.gBoson, self.h, self.rhotot, self.ef = y
        
        self.k_scale(lpar)
        self.dosCoeff()
        self.dEf = self.dosCeoff_dEf()
        self.ek0p = self.k2p/(2.*self.mf) - (self.muf - self.yval("ef"))
        self.ek0h = self.k2h/(2.*self.mf) - (self.muf - self.yval("ef"))
        self.rho = 0.
        if self.isBEC:
            self.rho = self.yval("rho") / self.yval("avv")
        self.ekp_cp = np.sqrt(pow(self.ek0p, 2)+ self.rho * pow(self.yval("h")*self.yval("dfac"), 2))
        self.ekh_cp = np.sqrt(pow(self.ek0h, 2)+ self.rho * pow(self.yval("h")*self.yval("dfac"), 2))
        self.nf_p = nF(self.ekp_cp, self.beta)
        self.nf_h = nF(self.ekh_cp, self.beta)
        self.secsq_p = 4. * self.nf_p * (1. - self.nf_p)
        self.tanh_p = (1. - 2.*self.nf_p)
        self.secsq_h = 4. * self.nf_h * (1. - self.nf_h)
        self.tanh_h = (1. - 2.*self.nf_h)
        #print("ekcp", self.ekp_cp, self.ekh_cp)
        return
 
    def ebDiag(self):
        if (self.beta*self.ekp_cp>1e-6):
            coeff_p = self.dosp * pow(self.yval("h"), 2)
            self.ebrs_p = -1. * coeff_p * self.tanh_p / (2. * self.ekp_cp)
        else:
            #print(self.ekp_cp)
            self.ebrs_p = -1. * pow(self.yval("h"), 2) * self.beta * self.dosp / 4. 
        if self.holeinvolv:
            if (self.beta*self.ekh_cp>1e-6):
                coeff_h = self.dosh * pow(self.yval("h"), 2)
                self.ebrs_h = -1. * coeff_h * self.tanh_h / (2. * self.ekh_cp)
                return self.ebrs_p + self.ebrs_h
            else:
                self.ebrs_h = -1. * self.beta * pow(self.yval("h"), 2) * self.dosh / 4.
                return self.ebrs_p + self.ebrs_h
        else:
            self.ebrs_h = 0.
            return self.ebrs_p

    def gDiag(self):
        if (abs(self.beta*self.ekp_cp)>1e-6):
            coeff_p = self.dosp * pow(self.yval("h"), 4)
            term1p = -1. * self.secsq_p * self.beta * self.ekp_cp 
            term2p = 2. * self.tanh_p  
            rs_p = coeff_p * (term1p + term2p) / (8. * pow(self.ekp_cp, 3))
        else:
            rs_p = self.dosp * pow(self.yval("h"), 4) * pow(self.beta, 3) / 48.
        if self.holeinvolv:
            if (abs(self.beta*self.ekh_cp)>1e-6):
                coeff_h = self.dosh * pow(self.yval("h"), 4)
                term1h = -1. * self.secsq_h * self.beta * self.ekh_cp 
                term2h = 2. * self.tanh_h  
                rs_h = coeff_h * (term1h + term2h) / (8. * pow(self.ekh_cp, 3))
                return rs_h+rs_p
            else:
                rs_h = self.dosh * pow(self.yval("h"), 4) * pow(self.beta, 3) / 48.
                return rs_h+rs_p
        else:
            return rs_p

    
    """
    dZ sign need to be checked
    """
    def dZDiag(self):
        dZp = self.ebrs_p * self.ek0p / (-2. * pow(self.ekp_cp, 2))
        if self.holeinvolv:
            dZh = self.ebrs_h * self.ek0h / (-2. * pow(self.ekh_cp, 2))
            return dZp + dZh
        else:
            return dZp

    def dhRen(self):
        dZ = self.dZDiag()
        return -(1./2.) * dZ * self.yval("h")

    def dDfac(self):
        if self.isBEC:
            ek_bos = self.k2Boson / (4. * self.mf)
            ek_sf = np.sqrt(ek_bos*(ek_bos + 2. * self.yval("g")*self.yval("avv")*self.yval("rho")) * self.yval("all")/self.yval("avv"))
            coeff = self.yval("all")*self.mf/(4.*np.pi*self.yval("rho"))
            coth = - 2. * nB(ek_sf, self.beta)
            return -1. * coeff * ek_sf * coth * self.yval("dfac")
        return 0.

    def drhoF(self):
        integ = (1./2.) * (1. - (1. - 2.*self.nf_p) * self.ek0p/self.ekp_cp) * self.dosp
        if self.holeinvolv:
            integ += (1./2.) * (1. - (1. - 2.*self.nf_h) * self.ek0h/self.ekh_cp) * self.dosh
        return integ

    def dylst(self, l, dy:prs.parseData):  # y = [self.h, self.rhotot]
        self.upd(l)
        dy.data["eb"] = self.ebDiag()
        dy.data["h"] = self.dhRen()
        dy.data["g"] = self.gDiag()
        dy.data["ef"] = self.dEf
        dy.data["dfac"] = self.dDfac()
        dy.data["rhoF"] = self.drhoF()
        return
        #return np.array([self.ebDiag(), self.gDiag(), self.dhRen(), 0.], dtype=np.double)

    def dylst_BCSonly(self, l, ylst):
        self.ydata.update(ylst)
        dy = self.ydata.zeroVecGen()
        self.dylst(l, dy)
        return dy.ylst()



def dh2dZ(dh, h):
    return -2. * dh / h


