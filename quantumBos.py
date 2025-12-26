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
import RigidBallRG as rbrg
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
        return np.real(1./(1.-phs*np.exp(exprl)))

ydatakeysPrompt = ["g", "rho", "avv", "all"]

class QuantumAction:
    def __init__(self, prsdata:prs.parseData, m, cutoff, lpar_init0, beta, g0, rho0, KTSwitch=True):
        self.KTSwitch = KTSwitch
        self.m = m 
        self.cutoff = cutoff
        self.lpar_0 = lpar_init0
        self.lpar = self.lpar_0
        #print(f"QuantumAction, lpar={self.lpar}")
        self.beta = beta
        self.k = self.cutoff * np.exp(-1.*self.lpar)
        self.cutoff_0 = cutoff * np.exp(-1.*self.lpar_0)

        self.ydatakeys = ydatakeysPrompt.copy() 
        self.ydata = prsdata
        self.ydata.dataAppend({"g": g0, "rho": rho0, "avv": 1., "all": 1.}, self.ydatakeys)
        self.yval = lambda x: self.ydata.value(x)
        self.ktStart = False
        self.updInternalVar()
        self.gnMax = 1
        self.rbAct = rbrg.KT(self.ydata, self.lutK(), self.gnMax, False)
        self.ktStart = False
        if KTSwitch:
            #self.ydata.dataConcat(self.rbAct.ydata)
            #self.ydatakeys += self.rbAct.ydatakeys[1:]  #[f"g{idx+1}" for idx in range(self.gnMax)]
            tmp = self.rbAct.ydatakeys.copy()
            tmp.remove("lutK")
            self.ydata.keysUpdAppend(tmp) 

    def updInternalVar(self):
        self.ek = pow(self.k, 2)/(2.*self.m)
        self.k2 = pow(self.k, 2)
        self.Ek = self.Ek_pole()
        self.nbval = -1. * nB(self.Ek, self.beta)
        self.coth = 1. + 2. * self.nbval
        self.csch2 = 4. * self.nbval * (self.nbval + 1.)
        self.all_div_avv_sqrt = np.sqrt(self.yval("all") / self.yval("avv"))
        self.dosCoeff = pow(self.k, 2) / (2. * np.pi)
        #self.ktStart = self.ktStart or self.isKTstart()

        #ktStartPre = self.ktStart
        self.ktStart = self.ktStart or self.isKTstart()
        #self.ktStart = self.isKTstart()
        #if (not ktStartPre) and self.ktStart:
        #    print(f"KT begins, dlpar={self.lpar-self.lpar_0}")
        return
        
    def Ek_pole(self):
        return np.sqrt(self.ek * (self.ek + 2.*self.yval("g")*self.yval("rho") * self.yval("avv")) * self.yval("all") / self.yval("avv"))
    
    """
    def drho(self, drhoSF, drho0, drhotot, dK=0.):
        drhoSF_Final = drhoSF + self.yval("all") * drhotot  + self.drhoKT(dK)
        drhotot_Final = drhotot
        drho0_Final = drho0 + self.yval("avv") * drhotot
        return drhoSF_Final, drhotot_Final, drho0_Final
    """
        
    def rho1_diag(self):
        return self.yval("all") * (-1. * self.ek * self.coth / self.Ek + 1./self.all_div_avv_sqrt)/2.
        #return self.yval("all") * (-1. * self.ek * self.coth / self.Ek) + self.all_div_avv_sqrt
        
    def rho2_diag(self):
        #coeff = -1. * pow(self.g*self.arho_ll, 2)
        coeff = -1. * pow(self.yval("g")*self.yval("all"), 2)
        term1 = pow(self.ek, 2) * self.coth / (2. * pow(self.Ek, 3))
        term2 = pow(self.ek/(2.*self.Ek), 2) * self.beta * self.csch2  #-1. * self.beta * pow(self.ek, 2) * self.csch2 * pow(self.diff * self.ek + self.allxrho0 * self.grhox2val, 2)/ (16. * pow(self.Ek*self.rho0*self.rho0, 2))
        return coeff * (term1 + term2)

    def vll_diag(self):
        #return -1. * pow(self.arho_ll, 2) * self.ek *self.beta * self.csch2 / 4.
        return -1. * pow(self.yval("all"), 2) * self.ek *self.beta * self.csch2 / 4.

    def vvv_diag(self):
        #return (self.arho_ll * self.arho_vv) * (self.ek * self.coth / self.Ek  - 1. / self.all_div_avv_sqrt)
        return (self.yval("all") * self.yval("avv")) * (self.ek * self.coth / self.Ek  - 1. / self.all_div_avv_sqrt)

    def upd(self, l):
        self.lpar = l
        self.k = self.cutoff * np.exp(-1.*self.lpar)

        #self.ydata.update(y)
        self.updInternalVar()
        if self.KTSwitch and self.ktStart:
            lutK = self.lutK()
            self.ydata.data["lutK"] = lutK
        #    myKTpar = self.getyKT()  #np.array([lutK] + list(self.ydata.subylst_keys(self.rbAct.ydatakeys[1:])), dtype=np.double)
        #    self.rbAct.parUpd(myKTpar)
        return

    def lutK(self):
        #return self.m / (self.rhoTot * self.arho_ll) / self.beta
        return self.m  / (self.yval("rho") * self.yval("all")) / self.beta

    def drhoKT(self, dK):
        Kkt = self.lutK()
        return -1. * self.m * dK / pow(Kkt, 2) / self.beta

    def dylst(self, l, dy:prs.parseData):
        self.upd(l)

        #dy = self.ydata.zeroVecGen()
        
        #mydKT = self.rbAct.eqRHS(l, np.array([self.lutK()] + list(y[4:]), dtype=np.double))
        """
        dZleg = 0.
        if callable(self.dZOuter):
            dZleg = float(self.dZOuter.__call__(l, y))
        dgOuterContrib = 0.
        if callable(self.gOuterDiag):
            dgOuterContrib = float(self.gOuterDiag.__call__(l, y))
        drhoOuterContrib = 0.
        if callable(self.ebOuterDiag):
            drhoOuterContrib = -1. * float(self.ebOuterDiag.__call__(l, y)) / self.g
        dgren, drhoTotren, drho0ren, drhoSFren = self.reNorm(dZleg)
        """
        drhoTot = self.rho1_diag() * self.dosCoeff  #+ drhoTotren + drhoOuterContrib
        dg = self.rho2_diag() * self.dosCoeff  #+ dgren + dgOuterContrib
        dall = self.vll_diag() * self.dosCoeff / self.yval("rho")  #self.rhoTot  #+ drhoSFren
        davv = self.vvv_diag() * self.dosCoeff / self.yval("rho")  #self.rhoTot  #+ drho0ren
        
        dy.data["rho"] += drhoTot
        dy.data["g"] += dg
        dy.data["all"] += dall
        dy.data["avv"] += davv

        """
            dK implementation
        """
        if self.KTSwitch and self.ktStart:
            #drhoSF_Final, drhoTot_Final, drho0_Final = self.drho(drhoSF, drho0, drhoTot, float(mydKT[0]))
            #mydKT = self.rbAct.eqRHS(l, self.getyKT())
            self.rbAct.eqRHS_ydata(l, dy)
            #dall += self.drhoKT(float(mydKT[0])) / self.yval("rho") #rhoTot
            dy.data["all"] += self.drhoKT(float(dy.data["lutK"])) / self.yval("rho") #rhoTot
            #dy.data["all"] = dall
            #for idx, varName in enumerate(self.rbAct.ydatakeys[1:]):
            #    dy.data[varName] = mydKT[idx]
            #            #np.array([drhoTot, dg, dall, davv] + list(mydKT[1:]), dtype=np.double)
        return  #np.array([drhoTot, dg, dall, davv] + [0.] * len(mydKT[1:]), dtype=np.double)

    def dylst_onlyBos(self, l, y):
        self.ydata.update(y)
        dy = self.ydata.zeroVecGen()
        self.dylst(l, dy)
        return dy.ylst()
    
    def reNorm(self, dZleg=0.):
        """
            rho_r = rho (1+dZleg)
        """
        #return  -2. * self.g * dZleg, self.rhoTot * dZleg
        return  -2. * self.yval("g") * dZleg, self.yval("rho") * dZleg
    
    def healLength(self):
        #return np.sqrt(1./(2.*self.m*self.g*self.rhoTot*self.arho_vv))
        return np.sqrt(1./(2.*self.m*self.yval("g")*self.yval("rho")*self.yval("avv")))

    def isKTstart(self):
        return (self.healLength()<=((2.*np.pi)/self.k))

    def meanfieldCrit(self):
        myKval = self.lutK()
        return myKval<(np.pi/2.)


class BECterminFunc:
    def __init__(self, m, beta, rhoidx, allidx):
        self.terminal = True
        self.direction = -1.
        self.m = m
        self.beta =beta
        self.rhoidx = rhoidx  #ydatakeysPrompt.index("rho")
        self.allidx = allidx  #ydatakeysPrompt.index("all")

    def __call__(self, _, y):
        #print("all", y[self.allidx])
        return (self.beta/self.m)*(y[self.rhoidx] * y[self.allidx])-1e-3
