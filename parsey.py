#!/usr/bin/python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "numpy",
# ]
# ///
from typing import Dict
import numpy as np
#from functools import reduce

class parseData:
    def __init__(self, data=None, keysUpd=None):
        if data is None:
            self.data = dict()
        else:
            self.data = data.copy()
        self.keysUpd = keysUpd

    def keysUpdFunc(self, keysUpd):
        self.keysUpd = list(keysUpd)

    def keysUpdAppend(self, keysNew):
        if self.keysUpd is None:
            self.keysUpd = keysNew.copy()
        else:
            for keyi in keysNew:
                if (keyi in self.keysUpd) is False: 
                    self.keysUpd.append(keyi)
        return
        
    def ylst(self):
        if self.keysUpd is None:
            return np.array(list(self.data.values()), dtype=np.double)
        else:
            return np.array([self.data[k] for k in self.keysUpd], dtype=np.double)

    def additem(self, kargs, values):
        self.data.update({kargs: values})
        return

    def dataCpy(self, dataDict:Dict):
        self.data = dataDict.copy()
        return

    def dataAppend(self, dataDict:Dict, keysNew=None):
        self.data.update(dataDict)
        if keysNew is None:
            pass 
        else:
            self.keysUpdAppend(keysNew)
        return

    def dataConcat(self, dataDict:parseData):
        self.dataAppend(dataDict.data, dataDict.keysUpd)
        return

    def update(self, ylst):
        if self.keysUpd is None:
            for idx, k in enumerate(self.data.keys()):
                self.data[k] = ylst[idx]
        else:
            for idx, k in enumerate(self.keysUpd):
                self.data[k] = ylst[idx]

    def value(self, key):
        return self.data[key]

    def subylst_keys(self, keyLst):
        return np.array([self.data[k] for k in keyLst], dtype=np.double)

    """Shallow copy"""
    def copy(self):
        tmp = parseData()
        if self.keysUpd is None:
            tmp.keysUpd = None
        else:
            tmp.keysUpd = self.keysUpd.copy()
        tmp.data  = self.data.copy()
        return tmp

    def zeroVecGen(self):
        tmp = parseData()
        tmp.keysUpd = self.keysUpd
        for k in self.data.keys():
            tmp.data.update({k: 0.})
        return tmp

    def sum_other(self, other):
        assert (len(self.data) == len(other.data)), "not addable"
        for ki in self.data.keys():
            self.data[ki] += other.data[ki]
        return self
        


