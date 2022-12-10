# -*- coding: utf-8 -*-
"""
Created on Tue May 19 18:36:45 2020

@author: John Greendeer Lee
"""
import numpy as np
import numpy.random as rand
import scipy.special as sp
import newt.clebschGordan as cg


def test_CG1():
    """
    Check the < j1 j2 m1 m2 | 0 0 > special case.
    """
    CGs = np.zeros(100)
    CGs2 = np.zeros(100)
    CGpred = np.zeros(100)
    for k in range(100):
        j1 = rand.randint(0, 6)
        j2 = rand.randint(0, 6)
        m1 = rand.randint(-j1, j1+1)
        m2 = rand.randint(-j2, j2+1)
        CGs[k] = cg.cgCoeff(j1, j2, m1, m2, 0, 0)
        CGs2[k] = cg.cgCoeff2(j1, j2, m1, m2, 0, 0)
        deltaj = (j1 == j2)
        deltam = (m1 == -m2)
        CGpred[k] = deltaj*deltam*(-1)**(j1-m1)/np.sqrt(2*j1+1)
    assert (abs(CGs - CGpred) < 6*np.finfo(float).eps).all()
    assert (abs(CGs2 - CGpred) < 6*np.finfo(float).eps).all()


def test_CG2():
    """
    Check that < j1 j2 j1 j2 | (j1+j2) (j1+j2) > = 1.
    """
    CGs = np.zeros(36)
    for j1 in range(6):
        for j2 in range(6):
            CGs[j1*6 + j2] = cg.cgCoeff(j1, j2, j1, j2, j1+j2, j1+j2)
    assert (abs(CGs - 1) < 15*np.finfo(float).eps).all()


def test_CG3():
    """
    Check that < j1 j1 m1 (-m1) | (2j1) 0 > =
    (2j1)!^2/((j1-m1)!(j1+m1)!sqrt((4j1)!))
    """
    CGs = np.zeros(36)
    CGspred = np.zeros(36)
    for j1 in range(6):
        for m1 in range(-j1, j1+1):
            CGs[j1*j1+j1+m1] = cg.cgCoeff(j1, j1, m1, -m1, 2*j1, 0)
            jmfac = sp.factorial(j1-m1)
            jpfac = sp.factorial(j1+m1)
            j4fac = sp.factorial(4*j1)
            j2fac = sp.factorial(2*j1)
            CGspred[j1*j1+j1+m1] = (j2fac)**2/(jmfac*jpfac*np.sqrt(j4fac))
    assert (abs(CGs - CGspred) < 6*np.finfo(float).eps).all()


def test_CG_q2Q():
    """
    Check that < L+l' l' 0 -m' | L -m' > = (-1)^(l'-m')(L+l')!
    [(2l')!(2L+1)!/(2(L+l')+1)!(l'+m')!(l'-m')!(L+m')!(L-m')!]^1/2
    """
    def cgval(L, lp, mp):
        val = (-1)**(lp-mp)*sp.factorial(L+lp)
        radval = sp.gammaln(2*lp+1)+sp.gammaln(2*L+2)
        radval -= sp.gammaln(2*(L+lp)+2)+sp.gammaln(lp+mp+1)+sp.gammaln(lp-mp+1)
        radval -= sp.gammaln(L+mp+1)+sp.gammaln(L-mp+1)
        val *= np.sqrt(np.exp(radval))
        return val
    # Test again all L, lp, m combos to LMax=10
    LMax = 10
    errlevel = 100*np.finfo(float).eps
    for L in range(LMax+1):
        for lp in range(LMax+1):
            for m in range(-lp, lp+1):
                if (np.abs(m) < L):
                    cgfac = cg.cgCoeff(L+lp, lp, 0, -m, L, -m)
                    assert(np.abs(cgval(L, lp, m) - cgfac) < errlevel)

def test_CG_Qlmb():
    """
    Check that < l l-L m 0 | L m > =
    (-1)^(l-L)/(l-L)![(l+m)!(l-m)!(2(l-L))!(2L+1)!/(2l+1)!(L+m)!(L-m)!]^1/2
    """
    def cgval(L, l, m):
        val = (-1)**(l-L)/sp.factorial(l-L)
        radval = sp.gammaln(l+m+1)+sp.gammaln(l-m+1)+sp.gammaln(2*(l-L)+1)
        radval += sp.gammaln(2*L+2)
        radval -= sp.gammaln(2*l+2)
        radval -= sp.gammaln(L+m+1)+sp.gammaln(L-m+1)
        val *= np.sqrt(np.exp(radval))
        return val
    # Test again all L, l, m combos to LMax=10
    LMax = 10
    errlevel = 100*np.finfo(float).eps
    for l in range(LMax+1):
        for L in range(l+1):
            for m in range(-l, l+1):
                if (np.abs(m) < L):
                    cgfac = cg.cgCoeff(l, l-L, m, 0, L, m)
                    assert(np.abs(cgval(L, l, m) - cgfac) < errlevel)

def test_CG_qlm():
    """
    Check that < L-l l 0 m | L m > =
    1/(L-l)![(L+m)!(L-m)!(2(L-l))!(2l)!/(2L)!(l+m)!(l-m)!]^1/2
    """
    def cgval(L, l, m):
        val = 1/sp.factorial(L-l)
        radval = sp.gammaln(L+m+1)+sp.gammaln(L-m+1)+sp.gammaln(2*(L-l)+1)
        radval += sp.gammaln(2*l+1)
        radval -= sp.gammaln(2*L+1)
        radval -= sp.gammaln(l+m+1)+sp.gammaln(l-m+1)
        val *= np.sqrt(np.exp(radval))
        return val
    # Test again all L, l, m combos to LMax=10
    LMax = 10
    errlevel = 100*np.finfo(float).eps
    for L in range(LMax+1):
        for l in range(L+1):
            for m in range(-l, l+1):
                if (np.abs(m) < L):
                    cgfac = cg.cgCoeff(L-l, l, 0, m, L, m)
                    assert(np.abs(cgval(L, l, m) - cgfac) < errlevel)
