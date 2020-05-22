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
    CGpred = np.zeros(100)
    for k in range(100):
        j1 = rand.randint(0, 6)
        j2 = rand.randint(0, 6)
        m1 = rand.randint(-j1, j1+1)
        m2 = rand.randint(-j2, j2+1)
        CGs[k] = cg.cgCoeff(j1, j2, m1, m2, 0, 0)
        deltaj = (j1 == j2)
        deltam = (m1 == -m2)
        CGpred[k] = deltaj*deltam*(-1)**(j1-m1)/np.sqrt(2*j1+1)
    assert (abs(CGs - CGpred) < 6*np.finfo(float).eps).all()


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
