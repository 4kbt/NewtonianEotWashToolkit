# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:45:28 2020

@author: John Greendeer Lee
"""
import numpy as np
import scipy.special as sp
import clebschGordan as cg


def translate_qlm(qlm, rPrime, LMax=10):
    r"""
    Takes in a 10x21 q_lm interior set of moments up to l=10 and returns the
    10x20 q_LM interior moments up to l=10 translated by the vector rPrime.

    Inputs
    ------
    qlm : ndarray
        (l+1)x(2l+1) array of sensor (interior) lowest order multipole moments.
    rPrime : ndarray
        [x,y,z] translation from the origin where q_lm components were computed
    LMax : int
        Maximum order of new translated multipole moments. If LMax > l of the
        untranslated moments, qlm, the results may be inaccurate.

    Returns
    -------
    qLM : ndarray
        (l+1)x(2l+1) array of sensor (interior) lowest order multipole moments.
        Moments are for the translated moments by rPrime.

    References
    ----------
    """
    rP = np.sqrt(rPrime[0]**2+rPrime[1]**2+rPrime[2]**2)
    phiP = np.arctan2(rPrime[1], rPrime[0])
    thetaP = np.arccos(rPrime[2]/rP)
    # Restrict LMax to size of qlm for now
    Lqlm = np.shape(qlm)[0] - 1
    LMax = Lqlm

    # Conjugate spherical harmonics
    ylmS = np.zeros([LMax+1, 2*LMax+1], dtype='complex')
    qLM = np.zeros([LMax+1, 2*LMax+1], dtype='complex')

    for l in range(LMax+1):
        ms = np.arange(-l, l+1)
        sphHarmL = np.conj(sp.sph_harm(ms, l, phiP, thetaP))
        ylmS[l, LMax-l:LMax+l+1] = sphHarmL

    for L in range(LMax+1):
        for M in range(L+1):
            for l in range(L+1):
                lP = L-l
                rPlP = rP**lP
                fac = np.sqrt(4.*np.pi*
                        np.exp(sp.gammaln(2*L+2)-sp.gammaln(2*lP+2)-sp.gammaln(2*l+2)))
                fac *= rPlP
                for m in range(-l, l+1):
                    mP = M - m
                    if abs(mP) <= lP:
                        cgFac = cg.cgCoeff(lP, l, mP, m, L, M)
                        qLM[L, LMax+M] += fac*cgFac*ylmS[lP, LMax+mP]*qlm[l, LMax+m]

    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-LMax, LMax+1)
    fac = (-1)**(np.abs(ms))
    qLM += np.conj(np.fliplr(qLM))*fac
    qLM[:, LMax] /= 2
    return qLM


def translate_Qlmb(Qlm, rPrime, LMax=10):
    r"""
    Takes in a 10x21 q_lm interior set of moments up to l=10 and returns the
    10x20 q_LM interior moments up to l=10 translated by the vector rPrime.

    Inputs
    ------
    qlm : ndarray
        (l+1)x(2l+1) array of sensor (interior) lowest order multipole moments.
    rPrime : ndarray
        [x,y,z] translation from the origin where q_lm components were computed
    LMax : int
        Maximum order of new translated multipole moments. If LMax > l of the
        untranslated moments, qlm, the results may be inaccurate.

    Returns
    -------
    qLM : ndarray
        (l+1)x(2l+1) array of sensor (interior) lowest order multipole moments.
        Moments are for the translated moments by rPrime.

    References
    ----------
    """
    rP = np.sqrt(rPrime[0]**2+rPrime[1]**2+rPrime[2]**2)
    phiP = np.arctan2(rPrime[1], rPrime[0])
    thetaP = np.arccos(rPrime[2]/rP)
    # Restrict LMax to size of qlm for now
    LQlm = np.shape(Qlm)[0] - 1
    LMax = LQlm

    # Conjugate spherical harmonics
    ylmS = np.zeros([LMax+1, 2*LMax+1], dtype='complex')
    QLM = np.zeros([LMax+1, 2*LMax+1], dtype='complex')

    for l in range(LMax+1):
        ms = np.arange(-l, l+1)
        sphHarmL = sp.sph_harm(ms, l, phiP, thetaP)
        ylmS[l, LMax-l:LMax+l+1] = sphHarmL

    for L in range(LMax+1):
        for M in range(L+1):
            for l in range(LMax-L+1):
                lP = L+l
                rPlP = rP**lP
                fac = np.sqrt(4.*np.pi*
                        np.exp(sp.gammaln(2*l+1)-sp.gammaln(2*lP+2)-sp.gammaln(2*L+1)))
                fac *= rPlP
                for m in range(-l, l+1):
                    mP = M - m
                    if (abs(mP) <= lP) and (L <= LMax):
                        cgFac = cg.cgCoeff(lP, l, mP, m, L, M)
                        QLM[L, LMax-M] += fac*cgFac*ylmS[lP, LMax+mP]*Qlm[l, LMax+m]

    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-LMax, LMax+1)
    fac = (-1)**(np.abs(ms))
    QLM += np.conj(np.fliplr(QLM))*fac
    QLM[:, LMax] /= 2
    return QLM


def translate_q2Q(qlm, rPrime, LMax=10):
    r"""
    Takes in a 10x21 q_lm interior set of moments up to l=10 and returns the
    10x20 Q_LM outer moments up to l=10 translated by the vector rPrime.

    Inputs
    ------
    qlm : ndarray
        (l+1)x(2l+1) array of sensor (interior) lowest order multipole moments.
    rPrime : ndarray
        [x,y,z] translation from the origin where q_lm components were computed
    LMax : int
        Maximum order of new translated multipole moments. If LMax > l of the
        untranslated moments, qlm, the results may be inaccurate.

    Returns
    -------
    qLM : ndarray
        (l+1)x(2l+1) array of sensor (interior) lowest order multipole moments.
        Moments are for the translated moments by rPrime.

    References
    ----------
    """
    rP = np.sqrt(rPrime[0]**2+rPrime[1]**2+rPrime[2]**2)
    phiP = np.arctan2(rPrime[1], rPrime[0])
    thetaP = np.arccos(rPrime[2]/rP)
    # Restrict LMax to size of qlm for now
    Lqlm = np.shape(qlm)[0] - 1
    LMax = Lqlm

    # Conjugate spherical harmonics
    ylmS = np.zeros([LMax+1, 2*LMax+1], dtype='complex')
    QLM = np.zeros([LMax+1, 2*LMax+1], dtype='complex')

    for l in range(LMax+1):
        ms = np.arange(-l, l+1)
        sphHarmL = sp.sph_harm(ms, l, phiP, thetaP)
        ylmS[l, LMax-l:LMax+l+1] = sphHarmL

    for L in range(LMax+1):
        for M in range(L+1):
            for l in range(LMax-L+1):
                lP = L+l
                rPlP1 = rP**(lP+1)
                fac = np.sqrt(4.*np.pi*
                        np.exp(sp.gammaln(2*l+1)-sp.gammaln(2*lP+2)-sp.gammaln(2*L+1)))
                fac /= rPlP1
                for m in range(-l, l+1):
                    mP = M - m
                    if (abs(mP) <= lP) and (lP <= LMax):
                        #cgFac = cg.cgCoeff(lP, l, -mP, m, L, M)*(-1)**mP
                        cgFac = cg.cgCoeff(lP, l, mP, m, L, M)
                        QLM[L, LMax+M] += fac*cgFac*ylmS[lP, LMax+mP]*qlm[l, LMax+m]

    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-LMax, LMax+1)
    fac = (-1)**(np.abs(ms))
    QLM += np.conj(np.fliplr(QLM))*fac
    QLM[:, LMax] /= 2
    return QLM
