# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:45:28 2020

@author: John Greendeer Lee
"""
import numpy as np
import scipy.special as sp
import newt.clebschGordan as cg
import newt.genCAD as gcad


def translate_qlm(qlm, rPrime, LMax):
    r"""
    Takes in an inner set of moments, q_lm, and returns the translated inner
    moments q_LM. The translation vector is rPrime.

    Inputs
    ------
    qlm : ndarray
        (l+1)x(2l+1) array of lowest order inner multipole moments.
    rPrime : ndarray
        [x,y,z] translation from the origin where q_lm components were computed

    Returns
    -------
    qLM : ndarray
        (l+1)x(2l+1) array of lowest order inner multipole moments. Moments are
        for the translated moments by rPrime. With cad option may be numpy array
        with qlm array as the first element and a madcad mesh as the second

    References
    ----------
    https://journals.aps.org/prd/abstract/10.1103/PhysRevD.55.7970
    """
    if gcad.enabled:
        qlm, mesh = qlm
        mesh = gcad.translate_mesh(mesh, rPrime)
    rP = np.sqrt(rPrime[0]**2+rPrime[1]**2+rPrime[2]**2)
    if rP == 0:
        return qlm
    phiP = np.arctan2(rPrime[1], rPrime[0])
    thetaP = np.arccos(rPrime[2]/rP)
    # Restrict LMax to size of qlm for now
    #LMax = np.shape(qlm)[0] - 1

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
                gamsum = sp.gammaln(2*L+2)-sp.gammaln(2*lP+2)-sp.gammaln(2*l+2)
                fac = np.sqrt(4.*np.pi*np.exp(gamsum))
                fac *= rPlP
                for m in range(-l, l+1):
                    mP = M - m
                    if abs(mP) <= lP:
                        cFac = fac*cg.cgCoeff(lP, l, mP, m, L, M)
                        qLM[L, LMax+M] += cFac*ylmS[lP, LMax+mP]*qlm[l, LMax+m]

    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-LMax, LMax+1)
    fac = (-1)**(np.abs(ms))
    qLM += np.conj(np.fliplr(qLM))*fac
    qLM[:, LMax] /= 2
    if not gcad.enabled:
        return qLM
    else:
        return np.array([qlm, mesh], dtype=object)


def translate_Qlmb(Qlm, rPrime):
    r"""
    Takes in an outer set of moments, Q_lm, and returns the translated outer
    moments Q_LM. The translation vector is rPrime.

    Inputs
    ------
    Qlm : ndarray
        (l+1)x(2l+1) array of lowest order outer multipole moments.
    rPrime : ndarray
        [x,y,z] translation from the origin where Q_lm components were computed

    Returns
    -------
    QLM : ndarray
        (l+1)x(2l+1) array of lowest order outer multipole moments. Moments are
        for the translated moments by rPrime.

    References
    ----------
    https://journals.aps.org/prd/abstract/10.1103/PhysRevD.55.7970
    """
    rP = np.sqrt(rPrime[0]**2+rPrime[1]**2+rPrime[2]**2)
    if rP == 0:
        return Qlm
    phiP = np.arctan2(rPrime[1], rPrime[0])
    thetaP = np.arccos(rPrime[2]/rP)
    # Restrict LMax to size of Qlm for now
    LMax = np.shape(Qlm)[0] - 1

    # Conjugate spherical harmonics
    ylmS = np.zeros([LMax+1, 2*LMax+1], dtype='complex')
    QLM = np.zeros([LMax+1, 2*LMax+1], dtype='complex')

    for l in range(LMax+1):
        ms = np.arange(-l, l+1)
        sphHarmL = sp.sph_harm(ms, l, phiP, thetaP)
        ylmS[l, LMax-l:LMax+l+1] = sphHarmL

    for L in range(LMax+1):
        for M in range(L+1):
            for l in range(L, LMax+1):
                lP = l-L
                rPlP = rP**lP
                gamsum = sp.gammaln(2*l+1)-sp.gammaln(2*lP+2)-sp.gammaln(2*L+1)
                fac = np.sqrt(4.*np.pi*np.exp(gamsum))
                fac *= rPlP
                for m in range(-l, l+1):
                    mP = M - m
                    if (abs(mP) <= lP) and (lP <= LMax):
                        cFac = fac*cg.cgCoeff(lP, l, mP, m, L, M)
                        QLM[L, LMax+M] += cFac*ylmS[lP, LMax+mP]*Qlm[l, LMax+m]

    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-LMax, LMax+1)
    fac = (-1)**(np.abs(ms))
    QLM += np.conj(np.fliplr(QLM))*fac
    QLM[:, LMax] /= 2
    return QLM


def translate_q2Q(qlm, rPrime, LMax):
    r"""
    Takes in an inner set of moments, q_lm, and returns the translated outer
    moments Q_LM. The translation vector is rPrime.

    Inputs
    ------
    qlm : ndarray
        (l+1)x(2l+1) array of lowest order inner multipole moments.
    rPrime : ndarray
        [x,y,z] translation from the origin where q_lm components were computed

    Returns
    -------
    QLM : ndarray
        (l+1)x(2l+1) array of lowest order outer multipole moments. Moments are
        for the translated moments by rPrime.

    References
    ----------
    https://journals.aps.org/prd/abstract/10.1103/PhysRevD.60.107501
    """
    if gcad.enabled:
        qlm, mesh = qlm
        mesh = gcad.translate_mesh(mesh, rPrime)
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
            for lP in range(LMax-L+1):
                l = L+lP
                rPl1 = rP**(l+1)
                gamsum = sp.gammaln(2*l+1)-sp.gammaln(2*lP+2)-sp.gammaln(2*L+1)
                fac = np.sqrt(4.*np.pi*np.exp(gamsum))
                fac /= rPl1
                for mP in range(-lP, lP+1):
                    m = M + mP
                    if (abs(m) <= l) and (l <= LMax):
                        cFac = fac*cg.cgCoeff(lP, l, -mP, m, L, M)*(-1)**mP
                        QLM[L, LMax+M] += cFac*ylmS[l, LMax+m]*qlm[lP, LMax+mP]

    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-LMax, LMax+1)
    fac = (-1)**(np.abs(ms))
    QLM += np.conj(np.fliplr(QLM))*fac
    QLM[:, LMax] /= 2
    if not gcad.enabled:
        return QLM
    else:
        return np.array([QLM, mesh], dtype=object)
