# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 10:46:34 2020

@author: John Greendeer Lee
"""
import numpy as np
import scipy.special as sp


def cgCoeff(j1, j2, m1, m2, J, M):
    """
    Calculate a specific Clebsch-Gordan coefficient based on the formula given
    on Wikipedia, < j1 j2; m1 m2 | J M >. This is useful for the calculation of
    translated multipole coefficients. At some point it may be useful to use a
    recursive calculation to speed this up.

    Inputs
    ------
    j1 : int
        Azimuthal quantum number of first particle
    j2 : int
        Azimuthal quantum number of second particle
    m1 : int
        Magnetic quantum number of first particle
    m2 : int
        Magnetic quantum number of second particle
    J : int
        Combined azimuthal quantum number
    M : int
        Combined magnetic quantum number

    Returns
    -------
    CG : float
        Clebsch-Gordan coefficient < j1 j2; m1 m2 | J M >.

    Reference
    ---------
    *https://en.wikipedia.org/wiki/Table_of_Clebsch%E2%80%93Gordan_coefficients
    """
    # Make sure the M = m1 + m2
    if (j1 < abs(m1)) or (j2 < abs(m2)) or (J < abs(M)):
        print('Invalid (j1, m1), (j2, m2), or (J, M) pair')
        return 0
    # Make sure the combined angular momentum makes sense, J <= j1+j2
    if (J > (j1+j2)) or (J < abs(j1-j2)):
        print('Invalid combined angular momentum: J > j1+j2 or J < |j1-j2|')
        return 0
    delta = (M == (m1 + m2))
    # If we picked a negative M, then use symmetry property
    if M < 0:
        delta *= (-1)**(J-j1-j2)
        M = -M
        m1 = -m1
        m2 = -m2

    # kmin and kmax specify valid range of summatiion index for factorials
    kmin = max([0, j2-m1-J, j1+m2-J])
    kmax = min([j1+j2-J, j1-m1, j2+m2])
    kmin, kmax = int(kmin), int(kmax)
    fac1 = sp.gammaln(J+j1-j2+1) + sp.gammaln(J-j1+j2+1)
    fac1 += sp.gammaln(j1+j2-J+1) - sp.gammaln(j1+j2+J+2)
    fac1 = np.sqrt((2*J+1)*np.exp(fac1))
    fac2 = sp.gammaln(J+M+1) + sp.gammaln(J-M+1)
    fac2 += sp.gammaln(j1-m1+1) + sp.gammaln(j1+m1+1)
    fac2 += sp.gammaln(j2-m2+1) + sp.gammaln(j2+m2+1)
    fac2 = np.sqrt(np.exp(fac2))
    sumCG = 0
    for k in range(kmin, kmax+1):
        denFac = sp.gammaln(k+1) + sp.gammaln(j1+j2-J-k+1)
        denFac += sp.gammaln(j1-m1-k+1) + sp.gammaln(j2+m2-k+1)
        denFac += sp.gammaln(J-j2+m1+k+1) + sp.gammaln(J-j1-m2+k+1)
        denFac = np.exp(denFac)
        sumCG += (-1)**k/denFac
    CG = delta*fac1*fac2*sumCG
    return CG


def cgCoeff2(j1, j2, m1, m2, J, M):
    """
    Calculate a specific Clebsch-Gordan coefficient based on the formula given
    in Wei(1999), < j1 j2; m1 m2 | J M >. This is useful for the calculation of
    translated multipole coefficients. This method seems slightly slower than
    the version implemented with log(gamma), cgCoeff.

    Inputs
    ------
    j1 : int
        Azimuthal quantum number of first particle
    j2 : int
        Azimuthal quantum number of second particle
    m1 : int
        Magnetic quantum number of first particle
    m2 : int
        Magnetic quantum number of second particle
    J : int
        Combined azimuthal quantum number
    M : int
        Combined magnetic quantum number

    Returns
    -------
    CG : float
        Clebsch-Gordan coefficient < j1 j2; m1 m2 | J M >.

    Reference
    ---------
    *https://www.sciencedirect.com/science/article/pii/S0010465599002325
    """
    # Make sure the M = m1 + m2
    if (j1 < abs(m1)) or (j2 < abs(m2)) or (J < abs(M)):
        print('Invalid (j1, m1), (j2, m2), or (J, M) pair')
        return 0
    # Make sure the combined angular momentum makes sense, J <= j1+j2
    if (J > (j1+j2)) or (J < abs(j1-j2)):
        print('Invalid combined angular momentum: J > j1+j2 or J < |j1-j2|')
        return 0
    delta = (M == (m1 + m2))
    # If we picked a negative M, then use symmetry property
    if M < 0:
        delta *= (-1)**(J-j1-j2)
        M = -M
        m1 = -m1
        m2 = -m2

    kmin = max([0, j2-J-m1, j1-J+m2])
    kmax = min([j1+j2-J, j1-m1, j2+m2])
    kmin, kmax = int(kmin), int(kmax)
    fac1 = sp.gammaln(J+j1-j2+1) + sp.gammaln(J-j1+j2+1)
    fac1 += sp.gammaln(j1+j2-J+1) - sp.gammaln(j1+j2+J+2)
    fac1 = np.sqrt((2*J+1)*np.exp(fac1))
    fac2 = sp.comb(2*j1, j1-j2+J)*sp.comb(2*j2, j1+j2-J)*sp.comb(2*J, -j1+j2+J)
    fac2 /= sp.comb(2*j1, j1+m1)*sp.comb(2*j2, j2+m2)*sp.comb(2*J, J+M)
    fac2 = np.sqrt(fac2)
    sumCG = 0
    for k in range(kmin, kmax+1):
        kfac = sp.comb(j1+j2-J, k)
        kfac *= sp.comb(j1-j2+J, j1-m1-k)
        kfac *= sp.comb(-j1+j2+J, j2+m2-k)
        sumCG += kfac*(-1)**k
    CG = delta*fac1*fac2*sumCG
    return CG
