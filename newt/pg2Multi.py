# -*- coding: utf-8 -*-
"""
Created on Thu Sep 08 15:01:34 2016

@author: John Greendeer Lee
"""
import numpy as np
import scipy.special as sp

BIG_G = 6.67428e-11


def qmoment(l, m, massArray):
    """
    Computes the small q(l, m) inner multipole moment of a point mass array by
    evaluating the regular solid harmonic at each point-mass position.

    Inputs
    ------
    l : int
        Multipole moment order
    m : int
        Multipole moment order, m < l
    massArray : ndarray
        Nx4 array of point masses [m, x, y, z]

    Returns
    -------
    qlm : complex
        Complex-valued inner multipole moment
    """
    r = np.sqrt(massArray[:, 1]**2 + massArray[:, 2]**2 + massArray[:, 3]**2)
    rids = np.where(r != 0)[0]
    theta = np.arccos(massArray[rids, 3]/r[rids])
    phi = np.arctan2(massArray[rids, 2], massArray[rids, 1]) % (2*np.pi)
    if l == 0:
        qlm = massArray[:, 0]/np.sqrt(4*np.pi)
    else:
        qlm = massArray[:, 0]*r**l*np.conj(sp.sph_harm(m, l, phi, theta))
    qlm = np.sum(qlm)
    return qlm


def Qmomentb(l, m, massArray):
    """
    Computes the large Q(l, m) outer multipole moment of a point mass array by
    evaluating the irregular solid harmonic at each point-mass position.

    Inputs
    ------
    l : int
        Multipole moment order
    m : int
        Multipole moment order, m <= l
    massArray : ndarray
        Nx4 array of point masses [m, x, y, z]

    Returns
    -------
    qlm : complex
        Complex-valued inner multipole moment
    """
    r = np.sqrt(massArray[:, 1]**2 + massArray[:, 2]**2 + massArray[:, 3]**2)
    if (r == 0).any():
        print('Outer multipole moments cannot be evaluated at the origin.')
        return 0
    theta = np.arccos(massArray[:, 3]/r)
    phi = np.arctan2(massArray[:, 2], massArray[:, 1]) % (2*np.pi)
    qlm = massArray[:, 0]*sp.sph_harm(m, l, phi, theta)/r**(l+1)
    qlm = np.sum(qlm)
    return qlm


def imoment(l, m, lmbd, massArray):
    """
    Computes the small i(l, m) inner Yukawa multipole moment of a point mass
    array by evaluating the spherical harmonic and modified spherical bessel
    function at each point-mass position.

    Inputs
    ------
    l : int
        Multipole moment order
    m : int
        Multipole moment order, m < l
    lmbd : float
        Yukawa length scale
    massArray : ndarray
        Nx4 array of point masses [m, x, y, z]

    Returns
    -------
    ilm : complex
        Complex-valued inner multipole moment
    """
    r = np.sqrt(massArray[:, 1]**2 + massArray[:, 2]**2 + massArray[:, 3]**2)
    rids = np.where(r != 0)[0]
    theta = np.arccos(massArray[rids, 3]/r[rids])
    phi = np.arctan2(massArray[rids, 2], massArray[rids, 1]) % (2*np.pi)
    il = sp.spherical_in(l, r/lmbd)
    if l == 0:
        ilm = massArray[:, 0]*il/np.sqrt(4*np.pi)
    else:
        ilm = massArray[:, 0]*il*np.conj(sp.sph_harm(m, l, phi, theta))
    ilm = np.sum(ilm)
    return ilm


def qmoments(l, massArray):
    """
    Computes all q(l, m) inner multipole moments of a point mass array up to a
    given maximum order, l. It does so by evaluating the regular solid harmonic
    at each point-mass position.

    Inputs
    ------
    l : int
        Maximum multipole moment order
    massArray : ndarray
        Nx4 array of point masses [m, x, y, z]

    Returns
    -------
    qlms : ndarry, complex
        Complex-valued inner multipole moments up to order l.
    """
    qlms = np.zeros([l+1, 2*l+1], dtype='complex')
    r = np.sqrt(massArray[:, 1]**2 + massArray[:, 2]**2 + massArray[:, 3]**2)
    rids = np.where(r != 0)[0]
    theta = np.arccos(massArray[rids, 3]/r[rids])
    phi = np.arctan2(massArray[rids, 2], massArray[rids, 1]) % (2*np.pi)

    # Handle q00 case separately to deal with r==0 cases
    qlm = massArray[:, 0]/np.sqrt(4*np.pi)
    qlms[0, l] = np.sum(qlm)

    for n in range(1, l+1):
        rl = r**n
        for m in range(n+1):
            qlm = np.conj(sp.sph_harm(m, n, phi[rids], theta[rids]))
            qlm *= massArray[rids, 0]*rl[rids]
            qlms[n, l+m] = np.sum(qlm)

    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-l, l+1)
    fac = (-1)**(np.abs(ms))
    qlms += np.conj(np.fliplr(qlms))*fac
    qlms[:, l] /= 2

    return qlms


def Qmomentsb(l, massArray):
    """
    Computes all Q(l, m) outer multipole moments of a point mass array up to a
    a given maximum order, l. It does so by evaluating the irregular solid
    harmonic at each point-mass position.

    Inputs
    ------
    l : int
        Maximum multipole moment order
    massArray : ndarray
        Nx4 array of point masses [m, x, y, z]

    Returns
    -------
    qlms : ndarry, complex
        Complex-valued outer multipole moments up to order l.
    """
    Qlmsb = np.zeros([l+1, 2*l+1], dtype='complex')
    ctr = 0
    r = np.sqrt(massArray[:, 1]**2 + massArray[:, 2]**2 + massArray[:, 3]**2)
    if (r == 0).any():
        print('Outer multipole moments cannot be evaluated at the origin.')
        return Qlmsb
    theta = np.arccos(massArray[:, 3]/r)
    phi = np.arctan2(massArray[:, 2], massArray[:, 1]) % (2*np.pi)
    for n in range(l+1):
        rl1 = r**(n+1)
        for m in range(n+1):
            Qlm = massArray[:, 0]*sp.sph_harm(m, n, phi, theta)/rl1
            Qlmsb[n, l+m] = np.sum(Qlm)
            print(n, m)
            ctr += 1

    # Moments always satisfy Q(l, -m) = (-1)^m Q(l, m)*
    ms = np.arange(-l, l+1)
    fac = (-1)**(np.abs(ms))
    Qlmsb += np.conj(np.fliplr(Qlmsb))*fac
    Qlmsb[:, l] /= 2
    return Qlmsb


def imoments(l, lmbd, massArray):
    """
    Computes all i(l, m) inner Yukawa multipole moments of a point mass array
    up to a given maximum order, l. It does so by evaluating the spherical
    harmonic and modified spherical bessel function at each point-mass
    position.

    Inputs
    ------
    l : int
        Maximum multipole moment order
    lmbd : float
        Yukawa length scale
    massArray : ndarray
        Nx4 array of point masses [m, x, y, z]

    Returns
    -------
    ilms : ndarry, complex
        Complex-valued inner Yukawa multipole moments up to order l.
    """
    ilms = np.zeros([l+1, 2*l+1], dtype='complex')
    r = np.sqrt(massArray[:, 1]**2 + massArray[:, 2]**2 + massArray[:, 3]**2)
    rids = np.where(r != 0)[0]
    theta = np.arccos(massArray[rids, 3]/r[rids])
    phi = np.arctan2(massArray[rids, 2], massArray[rids, 1]) % (2*np.pi)

    # Handle q00 case separately to deal with r==0 cases
    i0 = sp.spherical_in(0, r/lmbd)
    ilm = massArray[:, 0]*i0/np.sqrt(4*np.pi)
    ilms[0, l] = np.sum(ilm)

    for n in range(1, l+1):
        il = sp.spherical_in(l, r/lmbd)
        for m in range(n+1):
            ilm = np.conj(sp.sph_harm(m, n, phi[rids], theta[rids]))
            ilm *= massArray[rids, 0]*il[rids]
            ilms[n, l+m] = np.sum(ilm)

    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-l, l+1)
    fac = (-1)**(np.abs(ms))
    ilms += np.conj(np.fliplr(ilms))*fac
    ilms[:, l] /= 2

    return ilms


def torque_lm(qlm, Qlm, L=None):
    r"""
    Returns all gravitational torque_lm moments up to l=10 computed from sensor
    and source multipole moments. It assumes the sensor (interior) moments sit
    in a rotating frame of a turntable so that

    .. math::
        \bar{q_{lm}} = q_{lm}e^{-im\phi_{TT}}
    Then the torque is given by

    .. math::
        \tau = -4\pi i G \sum_{l=0}^{\infty}\frac{1}{2l+1}
        \sum_{m=-l}^{l}m\ q_{lm}Q_{lm}e^{-im\phi_{TT}}

    .. math::
        = 4\pi i G \sum_{l=0}^{\infty}\frac{1}{2l+1}\sum_{m=0}^{l}m\
        (q*_{lm}Q*_{lm}e^{im\phi_{TT}} - q_{lm}Q_{lm}e^{-im\phi_{TT}})

    Since the indices l and m are identical, we may simply do an element-wise
    multiplication and sum along rows.

    Inputs
    ------
    qlm : ndarray
        10x20 array of sensor (interior) lowest order multipole moments. The
        data should be lower triangular, with l denoting row and m/2 denoting
        real columns and m/2+1 denoting imaginary columns.
    Qlm : ndarray
        10x20 array of source (exterior) lowest order multipole moments. The
        data should be lower triangular, with l denoting row and m/2 denoting
        real columns and m/2+1 denoting imaginary columns.

    Returns
    -------
    nlm : ndarray
        10x20 array of torque multipole moments. The data should be lower
        triangular, with l denoting row and m/2 denoting cosine(m*phi) columns
        and m/2+1 denoting sine(m*phi) columns.
    """
    lqlm = len(qlm)
    lQlm = len(Qlm)
    minL = min([lqlm, lQlm])-1

    ls = np.arange(minL+1)
    lfac = 1/(2*ls+1)
    ms = np.arange(-minL, minL+1)
    nlm = 4*np.pi*BIG_G*1j*np.outer(lfac, ms)*qlm*Qlm

    nm = np.sum(nlm, 0)
    nc = nm[minL:] + nm[minL::-1]
    ns = nm[minL:] - nm[minL::-1]

    return nlm, nc, ns


def embed_qlm(qlm, LNew):
    """
    Embed or truncate the moments qlm given up to order LOld, in a space of
    moments given up to order LNew >= 0, assuming all higher order moments are
    zero. Typically, this should not be done in order to preserve accuracy.
    """
    LOld = np.shape(qlm)[0] - 1
    if LNew < 0:
        print('Order cannot be negative')
        qNew = 0
    elif LNew < LOld:
        qNew = np.zeros([LNew+1, 2*LNew+1], dtype='complex')
        qNew[:] = qlm[:LNew+1, LOld-LNew:LOld+LNew+1]
    elif LNew == LOld:
        qNew = np.copy(qlm)
    else:
        qNew = np.zeros([LNew+1, 2*LNew+1], dtype='complex')
        qNew[:LOld+1, LNew-LOld:LNew+LOld+1] = qlm
    return qNew
