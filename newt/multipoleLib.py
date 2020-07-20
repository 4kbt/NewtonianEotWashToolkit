# -*- coding: utf-8 -*-
"""
Created on Sat May  9 15:09:17 2020

@author: John Greendeer Lee
"""
import numpy as np
import scipy.special as sp

BIG_G = 6.67428e-11


def force_basis(L, x, y, z):
    """
    Calculates the force basis at a position (x, y, z) out to a maximum degree
    of L. It simultaneously calculates the improperly normalized regular solid
    harmonics as given recursively in Stirling (2017).
    """
    ytilC = np.zeros([L+1, 2*L+1], dtype='complex')
    bX = np.zeros([L+1, 2*L+1], dtype='complex')
    bY = np.zeros([L+1, 2*L+1], dtype='complex')
    bZ = np.zeros([L+1, 2*L+1], dtype='complex')

    if (x == 0) and (y == 0) and (z == 0):
        bZ[1, L] = 1
        bX[1, L+1] = -1/2
        bY[1, L+1] = 1j/2
    else:
        ytilC[0, L] = 1
        r2 = x**2 + y**2 + z**2
        for l in range(1, L+1):
            l21 = 2*l-1
            ytilC[l, L+l] = -(x-1j*y)*ytilC[l-1, L+l-1]/(2*l)
            bX[l, L+l] = -(ytilC[l-1, L+l-1] + bX[l-1, L+l-1]*(x-1j*y))/(2*l)
            bY[l, L+l] = (1j*ytilC[l-1, L+l-1] - bY[l-1, L+l-1]*(x-1j*y))/(2*l)
            bZ[l, L+l] = -(bZ[l-1, L+l-1]*(x-1j*y))/(2*l)
            if l == 1:
                ytilC[l, L] = l21*z*ytilC[0, L]/(l**2)
                bX[l, L] = l21*z*bX[0, L]/(l**2)
                bY[l, L] = l21*z*bY[0, L]/(l**2)
                bZ[l, L] = l21*(ytilC[0, L]+z*bZ[0, L])/(l**2)
            else:
                for m in range(l):
                    l2m2 = l**2 - m**2
                    ytilC[l, L+m] = l21*z*ytilC[l-1, L+m] - r2*ytilC[l-2, L+m]
                    ytilC[l, L+m] /= l2m2
                    bX[l, L+m] = (l21*z*bX[l-1, L+m])/l2m2
                    bX[l, L+m] -= (2*x*ytilC[l-2, L+m] + r2*bX[l-2, L+m])/l2m2
                    bY[l, L+m] = (l21*z*bY[l-1, L+m])/l2m2
                    bY[l, L+m] -= (2*y*ytilC[l-2, L+m] + r2*bY[l-2, L+m])/l2m2
                    bZ[l, L+m] = l21*(ytilC[l-1, L+m] + z*bZ[l-1, L+m])/l2m2
                    bZ[l, L+m] -= (2*z*ytilC[l-2, L+m] + r2*bZ[l-2, L+m])/l2m2

    # functions satisfy B_i(l, -m) = (-1)^m B_i*(l, m)
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    bX += np.conj(np.fliplr(bX))*mfac
    bX[:, L] /= 2
    bY += np.conj(np.fliplr(bY))*mfac
    bY[:, L] /= 2
    bZ += np.conj(np.fliplr(bZ))*mfac
    bZ[:, L] /= 2
    return bX, bY, bZ


def multipole_force(LMax, qlm, Qlmb, x, y, z):
    """
    Calculates the gravitational force from outer multipole moments Qlmb on
    inner multipole moments qlm centered at a position (x, y, z). The degree of
    the moments should all match LMax. Based on the work of Stirling (2017).

    Inputs
    ------
    qlm : ndarray
        (L+1)x(2L+1) array of sensor (interior) lowest degree multipole moments.
    Qlm : ndarray
        (L+1)x(2L+1) array of sensor (outer) lowest degree multipole moments.

    Returns
    -------
    nlm : ndarray
        1x3 array of forces along x, y, z
    """
    bX, bY, bZ = force_basis(LMax, x, y, z)
    force = np.zeros(3, dtype='complex')
    fac = 4*np.pi*BIG_G
    for lo in range(LMax+1):
        lofac = (2*lo+1)
        for mo in range(-lo, lo+1):
            gamlomo = sp.gammaln(lo+mo+1) + sp.gammaln(lo-mo+1)
            mofac = qlm[lo, LMax+mo]*fac
            for l in range(lo+1, LMax+1):
                lp = l-lo
                lfac = (2*l+1)
                lmofac = mofac/lfac*np.sqrt(lfac/lofac)
                for m in range(mo-lp, mo+lp+1):
                    gamsum = sp.gammaln(l+m+1) + sp.gammaln(l-m+1)
                    gamsum -= gamlomo
                    fac2 = Qlmb[l, LMax+m]*lmofac*np.sqrt(np.exp(gamsum))
                    force[0] += fac2*bX[l-lo, LMax+m-mo]
                    force[1] += fac2*bY[l-lo, LMax+m-mo]
                    force[2] += fac2*bZ[l-lo, LMax+m-mo]

    return force


def torque_lm(L, qlm, Qlm):
    r"""
    Returns all gravitational torque_lm moments up to L, computed from sensor
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
    qlm : ndarray, complex
        (L+1)x(2L+1) array of sensor (interior) lowest degree multipole moments
    Qlm : ndarray, complex
        (L+1)x(2L+1) array of source (exterior) lowest degree multipole moments

    Returns
    -------
    nlm : ndarray, complex
        (L+1)x(2L+1) array of torque multipole moments.
    nc : ndarray, complex
    ns : ndarray, complex
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
    Embed or truncate the moments qlm given up to degree LOld, in a space of
    moments given up to degree LNew >= 0, assuming all higher degree moments
    are zero. Typically, this should not be done in order to preserve accuracy.
    """
    LOld = np.shape(qlm)[0] - 1
    if LNew < 0:
        print('degree cannot be negative')
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
