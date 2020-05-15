# -*- coding: utf-8 -*-
"""
Created on Sat May  9 15:09:17 2020

@author: John Greendeer Lee
"""
import numpy as np
import scipy.special as sp

big_G = 6.67428e-11


def ytilde(L, x, y, z):
    """
    Calculates the improperly normalized solid harmonics as given in Stirling
    (2017). These can be used to calculate the force basis components at a
    position (x, y, z) out to a maximum order L.
    """
    ytil = np.zeros([L+1, 2*L+1], dtype='complex')
    r2 = x**2 + y**2 + z**2
    ytil[0, L] = 1
    for l in range(1, L+1):
        ytil[l, L+l] = -(x+1j*y)*ytil[l-1, L+l-1]/(2*l)
        if l == 1:
            ytil[l, L] = (2*l-1)*z*ytil[0, L]/(l**2)
        else:
            for m in range(l):
                ytil[l, L+m] = (2*l-1)*z*ytil[l-1, L+m] - r2*ytil[l-2, L+m]
                ytil[l, L+m] /= (l**2-m**2)
    
    # functions satisfy ytil(l, -m) = (-1)^m ytil*(l, m)
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    ytil += np.conj(np.fliplr(ytil))*mfac
    ytil[:, L] /= 2
    return ytil


def force_basis(L, x, y, z):
    """
    Calculates the force basis at a position (x, y, z) out to a maximum order
    of L.
    """
    bX = np.zeros([L+1, 2*L+1], dtype='complex')
    bY = np.zeros([L+1, 2*L+1], dtype='complex')
    bZ = np.zeros([L+1, 2*L+1], dtype='complex')

    if (x == 0) and (y == 0) and (z == 0):
        bZ[1, L] = 1
        bX[1, L+1] = -1/2
        bY[1, L+1] = 1j/2
    else:
        ytilS = np.conj(ytilde(L, x, y, z))
        r2 = x**2 + y**2 + z**2
        for l in range(1, L+1):
            bX[l, L+l] = -(ytilS[l-1, L+l-1] + bX[l-1, L+l-1]*(x-1j*y))/(2*l)
            bY[l, L+l] = (1j*ytilS[l-1, L+l-1] - bY[l-1, L+l-1]*(x-1j*y))/(2*l)
            bZ[l, L+l] = -(bZ[l-1, L+l-1]*(x-1j*y))/(2*l)
            if l == 1:
                bX[l, L] = (2*l-1)*z*bX[0, L]/(l**2)
                bY[l, L] = (2*l-1)*z*bY[0, L]/(l**2)
                bZ[l, L] = (2*l-1)*(ytilS[0, L]+z*bZ[0, L])/(l**2)
            else:
                for m in range(l):
                    l2m2 = l**2 - m**2
                    l21 = 2*l-1
                    bX[l, L+m] = (l21*z*bX[l-1, L+m])/l2m2
                    bX[l, L+m] -= (2*x*ytilS[l-2, L+m] + r2*bX[l-2, L+m])/l2m2
                    bY[l, L+m] = (l21*z*bY[l-1, L+m])/l2m2
                    bY[l, L+m] -= (2*y*ytilS[l-2, L+m] + r2*bY[l-2, L+m])/l2m2
                    bZ[l, L+m] = l21*(ytilS[l-1, L+m] + z*bZ[l-1, L+m])/l2m2
                    bZ[l, L+m] -= (2*z*ytilS[l-2, L+m] + r2*bZ[l-2, L+m])/l2m2

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
    inner multipole moments qlm centered at a position (x, y, z). The order of
    the moments should all match LMax. Based on the work of Stirling (2017).

    Inputs
    ------
    qlm : ndarray
        (L+1)x(2L+1) array of sensor (interior) lowest order multipole moments.
    Qlm : ndarray
        (L+1)x(2L+1) array of sensor (outer) lowest order multipole moments.

    Returns
    -------
    nlm : ndarray
        1x3 array of forces along x, y, z
    """
    bX, bY, bZ = force_basis(LMax, x, y, z)
    force = np.zeros(3, dtype='complex')
    fac = 4*np.pi*big_G
    for lo in range(LMax+1):
        for mo in range(-lo, lo+1):
            for l in range(lo+1, LMax+1):
                lp = l-lo
                lfac = (2*l+1)
                lofac = (2*lo+1)
                for m in range(mo-lp, mo+lp+1):
                    fac2 = fac*Qlmb[l, LMax+m]*qlm[lo, LMax+mo]/lfac
                    gamsum = sp.gammaln(l+m+1) + sp.gammaln(l-m+1)
                    gamsum -= sp.gammaln(lo+mo+1) + sp.gammaln(lo-mo+1)
                    fac2 *= np.sqrt(lfac/lofac*np.exp(gamsum))
                    force[0] += fac2*bX[l-lo, LMax+m-mo]
                    force[1] += fac2*bY[l-lo, LMax+m-mo]
                    force[2] += fac2*bZ[l-lo, LMax+m-mo]

    return force
