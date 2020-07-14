# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:53:06 2020

@author: John Greendeer Lee
"""
import numpy as np
import scipy.special as sp
import scipy.integrate as intg


def cyl_integrand_qlm(z, r, theta, l, m):
    """
    Integrand for cylindrical expansion of inner moments including volume
    element (r). The order of the arguments implies that z=z(r, theta) and
    r=r(theta) for integration limits using scipy.integrate.tplquad.
    """
    gfac = np.exp(sp.gammaln(l+m+1) + sp.gammaln(l-m+1))
    fac = (-1)**m*np.sqrt((2*l+1)*gfac/(4*np.pi))*np.exp(-1j*m*theta)
    shi = 0
    for k in range((l-m)//2+1):
        m2k = m+2*k
        gamfac = sp.gammaln(m+k+1) + sp.gammaln(k+1) + sp.gammaln(l-m2k+1)
        shifac = (-1)**k*r**(m2k)*z**(l-m2k)/(2**(m2k)*np.exp(gamfac))
        shi += shifac
    # *r because cylindrical coordinates
    shi *= fac*r
    return np.real(shi)


def cart_integrand_qlm_r(z, y, x, l, m):
    """
    Integrand for cartesian expansion of inner moments. The order of the
    arguments implies that z=z(x, y) and y=y(x) for integration limits using
    scipy.integrate.tplquad.
    """
    gfac = np.exp(sp.gammaln(l+m+1) + sp.gammaln(l-m+1))
    fac = (-1)**m*np.sqrt((2*l+1)*gfac/(4*np.pi))
    shi = 0
    xpy = x+1j*y
    xmy = x-1j*y
    for k in range((l-m)//2+1):
        m2k = m+2*k
        gamfac = sp.gammaln(m+k+1) + sp.gammaln(k+1) + sp.gammaln(l-m2k+1)
        shifac = (-1)**k*xpy**k*xmy**(m+k)*z**(l-m2k)/(2**(m2k)*np.exp(gamfac))
        shi += shifac
    shi *= fac
    return np.real(shi)


def cyl_mom(L, dens, phih, IR, OR, h1, h2):
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    for l in range(L+1):
        for m in range(l+1):
            qlm[l, m+L], err = intg.tplquad(cyl_integrand_qlm, -phih, phih, IR,
                                            OR, h1, h2, args=(l, m))
            print(l, m, err)

    # Scale by density
    qlm *= dens
    # Moments always satisfy Q(l, -m) = (-1)^m Q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def steinmetz(L, dens, r, R):
    def y_xp(x):
        """Boundary integral for r coordinate of trapezoid"""
        return np.sqrt(r**2-x**2)

    def y_xm(x):
        """Boundary integral for r coordinate of trapezoid"""
        return -np.sqrt(r**2-x**2)

    def z_xyp(x, y):
        """Boundary integral for r coordinate of trapezoid"""
        return np.sqrt(R**2-y**2)

    def z_xym(x, y):
        """Boundary integral for r coordinate of trapezoid"""
        return -np.sqrt(R**2-y**2)

    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    for l in range(L+1):
        for m in range(l+1):
            qlm[l, m+L], err = intg.tplquad(cart_integrand_qlm_r, -r, r, y_xm,
                                            y_xp, z_xym, z_xyp, args=(l, m))
            print(l, m, err)

    # Scale by density
    qlm *= dens
    # Moments always satisfy Q(l, -m) = (-1)^m Q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm
