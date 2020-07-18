# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:09:45 2020

@author: John Greendeer Lee
"""

import numpy as np
import scipy.special as sp
import scipy.integrate as intg


def cyl_integrand_Qlmb(z, r, theta, l, m):
    """
    Integrand for cylindrical expansion of outer moments including volume
    element (r). The order of the arguments implies that z=z(r, theta) and
    r=r(theta) for integration limits using scipy.integrate.tplquad.
    """
    gfac = np.exp(sp.gammaln(l+m+1) + sp.gammaln(l-m+1))
    fac = (-1)**m*np.sqrt((2*l+1)*gfac/(4*np.pi))*np.exp(1j*m*theta)
    fac /= (r**2 + z**2)**((2*l+1)/2)
    shi = 0
    for k in range((l-m)//2+1):
        m2k = m+2*k
        gamfac = sp.gammaln(m+k+1) + sp.gammaln(k+1) + sp.gammaln(l-m2k+1)
        shifac = (-1)**k*r**(m2k)*z**(l-m2k)/(2**(m2k)*np.exp(gamfac))
        shi += shifac
    # *r because cylindrical coordinates
    shi *= fac*r
    return np.real(shi)


def annulus(L, dens, h1, h2, IR, OR, phih):
    """
    Numerically integrates for the outer moments of an annulus using tplquad.
    The annular section has an axis of symmetry along zhat and extends
    vertically from h1 to h2, h2 > h1. Phih is defined to match qlm.annulus
    as the half-subtended angle so that the annular section subtends an angle
    of 2*phih, symmetric about the x-axis. The density is given by rho.
    Both the inner and outer radii (IR, OR) must be positive with IR < OR.

    Inputs
    ------
    LMax : int
        Maximum order of outer multipole moments.
    rho : float
        Density in kg/m^3
    h1 : float
        Starting z-location vertically along z-axis.
    h2 : float
        Ending z-location vertically along z-axis.
    IR : float
        Inner radius of annulus
    OR : float
        Outer radius of annulus
    phih : float
        Half angular width of annular section


    Returns
    -------
    Qlmb : ndarray
        (LMax+1)x(2LMax+1) complex array of outer moments
    """
    Qlmb = np.zeros([L+1, 2*L+1], dtype='complex')
    if (h1 > h2) or (OR < IR) or (phih == 0) or (OR <= 0):
        return Qlmb
    # Make sure this solid doesn't reach the origin
    if (h1 <= 0 <= h2) and (IR == 0):
        return Qlmb
    for l in range(L+1):
        for m in range(l+1):
            Qlmb[l, m+L], err = intg.tplquad(cyl_integrand_Qlmb, -phih, phih,
                                             IR, OR, h1, h2, args=(l, m))
            print(l, m, err)

    # Scale by density
    Qlmb *= dens
    # Moments always satisfy Q(l, -m) = (-1)^m Q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    Qlmb += np.conj(np.fliplr(Qlmb))*mfac
    Qlmb[:, L] /= 2
    return Qlmb


def outer_cone(L, dens, H, IR, OR, phih):
    """
    Numerically integrates for the outer moments of a section of a cone using
    tplquad. The cone section has an axis of symmetry along zhat, extending
    vertically above the xy-plane to a height H. The cone is defined so that at
    the outer radius OR, the height is 0, and at the inner radius IR, the
    height is H. Phih is defined to match qlm.annulus as the half-subtended
    angle so that the annular section subtends an angle of 2*phih, symmetric
    about the x-axis. The density is given by rho. Both the inner and outer
    radii (IR, OR) must be positive with IR < OR.

    Inputs
    ------
    LMax : int
        Maximum order of outer multipole moments.
    rho : float
        Density in kg/m^3
    H : float
        Height of the cone section above the xy-plane
    IR : float
        Inner radius of cone
    OR : float
        Outer radius of cone
    phih : float
        Half angular width of cone section


    Returns
    -------
    Qlmb : ndarray
        (LMax+1)x(2LMax+1) complex array of outer moments
    """
    Qlmb = np.zeros([L+1, 2*L+1], dtype='complex')
    if (H <= 0) or (OR < IR) or (phih == 0) or (OR <= 0):
        return Qlmb

    def cone_z(theta, r):
        """Boundary integral for z coordinate of cone"""
        return H*(OR-r)/(OR-IR)

    for l in range(L+1):
        for m in range(l+1):
            Qlmb[l, m+L], err = intg.tplquad(cyl_integrand_Qlmb, -phih, phih,
                                             IR, OR, 0, cone_z, args=(l, m))
            print(l, m, err)

    # Scale by density
    Qlmb *= dens
    # Moments always satisfy Q(l, -m) = (-1)^m Q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    Qlmb += np.conj(np.fliplr(Qlmb))*mfac
    Qlmb[:, L] /= 2
    return Qlmb


def trapezoid(L, dens, h1, h2, IR, OR, phih):
    """
    Numerically integrates for the outer moments of a trapezoid using tplquad.
    The trapezoidal section has an axis of symmetry along zhat and extends
    vertically from h1 to h2, h2 > h1. Phih is defined to match qlm.annulus
    as the half-subtended angle so that the trapezoidal section subtends an
    angle of 2*phih, symmetric about the x-axis. The density is given by rho.
    Both the inner and outer radii (IR, OR) must be positive with IR < OR,
    where the radii are defined as the circle that intersects the corners of
    the trapezoid at +/- phih.

    Inputs
    ------
    LMax : int
        Maximum order of outer multipole moments.
    rho : float
        Density in kg/m^3
    h1 : float
        Starting z-location vertically along z-axis.
    h2 : float
        Ending z-location vertically along z-axis.
    IR : float
        Inner radius of trapezoid
    OR : float
        Outer radius of trapezoid
    phih : float
        Half angular width of trapezoidal section


    Returns
    -------
    Qlmb : ndarray
        (LMax+1)x(2LMax+1) complex array of outer moments
    """
    Qlmb = np.zeros([L+1, 2*L+1], dtype='complex')
    if (h1 > h2) or (OR < IR) or (phih == 0) or (OR <= 0):
        return Qlmb
    # Make sure this solid doesn't reach the origin
    if ((h1 <= 0 <= h2) and (IR == 0)) or (phih >= np.pi/2):
        return Qlmb

    def trap_or(theta):
        """Boundary integral for r coordinate of trapezoid"""
        return OR*np.cos(phih)/np.cos(theta)

    def trap_ir(theta):
        """Boundary integral for r coordinate of trapezoid"""
        return IR*np.cos(phih)/np.cos(theta)

    for l in range(L+1):
        for m in range(l+1):
            Qlmb[l, m+L], err = intg.tplquad(cyl_integrand_Qlmb, -phih, phih,
                                             trap_ir, trap_or, h1, h2,
                                             args=(l, m))
            print(l, m, err)

    # Scale by density
    Qlmb *= dens
    # Moments always satisfy Q(l, -m) = (-1)^m Q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    Qlmb += np.conj(np.fliplr(Qlmb))*mfac
    Qlmb[:, L] /= 2
    return Qlmb
