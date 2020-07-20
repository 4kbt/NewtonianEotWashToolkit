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


def cart_integrand_qlm_zyx(z, y, x, l, m):
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
    """
    Numerically integrates for the inner moments of an annulus using tplquad.
    The annular section has an axis of symmetry along zhat and extends
    vertically from h1 to h2, h2 > h1. Phih is defined to match qlm.annulus
    as the half-subtended angle so that the annular section subtends an angle
    of 2*phih, symmetric about the x-axis. The density is given by rho.
    Both the inner and outer radii (IR, OR) must be positive with IR < OR.

    Inputs
    ------
    LMax : int
        Maximum order of inner multipole moments.
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
    qlm : ndarray
        (LMax+1)x(2LMax+1) complex array of inner moments
    """
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (h1 > h2) or (OR < IR) or (phih == 0) or (OR <= 0):
        return qlm
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


def cone_mom(L, dens, phih, IR, OR, H):
    """
    Numerically integrates for the inner moments of a section of a cone using
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
        Maximum order of inner multipole moments.
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
    qlm : ndarray
        (LMax+1)x(2LMax+1) complex array of inner moments
    """
    def cone_z(theta, r):
        """Boundary integral for z coordinate of cone"""
        return H*(OR-r)/(OR-IR)
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (H <= 0) or (OR < IR) or (phih == 0) or (OR <= 0):
        return qlm
    for l in range(L+1):
        for m in range(l+1):
            qlm[l, m+L], err = intg.tplquad(cyl_integrand_qlm, -phih, phih, IR,
                                            OR, 0, cone_z, args=(l, m))
            print(l, m, err)

    # Scale by density
    qlm *= dens
    # Moments always satisfy Q(l, -m) = (-1)^m Q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def trap_mom(L, dens, phih, IR, OR, h1, h2):
    """
    Numerically integrates for the inner moments of a trapezoid using tplquad.
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
        Maximum order of inner multipole moments.
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
    qlm : ndarray
        (LMax+1)x(2LMax+1) complex array of inner moments
    """
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (h1 > h2) or (OR < IR) or (phih == 0) or (OR <= 0):
        return qlm
    # Make sure the angle makes sense
    if (phih >= np.pi/2):
        return qlm

    def trap_or(theta):
        """Boundary integral for r coordinate of trapezoid"""
        return OR*np.cos(phih)/np.cos(theta)

    def trap_ir(theta):
        """Boundary integral for r coordinate of trapezoid"""
        return IR*np.cos(phih)/np.cos(theta)

    for l in range(L+1):
        for m in range(l+1):
            qlm[l, m+L], err = intg.tplquad(cyl_integrand_qlm, -phih, phih,
                                            trap_ir, trap_or, h1, h2,
                                            args=(l, m))
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
    """
    Numerically integrates for the inner moments of a Steinmetz solid using
    tplquad. The shape consists of the volume that would be removed by drilling
    a hole of radius r into a cylinder of radius R, r <= R. The symmetry axis
    of the hole is along zhat, and the cylinder has its symmetry axis along
    xhat (not yhat as for qlmACH.cylhole). The The density is given by rho.

    Inputs
    ------
    LMax : int
        Maximum order of inner multipole moments
    rho : float
        Density in kg/m^3
    r : float
        Radius of smaller cylinder hole
    R : float
        Radius of larger cylinder

    Returns
    -------
    qlm : ndarray
        (LMax+1)x(2LMax+1) complex array of inner moments
    """
    def y_xp(x):
        """Boundary integral for y coordinate of trapezoid"""
        return np.sqrt(r**2-x**2)

    def y_xm(x):
        """Boundary integral for y coordinate of trapezoid"""
        return -np.sqrt(r**2-x**2)

    def z_xyp(x, y):
        """Boundary integral for z coordinate of trapezoid"""
        return np.sqrt(R**2-y**2)

    def z_xym(x, y):
        """Boundary integral for z coordinate of trapezoid"""
        return -np.sqrt(R**2-y**2)

    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (r <= 0) or (R < r):
        return qlm
    for l in range(L+1):
        for m in range(l+1):
            qlm[l, m+L], err = intg.tplquad(cart_integrand_qlm_zyx, -r, r,
                                            y_xm, y_xp, z_xym, z_xyp,
                                            args=(l, m))
            print(l, m, err)

    # Scale by density
    qlm *= dens
    # Moments always satisfy Q(l, -m) = (-1)^m Q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def platehole(L, dens, t, r, theta):
    """
    Numerically integrates for the inner moments of a tilted cylinder using
    tplquad. The shape consists of the volume that would be removed by drilling
    a hole of radius r into a plate of thickess, t, at angle, theta. The
    symmetry axis of the hole is along z, and the plate is tilted by theta in
    the xz-plane, theta defined from z-hat. The density is given by rho.

    Inputs
    ------
    LMax : int
        Maximum order of inner multipole moments
    rho : float
        Density in kg/m^3
    t : float
        thickness of plate
    r : float
        Radius of cylinder hole
    theta : float
        Angle of plate tilt in xz-plane, defined relative to z-hat

    Returns
    -------
    qlm : ndarray
        (LMax+1)x(2LMax+1) complex array of inner moments
    """
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (r <= 0) or (theta >= np.pi/2) or (t <= 0):
        return qlm
    tanth = np.tan(theta)
    h = t/np.cos(theta)

    def y_xp(x):
        """Boundary integral for y coordinate of platehole"""
        return np.sqrt(r**2-x**2)

    def y_xm(x):
        """Boundary integral for y coordinate of platehole"""
        return -np.sqrt(r**2-x**2)

    def z_xyp(x, y):
        """Boundary integral for z coordinate of platehole"""
        return x*tanth + h/2

    def z_xym(x, y):
        """Boundary integral for z coordinate of platehole"""
        return x*tanth - h/2

    for l in range(L+1):
        for m in range(l+1):
            qlm[l, m+L], err = intg.tplquad(cart_integrand_qlm_zyx, -r, r,
                                            y_xm, y_xp, z_xym, z_xyp,
                                            args=(l, m))
            print(l, m, err)

    # Scale by density
    qlm *= dens
    # Moments always satisfy Q(l, -m) = (-1)^m Q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm
