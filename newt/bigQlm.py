# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:11:54 2020

@author: John Greendeer Lee
"""
import numpy as np
import scipy.special as sp


def sphere(LMax, dens, R, x, y, z):
    """
    A sphere behaves identically to a point mass. Given the density and radius,
    we can get the mass of our point and evaluate the outer moments at the
    given location at (x, y, z).

    Inputs
    ------
    LMax : int
        Maximum order of inner multipole moments.
    rho : float
        Density in kg/m^3
    R : float
        Radius of sphere
    x : float
        x-position of sphere
    y : float
        y-position of sphere
    z : float
        z-position of sphere

    Returns
    -------
    Qlmb : ndarray
        (LMax+1)x(2LMax+1) complex array of outer moments
    """
    M = dens*(4*np.pi*R**3)/3
    Qlmsb = np.zeros([LMax+1, 2*LMax+1], dtype='complex')
    d = np.sqrt(x**2 + y**2 + z**2)
    if (d == 0):
        print('Outer multipole moments cannot be evaluated at the origin.')
        return Qlmsb
    theta = np.arccos(z/d)
    phi = np.arctan2(y, x) % (2*np.pi)
    for n in range(LMax+1):
        rl1 = d**(n+1)
        for m in range(n+1):
            Qlm = M*sp.sph_harm(m, n, phi, theta)/rl1
            Qlmsb[n, LMax+m] = Qlm

    # Moments always satisfy Q(l, -m) = (-1)^m Q(l, m)*
    ms = np.arange(-LMax, LMax+1)
    fac = (-1)**(np.abs(ms))
    Qlmsb += np.conj(np.fliplr(Qlmsb))*fac
    Qlmsb[:, LMax] /= 2
    return Qlmsb


def annulus(LMax, dens, H, r0, r1, phic, phih):
    """
    Calculates the first five outer moments of an annulus explicity calculated
    with Mathematica. The annular section has an axis of symmetry along zhat
    and extends vertically above the xy-plane by height H. Phic and phih are
    defined to match qlm.annulus inputs as the average angle it faces and the
    half-subtended angle. Values are only known up to LMax=5. The density is
    given by rho. Both the inner and outer radii (r0, r1) must be positive.

    Inputs
    ------
    LMax : int
        Maximum order of inner multipole moments. Only known to LMax=5.
    rho : float
        Density in kg/m^3
    H : float
        Total height along z axis above the xy-plane if H>0.
    Ri : float
        Inner radius of annulus
    Ro : float
        Outer radius of annulus
    phic : float
        Central angle of annular section
    phih : float
        Half angular width of annular section


    Returns
    -------
    Qlmb : ndarray
        (LMax+1)x(2LMax+1) complex array of outer moments
    """
    if LMax < 5:
        L = 5
    else:
        L = LMax
    Qlmb = np.zeros([L+1, 2*L+1], dtype='complex')
    phih = phih % (2*np.pi)
    if (H == 0) or (r1 < r0) or (r1 < 0) or (r0 < 0) or (phih == 0) or (phih > np.pi):
        return Qlmb
    d1 = np.sqrt(H**2 + r1**2)
    d0 = np.sqrt(H**2 + r0**2)
    sH = np.sign(H)
    H = np.abs(H)
    Q00 = H*(d1-d0) + r1**2*np.log((H+d1)/r1) - r0**2*np.log((H+d0)/r0)
    Qlmb[0, L] = dens*phih*Q00/(2*np.sqrt(np.pi))
    Q10 = r1-r0-d1+d0
    Qlmb[1, L] = sH*dens*np.sqrt(3/np.pi)*phih*Q10
    Q11 = np.log((r1+d1)/(r0+d0))
    Qlmb[1, L+1] = -dens*H*np.sqrt(3/(2*np.pi))*Q11*np.sin(phih)
    Q22 = 1/d1 - 1/d0
    Qlmb[2, L] = dens*np.sqrt(5/(4*np.pi))*phih*Q22
    Q21 = r1/d1 - r0/d0 + np.log((r1*(r0+d0))/(r0*(r1+d1)))
    Qlmb[2, L+1] = -sH*dens*np.sqrt(5/(6*np.pi))*Q21*np.sin(phih)
    Q22 = H*(1/d0-1/d1) - 2*np.log((r0*(H+d1))/(r1*(H+d0)))
    Qlmb[2, L+2] = dens*np.sqrt(5/(6*np.pi))/2*Q22*np.sin(2*phih)/2
    Q30 = 1/r1 - 1/r0 - r1**2/d1**3 + r0**2/d0**3
    Qlmb[3, L] = sH*dens*np.sqrt(7/np.pi)/6*Q30*phih
    Q31 = (r1**3/d1**3 - r0**3/d0**3)/H
    Qlmb[3, L+1] = dens*np.sqrt(7/(3*np.pi))/4*Q31*np.sin(phih)
    Q32 = 3*(1/r1 - 1/r0) - (3*r1**2+2*H**2)/d1**3 + (3*r0**2+2*H**2)/d0**3
    Qlmb[3, L+2] = -sH*dens*np.sqrt(7/(30*np.pi))/2*Q32*np.sin(2*phih)/2
    Q33 = (8*H**4 + 12*H**2*r1**2 + 3*r1**4)/(r1*d1**3)
    Q33 -= (8*H**4 + 12*H**2*r0**2 + 3*r0**4)/(r0*d0**3)
    Qlmb[3, L+3] = dens*np.sqrt(7/(5*np.pi))/(12*H)*Q33*np.sin(3*phih)/3
    Q40 = H*(r0**2/d0**5 - r1**2/d1**5)
    Qlmb[4, L] = dens*(3/(8*np.sqrt(np.pi)))*Q40*phih
    Q41 = 3/r0**2 - (8*H**2+2*r0**2)*r0**3/(H**2*d0**5)
    Q41 -= 3/r1**2 - (8*H**2+2*r1**2)*r1**3/(H**2*d1**5)
    Qlmb[4, L+1] = sH*dens/(8*np.sqrt(5*np.pi))*Q41*np.sin(phih)
    Q42 = H*((2*H**2+5*r1**2)/d1**5 - (2*H**2+5*r0**2)/d0**5)
    Qlmb[4, L+2] = dens/(4*np.sqrt(10*np.pi))*Q42*np.sin(2*phih)/2
    Q43 = 5*(1/r1**2 - 1/r0**2) + 2*(r1**5/d1**5 - r0**5/d0**5)/H**2
    Qlmb[4, L+3] = sH*dens*3/(8*np.sqrt(35*np.pi))*Q43*np.sin(3*phih)/3
    Q44 = H*(24*H**4 + 56*H**2*r0**2 + 35*r0**4)/(r0**2*d0**5)
    Q44 -= H*(24*H**4 + 56*H**2*r1**2 + 35*r1**4)/(r1**2*d1**5)
    Qlmb[4, L+4] = dens/(8*np.sqrt(70*np.pi))*Q44*np.sin(4*phih)/4
    Q50 = 1/r0**3 + r0**2*(4*H**2-r0**2)/d0**7
    Q50 -= 1/r1**3 + r1**2*(4*H**2-r1**2)/d1**7
    Qlmb[5, L] = sH*dens*np.sqrt(11/np.pi)/40*Q50*phih
    Q51 = r1**3*(20*H**4+7*H**2*r1**2+2*r1**4)/d1**7
    Q51 -= r0**3*(20*H**4+7*H**2*r0**2+2*r0**4)/d0**7
    Qlmb[5, L+1] = dens*np.sqrt(11/(30*np.pi))/(24*H**3)*Q51*np.sin(phih)
    Q52 = 5/r1**3 + (4*H**4+14*H**2*r1**2-5*r1**4)/d1**7
    Q52 -= 5/r0**3 + (4*H**4+14*H**2*r0**2-5*r0**4)/d0**7
    Qlmb[5, L+2] = sH*dens*np.sqrt(11/(210*np.pi))/12*Q52*np.sin(2*phih)/2
    Q53 = r1**5*(7*H**2+r1**2)/d1**7 - r0**5*(7*H**2+r0**2)/d0**7
    Qlmb[5, L+3] = dens*np.sqrt(11/(35*np.pi))/(16*H**3)*Q53*np.sin(3*phih)/3
    Q54 = 35/r0**3 - (8*H**4+28*H**2*r0**2+35*r0**4)/d0**7
    Q54 -= 35/r1**3 - (8*H**4+28*H**2*r1**2+35*r1**4)/d1**7
    Qlmb[5, L+4] = sH*dens*np.sqrt(11/(70*np.pi))/72*Q54*np.sin(4*phih)/4
    Q55 = (128*H**10 + 448*H**8*r1**2 + 560*H**6*r1**4 + 280*H**4*r1**6 + 35*H**2*r1**8 + 10*r1**10)/(r1**3*d1**7)
    Q55 -= (128*H**10 + 448*H**8*r0**2 + 560*H**6*r0**4 + 280*H**4*r0**6 + 35*H**2*r0**8 + 10*r0**10)/(r0**3*d0**7)
    Qlmb[5, L+5] = dens*np.sqrt(11/(7*np.pi))/(720*H**3)*Q55*np.sin(5*phih)/5

    # Moments always satisfy Q(l, -m) = (-1)^m Q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    Qlmb += np.conj(np.fliplr(Qlmb))*mfac
    Qlmb[:, L] /= 2

    # Truncate if LMax < 5
    if LMax < 5:
        Qlmb = Qlmb[:LMax+1, L-LMax:L+LMax+1]
    return Qlmb
