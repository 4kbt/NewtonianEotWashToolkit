# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 20:09:40 2020

@author: John Greendeer Lee
"""
import numpy as np
import scipy.special as sp


def rect_prism(LMax, rho, x, y, z):
    """
    Inner multipoles of a rectangular box centered on the origin and specified
    by positive quantities x, y, and z - its dimensions along xhat, yhat, and
    zhat. Values are only known up to LMax=5. The density is given by rho.

    Inputs
    ------
    LMax : int
        Maximum order of inner multipole moments. Only known to LMax=5.
    rho : float
        Density in kg/m^3
    x : float
        Total length along x axis. Centered about x=0.
    y : float
        Total width along y axis. Centered about y=0.
    z : float
        Total height along z axis. Centered about z=0.

    Returns
    -------
    qlm : ndarray
        (LMax+1)x(2LMax+1) complex array of inner moments
    """
    if LMax < 5:
        L = 5
    else:
        L = LMax
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (x <= 0) or (y <= 0) or (z <= 0):
        return qlm
    q00 = rho*x*y*z/np.sqrt(4*np.pi)
    r2 = x**2 + y**2
    d2 = x**2 - y**2
    qlm[0, L] = q00
    qlm[2, L] = -q00*np.sqrt(5)*(r2 - 2*z**2)/24
    qlm[2, L+2] = q00*np.sqrt(5/6)*d2/8
    qlm[4, L] = q00*(9*x**4 + 10*x**2*y**2 + 9*y**4 - 40*r2*z**2 + 24*z**4)/640
    qlm[4, L+2] = -q00*d2*(3*r2-10*z**2)/(64*np.sqrt(10))
    qlm[4, L+4] = q00*np.sqrt(7/10)*(3*x**4 - 10*x**2*y**2 + 3*y**4)/128

    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2

    # Truncate if LMax < 5
    if LMax < 5:
        qlm = qlm[:LMax+1, L-LMax:L+LMax+1]
    return qlm


def annulus(LMax, rho, H, Ri, Ro, phic, phih):
    """
    Cylindrical annulus with axis of symmetry along zhat and symmetric about
    the xy-plane. Phic and phih are defined to match qlm.annulus inputs. Values
    are only known up to LMax=5. The density is given by rho.

    Inputs
    ------
    LMax : int
        Maximum order of inner multipole moments. Only known to LMax=5.
    rho : float
        Density in kg/m^3
    H : float
        Total height along z axis. Centered about z=0.
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
    qlm : ndarray
        (LMax+1)x(2LMax+1) complex array of inner moments
    """
    if LMax < 5:
        L = 5
    else:
        L = LMax
    phih = phih % (2*np.pi)
    dphi = phih*2
    dr = Ro-Ri
    r = (Ro+Ri)/2
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (Ro <= Ri) or (Ro <= 0) or (H <= 0) or (phih > np.pi):
        return qlm
    eiphi = np.exp(-1j*phic)
    sdphih = np.sin(dphi/2)
    sdphi3h = np.sin(dphi*3/2)
    qlm[0, L] = rho*H*r*dphi*dr/np.sqrt(4*np.pi)
    qlm[1, L+1] = -rho/(4*np.sqrt(6*np.pi))*eiphi
    qlm[1, L+1] *= H*dr*(dr**2 + 12*r**2)*sdphih
    qlm[2, L] = -rho*np.sqrt(5/np.pi)*H*r*dphi*dr*(3*dr**2 - 2*H**2 + 12*r**2)
    qlm[2, L] /= 48
    qlm[2, L+2] = rho*np.sqrt(15/(2*np.pi))*eiphi**2*H*r*dr
    qlm[2, L+2] *= (dr**2 + 4*r**2)*np.sin(dphi)/16
    qlm[3, L+1] = rho*np.sqrt(7/(3*np.pi))*eiphi*H*dr*sdphih/960
    qlm[3, L+1] *= (9*dr**4 - 20*dr**2*(H**2-18*r**2) - 240*r**2*(H**2-3*r**2))
    qlm[3, L+3] = -rho*np.sqrt(7/(5*np.pi))*eiphi**3*H*dr/192
    qlm[3, L+3] *= (dr**4 + 40*dr**2*r**2 + 80*r**4)*sdphi3h
    qlm[4, L] = 3*rho/np.sqrt(np.pi)*H*r*dphi*dr/1280
    q40 = 15*dr**4 - 40*dr**2*(H**2-5*r**2) + 8*(H**4-20*H**2*r**2+30*r**4)
    qlm[4, L] *= q40
    qlm[4, L+2] = -rho*np.sqrt(5/(2*np.pi))*eiphi**2*H*r*dr*np.sin(dphi)/128
    qlm[4, L+2] *= 3*dr**4 - 24*r**2*(H**2-2*r**2) + dr**2*(-6*H**2+40*r**2)
    qlm[4, L+4] = rho*np.sqrt(35/(2*np.pi))*eiphi**4*H*r*dr*np.sin(2*dphi)/512
    qlm[4, L+4] *= 3*dr**4 + 40*dr**2*r**2 + 48*r**4
    qlm[5, L+1] = -rho*np.sqrt(11/(30*np.pi))*eiphi*H*dr*sdphih/3584
    q51 = 15*dr**6 - 84*dr**4*(H**2-15*r**2)
    q51 += 672*r**2*(H**4-10*H**2*r**2+10*r**4)
    q51 += 56*dr**2*(H**4-60*H**2*r**2+150*r**4)
    qlm[5, L+1] *= q51
    qlm[5, L+3] = rho*np.sqrt(11/(35*np.pi))*eiphi**3*H*dr*sdphi3h/9216
    q53 = 15*dr**6 - 56*dr**4*H**2 + 140*dr**2*(9*dr**2-16*H**2)*r**2
    q53 += 560*(15*dr**2-8*H**2)*r**4 + 6720*r**6
    qlm[5, L+3] *= q53
    qlm[5, L+5] = -3*rho*np.sqrt(11/(7*np.pi))*eiphi**5*H*dr*np.sin(dphi*5/2)
    q55 = dr**6 + 84*dr**4*r**2 + 560*dr**2*r**4 + 448*r**6
    qlm[5, L+5] *= q55/5120

    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2

    # Truncate if LMax < 5
    if LMax < 5:
        qlm = qlm[:LMax+1, L-LMax:L+LMax+1]
    return qlm


def cone(LMax, rho, h, r1, r2):
    """
    The (truncated) cone is symmetric about zhat, and is specified by upper and
    lower radii, r1 and r2, and height h>0. The cone extends a distance h/2
    above and below the xy-plane. Complete cones have vanishing values of r1 or
    r2. Moments given out to LMax=5.

    Inputs
    ------
    LMax : int
        Maximum order of inner multipole moments. Only known to LMax=5.
    rho : float
        Density in kg/m^3
    h : float
        Total height along z axis. Centered about z=0.
    r1 : float
        Radius of lower section of cone
    r2 : float
        Radius of upper section of cone

    Returns
    -------
    qlm : ndarray
        (LMax+1)x(2LMax+1) complex array of inner moments
    """
    if LMax < 5:
        L = 5
    else:
        L = LMax
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (h <= 0):
        return qlm
    fac = rho*np.sqrt(np.pi)
    qlm[0, L] = fac*h*(r1**2 + r1*r2 + r2**2)/6
    # qlm[1, L] = fac*h**2*(r1**2 + 2*r1*r2 + 3*r2**2)/(8*np.sqrt(3))
    # Erik Shaw form of q(1,0)
    qlm[1, L] = fac*h**2*(r1-r2)*(r1+r2)/(8*np.sqrt(3))
    # q20 = 2*h**2*(r1**2 + 3*r1*r2 + 6*r2**2)
    # q20 -= 3*(r1**4 + 2*r1**3*r2 + 3*r1**2*r2**2 + r1*r2**3 + r2**4)
    q20 = h**2*(2*r1**2 + r1*r2 + 2*r2**2)
    q20 -= 3*(r1**4 + r1**3*r2 + r1**2*r2**2 + r1*r2**3 + r2**4)
    qlm[2, L] = fac*h*q20/(24*np.sqrt(5))
    # q3 = 2*h**2*(r1**2 + 4*r1*r2 + 10*r2**2)
    # q3 -= 3*(r1**4 + 2*r1**3*r2 + 3*r1**2*r2**2 + 4*r1*r2**3 + 5*r2**4)
    # qlm[3, L] = fac*h**2*q3*np.sqrt(7)/240
    q3 = (r1**2 - r2**2)*(h**2 - 2*(2*r1**2 + r1*r2 + 2*r2**2))
    qlm[3, L] = fac*h**2*q3*np.sqrt(7)/160
    # q4 = 8*h**4*(r1**2 + 5*r1*r2 + 15*r2**2)
    # q4 -= 12*h**2*(r1**4+3*r1**3*r2+6*r1**2*r2**2+10*r1*r2**3+15*r2**4)
    # q4 += 15*(r1**6 + r1**5*r2 + r1**4*r2**2 + r1**3*r2**3 + r1**2*r2**4)
    # q4 += 15*(r1*r2**5 + r2**6)
    # qlm[4, L] = fac*q4*h/560
    q4 = h**4*(3*r1**2 + r1*r2 + 3*r2**2)
    q4 -= 2*h**2*(11*r1**4+5*r1**3*r2+3*r1**2*r2**2+5*r1*r2**3+11*r2**4)
    q4 += 10*(r1**6 + r1**5*r2 + r1**4*r2**2 + r1**3*r2**3)
    q4 += 10*(r2**6 + r2**5*r1 + r2**4*r1**2)
    qlm[4, L] = fac*3*h*q4/1120
    # q5 = 8*h**4*(r1**2 + 6*r1*r2 + 21*r2**2)
    # q5 -= 12*h**2*(r1**4 + 4*r1**3*r2 + 10*r1**2*r2**2 + 20*r1*r2**3)
    # q5 -= 12*h**2*(35*r2**4)
    # q5 += 15*(r1**6 + 2*r1**5*r2 + 3*r1**4*r2**2 + 4*r1**3*r2**3)
    # q5 += 15*(5*r1**2*r2**4 + 6*r1*r2**5 + 7*r2**6)
    # qlm[5, L] = fac*h**2*q5*np.sqrt(11)/2688
    q5 = h**4 - 4*h**2*(3*r1**2 + r1*r2 + 3*r2**2)
    q5 += 5*(3*r1**4 + 2*r1**2*r2 + 4*r1**2*r2**2 + 2*r1*r2**3 + 3*r2**4)
    qlm[5, L] = fac*q5*h**2*(r1**2 - r2**2)*np.sqrt(11)/896

    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2

    # Truncate if LMax < 5
    if LMax < 5:
        qlm = qlm[:LMax+1, L-LMax:L+LMax+1]
    return qlm


def tri_prism(LMax, rho, h, d, y1, y2):
    """
    The shape has reflection symmetry about xy-plane and thickness h>0. The
    triangular faces have vertices at (x,y)=(0,0), (d,y1), and (d,y2). The
    restriction that one side be parallel to yhat is easily overcome using the
    rotational properties of the moments. Moments out to LMax=5.

    Inputs
    ------
    LMax : int
        Maximum order of inner multipole moments. Only known to LMax=5.
    rho : float
        Density in kg/m^3
    h : float
        Total height along z axis. Centered about z=0.
    d : float
        Length along x axis.
    y1 : float
        Position of first vertex along y-axis
    y2 : float
        Position of second vertex along y-axis

    Returns
    -------
    qlm : ndarray
        (LMax+1)x(2LMax+1) complex array of inner moments
    """
    if LMax < 5:
        L = 5
    else:
        L = LMax
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (h <= 0):
        return qlm
    y2p = (y1**2+y1*y2+y2**2)
    y4p = (y1**4+y1**3*y2+y1**2*y2**2+y1*y2**3+y2**4)
    y5p = (y1**5+y1**4*y2+y1**3*y2**2+y1**2*y2**3+y1*y2**4+y2**5)
    q00 = rho*h*abs(d*(y1-y2))/(4*np.sqrt(np.pi))
    qlm[0, L] = q00
    qlm[1, L+1] = q00*(-2*d + 1j*(y1+y2))/np.sqrt(6)
    qlm[2, L] = -q00*np.sqrt(5)*(3*d**2 - h**2 + y2p)/12
    q22 = 3*d**2 - y2p - 3j*d*(y1+y2)
    qlm[2, L+2] = q00*q22*np.sqrt(5/6)/4
    q31 = 36*d**3 - 18j*d**2*(y1+y2) + 1j*(y1+y2)*(10*h**2-9*(y1**2+y2**2))
    q31 -= 4*d*(5*h**2 - 3*y2p)
    qlm[3, L+1] = q00*q31*np.sqrt(7/3)/120
    q33 = ((1+1j)*d+y1-1j*y2)*((1+1j)*d-1j*y1+y2)*(2j*d+y1+y2)
    qlm[3, L+3] = q00*q33*np.sqrt(7/5)/8
    q40 = 30*d**4 + 3*h**4 - 10*h**2*y2p
    q40 += 6*(y1**4 + y1**3*y2+y1**2*y2**2 + y1*y2**3 + y2**4)
    q40 += 10*d**2*(-3*h**2+2*y2p)
    qlm[4, L] = q00*q40/80
    q42 = 20*d**4 - 15*d**2*h**2 - 20j*d**3*(y1+y2)
    q42 += 5*h**2*y2p + 5j*d*(y1+y2)*(3*h**2-2*(y1**2+y2**2)) - 4*y4p
    qlm[4, L+2] = -q00*q42/(16*np.sqrt(10))
    q44 = 5*d**4 + y1**4 + y1**3*y2 + y1**2*y2**2 + y1*y2**3 + y2**4
    q44 += -10j*d**3*(y1+y2) + 5j*d*(y1+y2)*(y1**2+y2**2)
    q44 -= 10*d**2*y2p
    qlm[4, L+4] = q00*q44*np.sqrt(7/10)/8
    q51 = 60*d**5 - 30j*d**4*(y1+y2) + 6j*d**2*(y1+y2)*(7*h**2-5*(y1**2+y2**2))
    q51 -= 4*d**3*(21*h**2-10*y2p)
    q51 += 2*d*(7*h**4-14*h**2*y2p+6*y4p)
    q51 -= 1j*(7*h**4*(y1+y2)-21*h**2*(y1**3+y1**2*y2+y1*y2**2+y2**3)+10*y5p)
    qlm[5, L+1] = -q00*q51*np.sqrt(11/30)/112
    q53 = 30*d**5 - 45j*d**4*(y1+y2)+3j*d**2*(y1+y2)*(14*h**2-5*(y1**2+y2**2))
    q53 -= 4*d**3*(7*h**2+5*y2p)
    q53 -= 1j*(y1+y2)*(7*h**2*(y1**2+y2**2)-5*(y1**4+y1**2*y2**2+y2**4))
    q53 += d*(28*h**2*y2p - 18*y4p)
    qlm[5, L+3] = q00*q53*np.sqrt(11/35)/48
    q55 = (-2*d+1j*(y1+y2))*(d**2-y1**2+y1*y2-y2**2-1j*d*(y1+y2))
    q55 *= (3*d**2-y2p-3j*d*(y1+y2))
    qlm[5, L+5] = q00*q55*np.sqrt(11/7)/16

    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2

    # Truncate if LMax < 5
    if LMax < 5:
        qlm = qlm[:LMax+1, L-LMax:L+LMax+1]
    return qlm


def tetrahedron(LMax, rho, x, y, z):
    """
    This shape consists of a tetrahedron having three mutually perpendicular
    triangular faces that meet at the origin. The fourth triangular face is
    defined by points at corrdinates x, y, and z along the xhat, yhat, and zhat
    axes respectively. Values are only known up to LMax=5. The density is given
    by rho.

    Inputs
    ------
    LMax : int
        Maximum order of inner multipole moments. Only known to LMax=5.
    rho : float
        Density in kg/m^3
    x : float
        Distance to vertex along x-axis
    y : float
        Distance to vertex along y-axis
    z : float
        Distance to vertex along z-axis

    Returns
    -------
    qlm : ndarray
        (LMax+1)x(2LMax+1) complex array of inner moments
    """
    if LMax < 5:
        L = 5
    else:
        L = LMax
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (x <= 0) or (y <= 0) or (z <= 0):
        return qlm
    q00 = rho*x*y*z/(6*np.sqrt(4*np.pi))
    qlm[0, L] = q00
    qlm[1, L] = q00*z*np.sqrt(3)/4
    qlm[1, L+1] = -q00*3*(x-1j*y)/(4*np.sqrt(6))
    qlm[2, L] = -q00*(x**2 + y**2 - 2*z**2)/(4*np.sqrt(5))
    qlm[2, L+1] = -q00*3*z*(x-1j*y)/(4*np.sqrt(30))
    qlm[2, L+2] = q00*3*(x**2 - 1j*x*y - y**2)/(4*np.sqrt(30))
    qlm[3, L] = -q00*z*(x**2 + y**2 - 2*z**2)*np.sqrt(7)/40
    qlm[3, L+1] = q00*(3*x**3-1j*x**2*y+x*y**2-3j*y**3-4*(x-1j*y)*z**2)
    qlm[3, L+1] *= np.sqrt(7/3)/80
    qlm[3, L+2] = q00*z*(x**2-1j*x*y-y**2)*np.sqrt(7/30)/8
    qlm[3, L+3] = -q00*(x**2-y**2)*(x-1j*y)*np.sqrt(7/5)/16
    qlm[4, L] = q00*(3*x**4+x**2*y**2+3*y**4-4*z**2*(x**2+y**2)+8*z**4)*3/280
    q41 = z*(3*x**3-1j*x**2*y+x*y**2-3j*y**3-4*z**2*(x-1j*y))
    qlm[4, L+1] = q00*q41*3/(112*np.sqrt(5))
    q42 = 2*x**4-1j*x**3*y-1j*x*y**3-2*y**4-2*(x**2-1j*x*y-y**2)*z**2
    qlm[4, L+2] = -q00*q42*3/(56*np.sqrt(10))
    qlm[4, L+3] = -q00*z*(x**2-y**2)*(x-1j*y)*3/(16*np.sqrt(35))
    q44 = x**4-1j*x**3*y-x**2*y**2+1j*x*y**3+y**4
    qlm[4, L+4] = q00*q44*3/(8*np.sqrt(70))
    q50 = z*(3*x**4+x**2*y**2+3*y**4-4*(x**2+y**2)*z**2+8*z**4)
    qlm[5, L] = q00*q50*np.sqrt(11)/448
    q51 = 5*x**5 - 1j*x**4*y + x**3*y**2 - 1j*x**2*y**3 + x*y**4 - 5j*y**5
    q51 += -2*(x+1j*y)*(3*x**2-4j*x*y-3*y**2)*z**2 + 8*(x-1j*y)*z**4
    qlm[5, L+1] = -q00*q51*np.sqrt(11/30)*3/448
    q52 = z*(-2*x**4+1j*x**3*y+1j*x*y**3+2*y**4+2*(x**2-1j*x*y-y**2)*z**2)
    qlm[5, L+2] = q00*q52*np.sqrt(11/210)*3/64
    q53 = 5*x**5-3j*x**4*y-x**3*y**2-1j*x**2*y**3-3*x*y**4+5j*y**5
    q53 -= 4*(x**2-y**2)*(x-1j*y)*z**2
    qlm[5, L+3] = q00*q53*np.sqrt(11/35)/128
    #q54 = z**3*(x**4-1j*x**3*y-x**2*y**2+1j*x*y**3+y**4)
    q54 = z*(x**4-1j*x**3*y-x**2*y**2+1j*x*y**3+y**4)
    qlm[5, L+4] = q00*q54*np.sqrt(11/70)*3/64
    q55 = (x-1j*y)*((x**2-y**2)**2 + x**2*y**2)
    qlm[5, L+5] = -q00*q55*np.sqrt(11/7)*3/128

    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2

    # Truncate if LMax < 5
    if LMax < 5:
        qlm = qlm[:LMax+1, L-LMax:L+LMax+1]
    return qlm


def cylhole(LMax, rho, r, R):
    """
    The shape consists of the volume that would be removed by drilling a hole
    of radius r into a cylinder of radius R. The symmetry axis of the hole is
    along zhat, and the cylinder has its symmetry axis along yhat. The moments
    requre the hypergeometric function 2F1(a, b; c; x). Moments out to LMax=5.

    Inputs
    ------
    LMax : int
        Maximum order of inner multipole moments. Only known to LMax=5.
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
    if LMax < 5:
        L = 5
    else:
        L = LMax
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (r <= 0) or (R < r):
        return qlm
    fac = rho*np.sqrt(np.pi)*r*R
    x = (r/R)**2
    Fc2 = sp.hyp2f1(-.5, .5, 2, x)
    Fc3 = sp.hyp2f1(-.5, 1.5, 3, x)
    Fc4 = sp.hyp2f1(-.5, 2.5, 4, x)
    qlm[0, L] = fac*r*Fc2
    qlm[2, L] = -fac*r*np.sqrt(5)/6*((r**2-2*R**2)*Fc2 + r**2*Fc3)
    qlm[2, L+2] = fac*r*np.sqrt(15/2)*Fc2
    q40 = (9*r**5-40*r**3*R**2+24*r*R**4)*Fc2 + (13*r**5-32*r**3*R**2)*Fc3
    q40 += 16*r**5*Fc4
    qlm[4, L] = fac/40*q40
    q42 = (6*r**2-20*R**2)*Fc2 + 2*(r**2+10*R**2)*Fc3 - 13*r**2*Fc4
    qlm[4, L+2] = fac*r**3*q42/(8*np.sqrt(10))
    qlm[4, L+4] = 3*fac*r*np.sqrt(35/2)*Fc2/2

    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2

    # Truncate if LMax < 5
    if LMax < 5:
        qlm = qlm[:LMax+1, L-LMax:L+LMax+1]
    return qlm


def platehole(LMax, rho, t, r, theta):
    """
    This shape consists of the volume that would be removed by drilling a hole
    of radius r through a parallel-sided plate of thickness t. The plate is
    centered on the xy-plane. The hole axis, which passes through the origin,
    lies in the xz-plane at an angle theta measured from zhat, where -pi/2 <
    theta < pi/2. Values are only known up to LMax=5. The density is given by
    rho.

    Inputs
    ------
    LMax : int
        Maximum order of inner multipole moments. Only known to LMax=5.
    rho : float
        Density in kg/m^3
    t : float
        Thickness of rectangular plate, centered on xy-plane
    r : float
        Radius of cylindrical hole
    theta : float
        Angle of hole relative to z axis, tilted toward x axis.

    Returns
    -------
    qlm : ndarray
        (LMax+1)x(2LMax+1) complex array of inner moments
    """
    if LMax < 5:
        L = 5
    else:
        L = LMax
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (r <= 0) or (t <= 0) or (abs(theta) > np.pi/2):
        return qlm
    s = np.tan(theta)
    q00 = rho*t*r**2*np.sqrt(np.pi*(1+s**2))/2
    qlm[0, L] = q00
    qlm[2, L] = -q00*(3*r**2*(s**2+2) + t**2*(s**2-2))*np.sqrt(5)/24
    qlm[2, L+1] = -q00*t**2*s*np.sqrt(5/6)/4
    qlm[2, L+2] = q00*s**2*(3*r**2+t**2)*np.sqrt(5/6)/8
    q40 = 10*r**4*(8+8*s**2+3*s**4) + 10*t**2*r**2*(3*s**4-8)
    q40 += t**4*(8-24*s**2+3*s**4)
    qlm[4, L] = q00*q40*3/640
    q41 = t**2*s*(5*r**2*(4+3*s**2) + t**2*(3*s**2-4))
    qlm[4, L+1] = q00*q41*3/(64*np.sqrt(5))
    q42 = s**2*(10*r**4*(s**2+2) + 10*(r*s*t)**2 + t**4*(s**2-6))
    qlm[4, L+2] = -q00*q42*3/(64*np.sqrt(10))
    qlm[4, L+3] = -q00*s**3*(t**4 + 5*r**2*t**2)*np.sqrt(7/5)*3/64
    qlm[4, L+4] = q00*s**4*(10*r**4+10*r**2*t**2+t**4)*np.sqrt(7/10)*3/128

    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2

    # Truncate if LMax < 5
    if LMax < 5:
        qlm = qlm[:LMax+1, L-LMax:L+LMax+1]
    return qlm


def pyramid(LMax, rho, h, x, y):
    """
    This shape consists of a symmetric pyramid whose base with side lengths x
    and y lies in the xy-plane centered about the origin, and whose apex is at
    (x,y,z) = (0,0,h). Values are only known up to LMax=5. The density is given
    by rho.

    Inputs
    ------
    LMax : int
        Maximum order of inner multipole moments. Only known to LMax=5.
    rho : float
        Density in kg/m^3
    h : float
        Height along z-axis
    x : float
        Total length along x-axis, centered on x=0.
    y : float
        Total length along y-axis, centered on y=0.

    Returns
    -------
    qlm : ndarray
        (LMax+1)x(2LMax+1) complex array of inner moments
    """
    if LMax < 5:
        L = 5
    else:
        L = LMax
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (x <= 0) or (y <= 0) or (h <= 0):
        return qlm
    q00 = rho*h*x*y/(6*np.sqrt(np.pi))
    qlm[0, L] = q00
    qlm[1, L] = q00*h*np.sqrt(3)/4
    qlm[2, L] = q00*(4*h**2-x**2-y**2)/(8*np.sqrt(5))
    qlm[2, L+2] = q00*(x**2-y**2)*np.sqrt(3/10)/8
    qlm[3, L] = q00*h*(4*h**2-x**2-y**2)*np.sqrt(7)/80
    qlm[3, L+2] = q00*h*(x**2-y**2)*np.sqrt(7/30)/16
    q40 = 128*h**4 + 9*x**4 + 10*x**2*y**2 + 9*y**4 - 32*h**2*(x**2+y**2)
    qlm[4, L] = q00*q40*3/4480
    q42 = (x**2-y**2)*(8*h**2-3*(x**2+y**2))
    qlm[4, L+2] = q00*q42*3/(448*np.sqrt(10))
    q44 = 3*x**4 - 10*x**2*y**2 + 3*y**4
    qlm[4, L+4] = q00*q44*3/(128*np.sqrt(70))
    #q50 = h*(128*h**4 + x**4 + 10*x**2*y**2 + 9*y**2 - 32*h**2*(x**2+y**2))
    q50 = h*(128*h**4 + 9*x**4 + 10*x**2*y**2 + 9*y**4 - 32*h**2*(x**2+y**2))
    qlm[5, L] = q00*q50*np.sqrt(11)/7168
    q52 = h*(x**2-y**2)*(8*h**2-3*(x**2+y**2))
    qlm[5, L+2] = q00*q52*np.sqrt(33/70)/512
    qlm[5, L+4] = q00*(3*x**4-10*x**2*y**2+3*y**4)*np.sqrt(11/70)*3*h/1024

    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2

    # Truncate if LMax < 5
    if LMax < 5:
        qlm = qlm[:LMax+1, L-LMax:L+LMax+1]
    return qlm
