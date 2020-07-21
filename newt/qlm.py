# -*- coding: utf-8 -*-
"""
Created on Thu Aug 03 15:58:03 2017

@author: John Greendeer Lee
"""
import numpy as np
import scipy.special as sp


def sphere(L, mass, R):
    """
    A sphere with a given mass and radius, R, behaves exactly as a point mass.
    When placed at the origin, it trivially only gives rise to the q00 moment.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    mass : float
        Mass of the cylinder
    H : float
        Total height of the cylinder
    R : float
        Radius of the cylinder

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    qlm[0, L] = mass/np.sqrt((4*np.pi))
    return qlm


def cylinder(L, mass, H, R):
    """
    Recursive calculation for inner multipole moments of a cylinder symmetric
    around the z-axis and x,y-plane or height H and radius R. The result is
    proportional to the mass (= density * (pi H R^2)). The result is computed
    only up to L and includes only the surviving q(l,m) with l even and m=0.
    The cylinder has a height H and extends above and below the xy-plane by
    H/2.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    mass : float
        Mass of the cylinder
    H : float
        Total height of the cylinder
    R : float
        Radius of the cylinder

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = mass*np.sqrt(1/(4.*np.pi))
    Slk = np.zeros([L+1, L//2+1])
    Slk[0, 0] = 1.0
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (H == 0) or (R == 0):
        return qlm
    qlm[0, L] = factor*Slk[0, 0]
    for l in range(2, L+1, 2):
        Slk[l, 0] = (l-1)*H**2/(4.0*(l+1))*Slk[l-2, 0]
        for k in range(1, l//2+1):
            Slk[l, k] = -Slk[l, k-1]*(l-2*k+3)*(l-2*k+2)/((k+1)*k)*(R/H)**2
        qlm[l, L] = factor*np.sqrt(2*l+1)*np.sum(Slk[l])
    return qlm


def annulus(L, mass, H, Ri, Ro, phic, phih):
    """
    Only L-M even survive. We use the notation of Stirling and Schlamminger to
    compute the inner moments of an annular section. This is a non-recursive
    attempt. The solid has a height H and extends above and below the xy-plane
    by H/2.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    mass : float
        Mass of the annular section
    H : float
        Total height of the annular section
    Ri : float
        Inner radius of the annular section
    Ro : float
        Outer radius of the annular section
    phic : float
        Average angle of annular section
    phih : float
        Half of the total angular span of the annular section

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = mass*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    phih = phih % (2*np.pi)
    if (H == 0) or (Ro < Ri) or (phih == 0) or (phih > np.pi):
        return qlm
    rfac = Ro**2 - Ri**2
    for l in range(L+1):
        fac = factor*np.sqrt(2*l+1)
        for m in range(l+1):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            fac2 *= np.sinc(m*phih/np.pi)*np.exp(-1j*m*phic)
            # Make sure (l-m) even
            if not (l-m) % 2:
                for k in range((l-m)//2+1):
                    gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                    gamsum += sp.gammaln(l-m-2*k+2)
                    slk = (-1)**(k+m)*H**(l-2*k-m)/(2**(l-1)*(2*k+m+2))
                    slk *= (Ro**(2*k+m+2) - Ri**(2*k+m+2))/np.exp(gamsum)
                    qlm[l, L+m] += slk
                # Multiply by factor dependent only on (l,m)
                qlm[l, L+m] *= fac2
    # Divide by common factor
    qlm /= rfac
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def cone(L, mass, P, R, phic, phih):
    """
    We use the notation of Stirling and Schlamminger to compute the inner
    moments of a section of a cone of with apex at z=P and base radius of R.
    This is a non-recursive attempt.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    mass : float
        Mass of the cone section
    P : float
        Total height of the cone section, extends from the xy-plane up to z=P.
    R : float
        Radius of the base of the cone section
    phic : float
        Average angle of cone section
    phih : float
        Half of the total angular span of the cone section

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = 3*mass*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    phih = phih % (2*np.pi)
    if (P == 0) or (R == 0) or (phih == 0) or (phih > np.pi):
        return qlm
    for l in range(L+1):
        fac = factor*np.sqrt(2*l+1)/np.exp(sp.gammaln(l+4))
        for m in range(l+1):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            fac2 *= np.sinc(m*phih/np.pi)*np.exp(-1j*m*phic)
            # No restriction on (l-m)
            for k in range((l-m)//2+1):
                gamsum = sp.gammaln(2*k+m+2)
                gamsum -= sp.gammaln(k+1) + sp.gammaln(m+k+1)
                slk = (-1)**(k+m)*P**(l-2*k-m)*R**(2*k+m)/(2**(2*k+m-1))
                qlm[l, L+m] += slk*np.exp(gamsum)
            # Multiply by factor dependent only on (l,m)
            qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def tri_iso_prism(L, mass, H, a, d, phic):
    """
    The isosceles triangular prism has a height H and extends above and below
    the xy-plane by H/2. The triangular faces have vertices at (x,y)=(0,0),
    (d,a/2), and (d,-a/2).

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    mass : float
        Mass of the prism
    H : float
        Total height of the prism
    a : float
        Length of side opposite origin
    d : float
        Distance to the side opposite the origin
    phic : float
        Average angle of prism

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = mass*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (H == 0) or (d == 0) or (a == 0):
        return qlm
    aod = a/(2*d)
    for l in range(L+1):
        fac = factor*np.sqrt(2*l+1)/2**(l-1)
        for m in range(l+1):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            fac2 *= np.exp(-1j*m*phic)
            # Only (l-m) even
            if not (l-m) % 2:
                for k in range((l-m)//2+1):
                    gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                    gamsum += sp.gammaln(l-m-2*k+2)
                    slk = (-1)**(k+m)*H**(l-2*k-m)*d**(2*k+m)/(2*k+m+2)
                    psum = 0
                    for p in range(m//2 + 1):
                        ksum = 0
                        pfac = (-1)**p*sp.comb(m, 2*p)
                        for j in range(k+1):
                            ksum += sp.comb(k, j)*(aod)**(2*j+2*p)/(2*j+2*p+1)
                        psum += pfac*ksum
                    slk *= psum/np.exp(gamsum)
                    qlm[l, L+m] += slk
                # Multiply by factor dependent only on (l,m)
                qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def tri_iso_prism2(L, mass, H, R, phic, phih):
    """
    The isosceles triangular prism has a height H and extends above and below
    the xy-plane by H/2. The triangular faces span an angle of phih where the
    equal length sides have length R.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    mass : float
        Mass of the prism
    H : float
        Total height of the prism
    R : float
        Length of equal length sides of triangular face
    phic : float
        Average angle of prism
    phih : float
        Half of the total angular span of the prism

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = mass*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    phih = phih % (2*np.pi)
    if (H == 0) or (R == 0) or (phih == 0) or (phih > np.pi/2):
        return qlm
    d = R*np.cos(phih)
    a = np.tan(phih)
    for l in range(L+1):
        fac = factor*np.sqrt(2*l+1)/2**(l-1)
        for m in range(l+1):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            fac2 *= np.exp(-1j*m*phic)
            # Only (l-m) even
            if not (l-m) % 2:
                for k in range((l-m)//2+1):
                    gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                    gamsum += sp.gammaln(l-m-2*k+2)
                    slk = (-1)**(k+m)*H**(l-2*k-m)*d**(2*k+m)/(2*k+m+2)
                    psum = 0
                    for p in range(m//2 + 1):
                        ksum = 0
                        pfac = (-1)**p*sp.comb(m, 2*p)
                        for j in range(k+1):
                            ksum += sp.comb(k, j)*a**(2*j+2*p)/(2*j+2*p+1)
                        psum += pfac*ksum
                    slk *= psum/np.exp(gamsum)
                    qlm[l, L+m] += slk
                # Multiply by factor dependent only on (l,m)
                qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def tri_prism(L, mass, H, d, y1, y2):
    """
    The triangular prism has a height H and extends above and below the
    xy-plane by H/2. The triangular faces have vertices at (x,y)=(0,0),
    (d,y1), and (d,y2).

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    mass : float
        Mass of the prism
    H : float
        Total height of the prism
    d : float
        X-position of first and second vertices
    y1 : float
        Y-position of first vertex
    y2 : float
        Y-position of second vertex
    phic : float
        Average angle of prism

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = mass*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (H == 0) or (d <= 0) or (y1 == y2):
        return qlm

    for l in range(L+1):
        fac = factor*np.sqrt(2*l+1)/2**(l-1)
        for m in range(l+1):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            # Only (l-m) even
            if not (l-m) % 2:
                for k in range((l-m)//2+1):
                    gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                    gamsum += sp.gammaln(l-m-2*k+2)
                    slk = (-1)**(k+m)*H**(l-2*k-m)*d**(2*k+m)/(2*k+m+2)
                    psum = 0
                    for p in range(m+k+1):
                        ksum = 0
                        pfac = sp.comb(m+k, p)
                        for j in range(k+1):
                            pj = p+j
                            kfac = (1j)**pj*(-1)**p*(y2**(pj+1)-y1**(pj+1))
                            ksum += sp.comb(k, j)*kfac*d**(-pj)/(pj+1)
                        psum += pfac*ksum
                    slk *= psum/np.exp(gamsum)
                    qlm[l, L+m] += slk
                # Multiply by factor dependent only on (l,m)
                qlm[l, L+m] *= fac2/(y2-y1)
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def rect_prism(L, mass, H, a, b, phic):
    """
    Rectangular prism centered on the origin with height H and sides of length
    a and b extending along the x and y axes respectively when phic=0.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    mass : float
        Mass of the prism
    H : float
        Total height of the prism
    a : float
        Length of prism
    b : float
        Width of prism
    phic : float
        Average angle of prism

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = mass*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (H <= 0) or (b <= 0) or (a <= 0):
        return qlm
    # l-m even, m even -> l even
    for l in range(0, L+1, 2):
        fac = factor*np.sqrt(2*l+1)
        # m even
        for m in range(0, l+1, 2):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            fac2 *= np.exp(-1j*m*phic)
            for k in range((l-m)//2+1):
                gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                gamsum += sp.gammaln(l-m-2*k+2)
                slk = (-1)**(k)*H**(l-2*k-m)/((2*k+m+2)*2**(l+2*k+m))
                psum = 0
                for p in range(m//2 + 1):
                    ksum = 0
                    pfac = (-1)**p*sp.comb(m, 2*p)
                    for j in range(k+1):
                        jp2 = 2*j+2*p
                        kfac = a**(2*k+m-jp2)*b**(jp2)
                        kfac += (-1)**(m//2)*b**(2*k+m-jp2)*a**(jp2)
                        ksum += sp.comb(k, j)*kfac/(jp2+1)
                    psum += pfac*ksum
                slk *= psum/np.exp(gamsum)
                qlm[l, L+m] += slk
            # Multiply by factor dependent only on (l,m)
            qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def ngon_prism(L, mass, H, a, phic, N):
    """
    Regular N-sided prism centered on the origin with height H with sides of
    length a. When phic=0, the first side is oriented parallel to the y-axis.

    Inputs
    ------
    L : int
        Maximum order for multipole expansion
    mass : float
        Mass of the prism
    H : float
        Total height of the prism
    a : float
        Length of sides of prism
    phic : float
        Average angle of prism
    N : int
        Number of sides to regular prism

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = mass*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (H == 0) or (a == 0) or (N <= 0):
        return qlm
    tanN = np.tan(np.pi/N)
    for l in range(L+1):
        fac = factor*np.sqrt(2*l+1)
        for m in range(0, l+1, N):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            fac2 *= np.exp(-1j*m*phic)
            # Only (l-m) even
            if not (l-m) % 2:
                for k in range((l-m)//2+1):
                    gamsum = sp.gammaln(k+1) + sp.gammaln(m+k+1)
                    gamsum += sp.gammaln(l-m-2*k+2)
                    slk = (-1)**(k+m)*H**(l-2*k-m)*a**(2*k+m)
                    slk /= (2*k+m+2)*2**(l+2*k+m-1)
                    psum = 0
                    for p in range(m//2 + 1):
                        ksum = 0
                        pfac = (-1)**p*sp.comb(m, 2*p)
                        for j in range(k+1):
                            jp2 = 2*j+2*p
                            ksum += sp.comb(k, j)*tanN**(jp2-2*k-m)/(jp2+1)
                        psum += pfac*ksum
                    slk *= psum/np.exp(gamsum)
                    qlm[l, L+m] += slk
                # Multiply by factor dependent only on (l,m)
                qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def tetrahedron(L, mass, x, y, z):
    """
    This shape consists of a tetrahedron having three mutually perpendicular
    triangular faces that meet at the origin. The fourth triangular face is
    defined by points at corrdinates x, y, and z along the xhat, yhat, and zhat
    axes respectively.

    Inputs
    ------
    L : int
        Maximum order of inner multipole moments
    mass : float
        Mass of the tetrahedron
    x : float
        Distance to vertex along x-axis
    y : float
        Distance to vertex along y-axis
    z : float
        Distance to vertex along z-axis

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = 6*mass*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (x <= 0) or (y <= 0) or (z <= 0):
        return qlm

    for l in range(L+1):
        fac = factor*np.sqrt(2*l+1)/np.exp(sp.gammaln(l+4))
        for m in range(l+1):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            for k in range((l-m)//2+1):
                m2k = m+2*k
                slk = (-1)**(m+k)*z**(l-m2k)/2**m2k
                psum = 0
                for p in range(m+k+1):
                    ksum = 0
                    for j in range(k+1):
                        pj = p+j
                        kfac = (1j)**pj*(-1)**p*y**pj*x**(m2k-pj)
                        gampj = sp.gammaln(m2k-pj+1) + sp.gammaln(pj+1)
                        gampj -= sp.gammaln(j+1) + sp.gammaln(k-j+1)
                        gampj -= sp.gammaln(p+1) + sp.gammaln(m+k-p+1)
                        ksum += kfac*np.exp(gampj)
                    psum += ksum
                slk *= psum
                qlm[l, L+m] += slk
            # Multiply by factor dependent only on (l,m)
            qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def tetrahedron2(L, mass, x, y1, y2, z):
    """
    This shape consists of a tetrahedron having three mutually perpendicular
    triangular faces that meet at the origin. The fourth triangular face is
    defined by points at corrdinates x, y, and z along the xhat, yhat, and zhat
    axes respectively.

    Inputs
    ------
    L : int
        Maximum order of inner multipole moments
    mass : float
        Mass of the tetrahedron
    x : float
        Distance to vertex along x-axis
    y1 : float
        Distance to vertex along y-axis
    z : float
        Distance to vertex along z-axis

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = 6*mass*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (x <= 0) or (y1 == y2) or (z <= 0):
        return qlm

    for l in range(L+1):
        fac = factor*np.sqrt(2*l+1)/np.exp(sp.gammaln(l+4))/(y2-y1)
        for m in range(l+1):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            for k in range((l-m)//2+1):
                m2k = m+2*k
                slk = (-1)**(m+k)*z**(l-m2k)/2**m2k
                psum = 0
                for p in range(m+k+1):
                    ksum = 0
                    for j in range(k+1):
                        pj = p+j
                        kfac = (1j)**pj*(-1)**p*(y2**(pj+1)-y1**(pj+1))*x**(m2k-pj)
                        gampj = sp.gammaln(m2k+2)
                        gampj -= sp.gammaln(j+1) + sp.gammaln(k-j+1)
                        gampj -= sp.gammaln(p+1) + sp.gammaln(m+k-p+1)
                        ksum += kfac*np.exp(gampj)/(pj+1)
                    psum += ksum
                slk *= psum
                qlm[l, L+m] += slk
            # Multiply by factor dependent only on (l,m)
            qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def pyramid(L, mass, x, y, z):
    """
    This shape consists of a pyramid having three mutually perpendicular
    triangular faces that meet at the origin. The fourth triangular face is
    defined by points at corrdinates x, y, and z along the xhat, yhat, and zhat
    axes respectively.

    Inputs
    ------
    L : int
        Maximum order of inner multipole moments
    mass : float
        Mass of the pyramid
    x : float
        Distance to vertex along x-axis
    y1 : float
        Distance to vertex along y-axis
    z : float
        Distance to vertex along z-axis

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    factor = 3*mass*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (x <= 0) or (y <= 0) or (z <= 0):
        return qlm

    for l in range(L+1):
        fac = factor*np.sqrt(2*l+1)/np.exp(sp.gammaln(l+4))
        # m even
        for m in range(0, l+1, 2):
            fac2 = fac*np.sqrt(np.exp(sp.gammaln(l+m+1)+sp.gammaln(l-m+1)))
            for k in range((l-m)//2+1):
                m2k = m+2*k
                slk = (-1)**(m+k)*z**(l-m2k)/2**m2k
                psum = 0
                for p in range(m+k+1):
                    ksum = 0
                    for j in range(k+1):
                        pj = p+j
                        # p+j must be even, but doesn't simplify well
                        if (pj % 2) == 0:
                            kfac = y**pj*x**(m2k-pj)
                            kfac += (-1)**(m//2)*x**pj*y**(m2k-pj)
                            kfac *= (1j)**pj*(-1)**p
                            gampj = sp.gammaln(m2k+2)
                            gampj -= sp.gammaln(j+1) + sp.gammaln(k-j+1)
                            gampj -= sp.gammaln(p+1) + sp.gammaln(m+k-p+1)
                            ksum += kfac*np.exp(gampj)/(pj+1)
                    psum += ksum
                slk *= psum
                qlm[l, L+m] += slk
            # Multiply by factor dependent only on (l,m)
            qlm[l, L+m] *= fac2
    # Moments always satisfy q(l, -m) = (-1)^m q(l, m)*
    ms = np.arange(-L, L+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, L] /= 2
    return qlm


def cyl_mom(L, M, dens, H, R):
    gamFac = np.sqrt(np.exp(sp.gammaln(L-M+1)-sp.gammaln(L+M+1)))
    fact = np.sqrt((2*L+1)/(4*np.pi))*gamFac*sp.factorial(L-M)/(2.*np.pi)
    R22 = np.sqrt(R**2+H**2)
    theta22 = H/R22
    theta21 = -H/R22
    mom = 0
    PLM22 = sp.lpmn(L+2, L+2, theta22)[0][:, -1]
    PLM21 = sp.lpmn(L+2, L+2, theta21)[0][:, -1]
    momval = PLM22-PLM21
    ks = np.arange(L-M)
    fac2 = (ks*2+1+M)*2*M/((M+2*ks)*(M+2*ks+2))
    mom = fact*fac2*R*R22**(L+2)*momval[-len(ks):]
    mom = np.sum(mom)
    if M == 0:
        mom = fact*R*R22**(L+2)*momval[1]
    return mom
