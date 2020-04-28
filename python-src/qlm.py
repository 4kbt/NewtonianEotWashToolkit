# -*- coding: utf-8 -*-
"""
Created on Thu Aug 03 15:58:03 2017

@author: John Greendeer Lee
"""
import numpy as np
import scipy.special as sp


def sphere(L, mass, R):
    """
    """
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    qlm[0, L] = np.sqrt(1/(4*np.pi))*mass*4*np.pi*R**3/3
    return qlm


def cylinder(L, mass, H, R):
    """
    Recursive calculation for inner multipole moments of a cylinder symmetric
    around the z-axis and x,y-plane or height H and radius R. The result is
    proportional to the mass (= density * (pi H R^2)). The result is computed
    only up to L and includes only the surviving q(l,m) with l even and m=0.
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
    attempt
    """
    factor = mass*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    phih = phih % (2*np.pi)
    if (H == 0) or (Ro < Ri) or (phih == 0) or (phih > np.pi/2):
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
    This is a non-recursive attempt
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


def tri_prism(L, mass, H, a, d, phic):
    """
    Attempt at isosceles triangular prism. This think looks like a mess.

    XXX Need to compare to MULTIN calculations!
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


def tri_prism2(L, mass, H, R, phic, phih):
    """
    Attempt at isosceles triangular prism. This think looks like a mess.

    XXX Need to compare to MULTIN calculations!
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


def rect_prism(L, mass, H, a, b, phic):
    """
    Attempt at isosceles triangular prism. This think looks like a mess.

    XXX Need to compare to MULTIN calculations!
    """
    factor = mass*np.sqrt(1/(4.*np.pi))
    qlm = np.zeros([L+1, 2*L+1], dtype='complex')
    if (H == 0) or (b == 0) or (a == 0):
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
                        kfac = a**(2*k+m-jp2)*b**(jp2)-b**(2*k+m-jp2)*a**(jp2)
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
    Attempt at N-sided regular prism. This think looks like a mess.

    XXX Need to compare to MULTIN calculations!
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
