# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 22:52:29 2020

@author: John Greendeer Lee
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp


def rotate_H_recurs(p, beta, Hn_prev=None):
    """
    Rotates the wigner-like coefficients with recursive algorithm developed by
    Gumerov and Duraiswami.

    Inputs
    ------
     : int
        Order of multipole expansion to output rotation matrix coefficient H

    Reference
    ---------
    "Recursive Computation of Spherical Harmonic Rotation Coefficients of Large
    Degree" N. Gumerov, R. Duraiswami

    https://arxiv.org/pdf/1403.7698v1.pdf
    """
    if p == 0:
        return 1.0
    else:
        H = []

        cb = np.cos(beta)
        sb = np.sin(beta)
        # cbh = np.cos(beta/2)
        # sbh = np.sin(beta/2)
        # Compute H_n^{0,m} with associated Legendre polynomials
        lps = sp.lpmn(p+1, p+1, cb)[0]
        # Do for only positive ms and use symmetry (step 2)
        for k in range(p+2):
            Hk = np.zeros([2*k+1, 2*k+1])
            ms = np.arange(k+1)
            Hk[k, k:] = (-1)**(ms[:])*np.sqrt(rfac(k-ms[:], k+ms[:]))*lps[ms[:], k]
            H.append(Hk)

        # Compute H_n^{1,m} from H_n^{0,m}'s (step 3)
        for k in range(1, p+1):
            b0 = bnm(k+1, 0)
            ms = np.arange(1, k+1)
            bsM = bnm(k+1, -ms-1)
            bsP = bnm(k+1, ms-1)
            as0 = anm(k, ms)
            H[k][k+1, k+1:] = (bsM[:]*(1-cb)/2.*H[k+1][k+1, k+3:]
                               - bsP[:]*(1+cb)/2.*H[k+1][k+1, k+1:-2]
                               - as0[:]*sb*H[k+1][k+1, k+2:-1])/b0

        # steps 4 & 5
        for n in range(1, p+1):
            # step 5
            # H_n^{-mp,m}'s rely on H_n^{0,m}'s and H_n^{1,m}'s and recursion
            # step 5, mp=0 first
            mp = 0
            fac = dnm(n, -mp)
            fac2 = dnm(n, -mp-1)
            ms = np.arange(mp, n-1)
            H[n][n-mp-1, n+mp+1:-1] = (fac*H[n][n-mp+1, n+mp+1:-1]
                                       + dnm(n, ms)*H[n][n-mp, n+mp:-2]
                                       - dnm(n, ms+1)*H[n][n-mp, n+mp+2:])/fac2

            H[n][n-mp-1, -1] = (fac*H[n][n-mp+1, -1]
                                + dnm(n, n-1)*H[n][n-mp, -2])/fac2
            for mp in range(1, n):
                fac = dnm(n, -mp)
                fac2 = dnm(n, -mp-1)
                ms = np.arange(mp, n-1)
                H[n][n-mp-1, n+mp+1:-1] = (fac*H[n][n-mp+1, n+mp+1:-1]
                                           + dnm(n, ms)*H[n][n-mp, n+mp:-2]
                                           - dnm(n, ms+1)*H[n][n-mp, n+mp+2:])/fac2
                # Handling endpoint stable with other recursion formula
                H[n][n-mp-1, -1] = (fac*H[n][n-mp+1, -1]
                                    + dnm(n, n-1)*H[n][n-mp, -2])/fac2

                # step 4
                fac = dnm(n, mp-1)
                fac2 = dnm(n, mp)
                H[n][n+mp+1, n+mp+1:-1] = (fac*H[n][n+mp-1, n+mp+1:-1]
                                           - dnm(n, ms)*H[n][n+mp, n+mp:-2]
                                           + dnm(n, ms+1)*H[n][n+mp, n+mp+2:])/fac2
                # Handling endpoint stable with other recursion formula
                H[n][n+mp+1, -1] = (fac*H[n][n+mp-1, -1]
                                    - dnm(n, n-1)*H[n][n+mp, -2])/fac2

            # Copy upper triangular to lower triangular
            # H_n^{mp,m} = H_n^{m,mp}
            H[n] = H[n] + np.transpose(H[n]) - np.diag(np.diag(H[n]))
            # Copy lower right triangular to upper left triangular
            H[n] = np.rot90(copy_tri(np.rot90(H[n], -1)), 1)

    return H[:-1]


def copy_tri(M):
    """
    Copies upper triangular half of matrix to the lower triangular half. This
    assumes that the lower half is empty though.
    """
    return M + np.transpose(M)-np.diag(np.diag(M))


def dnm(n, m):
    """
    For H recursion
    """
    return rsign(m)*np.sqrt((n-m)*(n+m+1))/2.


def bnm(n, m):
    """
    For H[k][k+1,k+1:] in recursion
    """
    return rsign(m)*np.sqrt((n-m-1)*(n-m)/((2*n+1)*(2*n-1)))


def anm(n, m):
    """
    For H[k][k+1,k+1:] in recursion

    Since it's used only for computing H_n^{1,m}, we will ignore that
    anm(n<|m|) is zero.
    """
    return np.sqrt((n+1+m)*(n+1-m)/((2.*n+1.)*(2.*n+3.)))


def rfac(n, m):
    r"""
    Computes a ratio of factorials, hopefully somewhat more leniently for
    larger values. This also handles an array of inputs. This is not a
    combinatoric binomial coefficient.

    .. math::

       rfac(n,m) = \frac{n!}{m!}
    """
    return np.exp(sp.gammaln(n+1)-sp.gammaln(m+1))


def rsign(n):
    """
    Returns the sign of the value or array as -1 for n[i]<0, otherwise 1.
    """
    if np.ndim(n) == 0:
        if n < 0:
            return -1
        else:
            return 1
    else:
        return np.array([-1 if n[i] < 0 else 1 for i in range(len(n))])


def epsm(ms):
    """
    Sign convention function to convert from H matrix to small Wigner-d.
    """
    if np.ndim(ms) > 0:
        return [(-1)**m if (m > -1) else 1 for m in ms]
    else:
        if ms > -1:
            return (-1)**(ms)
        else:
            return 1


def dlmn(LMax, beta):
    """
    Compute all wigner small d matrices of angle beta for all orders up to
    LMax, using the recursive method.
    """
    beta = beta % (2*np.pi)
    ms = np.arange(-LMax, LMax+1)
    # If beta is between (pi, 2pi] we need to use as H(-b) and symmetry
    if beta > np.pi:
        beta = 2*np.pi - beta
        fac = np.outer((-1)**(np.abs(ms)), (-1)**(np.abs(ms)))
    else:
        fac = 1
    ds = rotate_H_recurs(LMax, beta)
    epsmmp = epsm(-ms)
    epsmm = epsm(ms)
    efac = np.outer(epsmm, epsmmp)*fac
    for l in range(LMax):
        ds[l] *= efac[LMax-l:LMax+l+1, LMax-l:LMax+l+1]
    return ds


def wignerDl(LMax, alpha, beta, gamma):
    """
    """
    beta = beta % (2*np.pi)
    ms = np.arange(-LMax, LMax+1)
    # If beta is between (pi, 2pi] we need to use as H(-b) and symmetry
    if beta > np.pi:
        beta = 2*np.pi - beta
        fac = np.outer((-1)**(np.abs(ms)), (-1)**(np.abs(ms)))
    else:
        fac = 1
    ds = rotate_H_recurs(LMax, beta)
    expa = np.exp(-1j*ms*alpha)
    expg = np.exp(-1j*ms*gamma)
    epsmmp = epsm(-ms)
    epsmm = epsm(ms)
    expfac = np.outer(expa, expg)
    efac = np.outer(epsmm, epsmmp)*expfac*fac
    for l in range(LMax):
        ds[l] = ds[l].astype('complex')
        ds[l] *= efac[LMax-l:LMax+l+1, LMax-l:LMax+l+1]
    return ds


def rotate_qlm(qlm, alpha, beta, gamma):
    LMax = np.shape(qlm)[0] - 1
    qNew = np.copy(qlm)
    # XXX Should test to make sure really need to go to LMax + 1
    ds = wignerDl(LMax+1, alpha, beta, gamma)
    for k in range(1, LMax+1):
        qNew[k, LMax-k:LMax+k+1] = np.dot(ds[k], qlm[k, LMax-k:LMax+k+1])
    return qNew


def Dlmn(l, m, n, alpha, beta, gamma):
    """
    Compute the (m, n) term of the Wigner D matrix, D^l_{m,n}.
    """
    Dlmn = 0
    kmin = max([m-n, 0])
    kmax = min([l+m, l-n])
    for k in range(int(kmin), int(kmax)+1):
        val = (-1)**k*np.cos(beta/2)**(2*l+m-n-2*k)*np.sin(beta/2)**(n-m+2*k)
        val /= (sp.factorial(k)*sp.factorial(l+m-k)*sp.factorial(l-n-k)*sp.factorial(n-m+k))
        Dlmn += val
    Dlmn *= np.exp(-1j*(alpha*n + gamma*m))*(-1)**(n-m)
    Dlmn *= np.sqrt(sp.factorial(l-m)*sp.factorial(l+m)*sp.factorial(l-n)*sp.factorial(l+n))
    return Dlmn


def Dl(l, alpha, beta, gamma):
    """Compute the mxm Wigner D matrix."""
    Nm = 2*l+1
    Dl = np.zeros([Nm, Nm], dtype=complex)
    for m in range(Nm):
        for n in range(Nm):
            Dl[m, n] = Dlmn(l, -l + m, -l + n, alpha, beta, gamma)
    return Dl


# Example script
# Recursive rotation matrix calculation
beta = np.pi/4
Hs = rotate_H_recurs(100, beta)
epsmmp = epsm(-np.arange(-40, 41))  # sign factor #1
epsmm = epsm(np.arange(-40, 41))    # sign factor #2
# Classic calcultion
H40 = Dl(40, 0, beta, 0)  # transposed relative to Hs

fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].imshow(np.outer(epsmm, epsmmp)*Hs[40])
ax[1].imshow(np.real(H40.T))
