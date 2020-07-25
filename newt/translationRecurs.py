# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 08:16:19 2020

@author: John Greendeer Lee
"""
import numpy as np
import scipy.special as sp
import scipy.linalg as sla


def transl_yuk_z_SR_recurs(LMax, dr, k):
    """
    Translate coaxially from regular to singular, Gumerov and Duraiswami.

    Inputs
    ------
    l : int
        Order of multipole expansion to output rotation matrix coefficient H

    Reference
    ---------
    "Recursions for the computation of multipole translation and rotation
    coefficients for the 3-d helmholtz equation."

    https://arxiv.org/pdf/1403.7698v1.pdf
    """
    # Create (E|F)_{l,m}^{m}
    eflm = np.zeros([2*LMax+1, LMax+1])
    # Start (E|F)_{l, 0}^{0}
    ls = np.arange(2*LMax+1)
    eflm[:, 0] = (-1)**ls*np.sqrt(2*ls+1)*sp.spherical_kn(ls, k*dr)
    # now recursion 4.86 for (E|F)_{l, m}^{m}
    for m in range(LMax):
        for l in range(2*LMax-m):
            # (n=m)
            blmm = get_bnm(l, -m-1)
            blmp = get_bnm(l+1, m)
            bmm = get_bnm(m+1, -m-1)
            eflm[l, m+1] = (eflm[l-1, m]*blmm - eflm[l+1, m]*blmp)/bmm

    efms = []
    # Now create (E|F)^m matrices (p-|m|)x(p-|m|)
    # Start with (2*p-|m|)x(p-|m|) and truncate
    for m in range(LMax+1):
        efm = np.zeros([2*LMax+1-2*m, LMax-m+1])
        efm[:, 0] = eflm[m:2*LMax+1-m, m]
        for n in range(LMax-m):
            for l in range(n+1, 2*(LMax-m)-n):
                anm = get_anm(n+m, m)
                alm = get_anm(l+m, m)
                almm = get_anm(l-1+m, m)
                anmm = get_anm(n-1+m, m)
                print(l, n+1, l+m, anm, anmm, alm, almm)
                if anmm != 0:
                    print(l, n, m)
                    efm[l, n+1] = (almm*efm[l-1, n] - alm*efm[l+1, n]
                                   + anmm*efm[l, n-1])/anm
                else:
                    efm[l, n+1] = (alm*efm[l+1, n] - almm*efm[l-1, n])/anm
        efms.append(efm)

    for m in range(LMax+1):
        efms[m] = efms[m][:LMax-m+1, :LMax-m+1]
        lm = len(efms[m])
        # Start from m since (E|F)^m matrix starts at (m,m) corner
        nls = (np.arange(m, m+lm))
        enl = np.transpose(efms[m])*np.outer((-1)**(nls), (-1)**(nls))
        efms[m] += enl
        efms[m] -= np.diag(np.diag(enl))

    return eflm, efms


def transl_yuk_z_RR_recurs(LMax, dr, k):
    """
    Translate coaxially from regular to regular, Gumerov and Duraiswami.

    Inputs
    ------
    l : int
        Order of multipole expansion to output rotation matrix coefficient H

    Reference
    ---------
    "Recursions for the computation of multipole translation and rotation
    coefficients for the 3-d helmholtz equation."

    https://arxiv.org/pdf/1403.7698v1.pdf
    """
    # Create (E|F)_{l,m}^{m}
    eflm = np.zeros([2*LMax+1, LMax+1])
    # Start (E|F)_{l, 0}^{0}
    ls = np.arange(2*LMax+1)
    eflm[:, 0] = (-1)**ls*np.sqrt(2*ls+1)*sp.spherical_in(ls, k*dr)
    # now recursion 4.86 for (E|F)_{l, m}^{m}
    for m in range(LMax):
        for l in range(2*LMax-m):
            # (n=m)
            blmm = get_bnm(l, -m-1)
            blmp = get_bnm(l+1, m)
            bmm = get_bnm(m+1, -m-1)
            eflm[l, m+1] = (eflm[l-1, m]*blmm - eflm[l+1, m]*blmp)/bmm

    efms = []
    # Now create (E|F)^m matrices (p-|m|)x(p-|m|)
    # Start with (2*p-|m|)x(p-|m|) and truncate
    for m in range(LMax+1):
        efm = np.zeros([2*LMax+1-2*m, LMax-m+1])
        efm[:, 0] = eflm[m:2*LMax+1-m, m]
        for n in range(LMax-m):
            for l in range(n+1, 2*(LMax-m)-n):
                anm = get_anm(n+m, m)
                alm = get_anm(l+m, m)
                almm = get_anm(l-1+m, m)
                anmm = get_anm(n-1+m, m)
                print(l, n+1, l+m, anm, anmm, alm, almm)
                if anmm != 0:
                    print(l, n, m)
                    efm[l, n+1] = (almm*efm[l-1, n] - alm*efm[l+1, n]
                                   + anmm*efm[l, n-1])/anm
                else:
                    efm[l, n+1] = (alm*efm[l+1, n] - almm*efm[l-1, n])/anm
        efms.append(efm)

    for m in range(LMax+1):
        efms[m] = efms[m][:LMax-m+1, :LMax-m+1]
        lm = len(efms[m])
        # Start from m since (E|F)^m matrix starts at (m,m) corner
        nls = (np.arange(m, m+lm))
        enl = np.transpose(efms[m])*np.outer((-1)**(nls), (-1)**(nls))
        efms[m] += enl
        efms[m] -= np.diag(np.diag(enl))

    return eflm, efms


def transl_newt_z_SR_recurs(LMax, dr):
    """
    Translate coaxially from regular to singular, Gumerov and Duraiswami.

    Inputs
    ------
    l : int
        Order of multipole expansion to output rotation matrix coefficient H

    Reference
    ---------
    "Recursions for the computation of multipole translation and rotation
    coefficients for the 3-d helmholtz equation."

    https://arxiv.org/pdf/1403.7698v1.pdf
    """
    # Create (E|F)_{l,m}^{m}
    eflm = np.zeros([2*LMax+1, LMax+1])
    # Start (E|F)_{l, 0}^{0}
    ls = np.arange(2*LMax+1)
    eflm[:, 0] = (-1)**ls*np.sqrt(2*ls+1)*sp.factorial2(2*ls-1)/dr**(ls+1)
    # now recursion 4.86 for (E|F)_{l, m}^{m}
    for m in range(LMax):
        for l in range(2*LMax-m):
            # (n=m)
            blmm = get_bnm(l, -m-1)
            blmp = get_bnm(l+1, m)
            bmm = get_bnm(m+1, -m-1)
            eflm[l, m+1] = (eflm[l-1, m]*blmm - eflm[l+1, m]*blmp)/bmm

    efms = []
    # Now create (E|F)^m matrices (p-|m|)x(p-|m|)
    # Start with (2*p-|m|)x(p-|m|) and truncate
    for m in range(LMax+1):
        efm = np.zeros([2*LMax+1-2*m, LMax-m+1])
        efm[:, 0] = eflm[m:2*LMax+1-m, m]
        for n in range(LMax-m):
            for l in range(n+1, 2*(LMax-m)-n):
                anm = get_anm(n+m, m)
                alm = get_anm(l+m, m)
                almm = get_anm(l-1+m, m)
                anmm = get_anm(n-1+m, m)
                print(l, n+1, l+m, anm, anmm, alm, almm)
                if anmm != 0:
                    print(l, n, m)
                    efm[l, n+1] = (almm*efm[l-1, n] - alm*efm[l+1, n]
                                   + anmm*efm[l, n-1])/anm
                else:
                    efm[l, n+1] = (alm*efm[l+1, n] - almm*efm[l-1, n])/anm
        efms.append(efm)

    for m in range(LMax+1):
        efms[m] = efms[m][:LMax-m+1, :LMax-m+1]
        lm = len(efms[m])
        # Start from m since (E|F)^m matrix starts at (m,m) corner
        nls = (np.arange(m, m+lm))
        enl = np.transpose(efms[m])*np.outer((-1)**(nls), (-1)**(nls))
        efms[m] += enl
        efms[m] -= np.diag(np.diag(enl))

    return eflm, efms


def transl_newt_z_RR_recurs(LMax, dr):
    """
    Translate coaxially from regular to regular, Gumerov and Duraiswami.

    Inputs
    ------
    l : int
        Order of multipole expansion to output rotation matrix coefficient H

    Reference
    ---------
    "Recursions for the computation of multipole translation and rotation
    coefficients for the 3-d helmholtz equation."

    https://arxiv.org/pdf/1403.7698v1.pdf
    """
    # Create (E|F)_{l,m}^{m}
    eflm = np.zeros([2*LMax+1, LMax+1])
    # Start (E|F)_{l, 0}^{0}
    ls = np.arange(2*LMax+1)
    eflm[:, 0] = (-1)**ls*np.sqrt(2*ls+1)*dr**ls/sp.factorial2(2*ls+1)
    # now recursion 4.86 for (E|F)_{l, m}^{m}
    for m in range(LMax):
        for l in range(2*LMax-m):
            # (n=m)
            blmm = get_bnm(l, -m-1)
            blmp = get_bnm(l+1, m)
            bmm = get_bnm(m+1, -m-1)
            eflm[l, m+1] = (eflm[l-1, m]*blmm - eflm[l+1, m]*blmp)/bmm

    efms = []
    # Now create (E|F)^m matrices (p-|m|)x(p-|m|)
    # Start with (2*p-|m|)x(p-|m|) and truncate
    for m in range(LMax+1):
        efm = np.zeros([2*LMax+1-2*m, LMax-m+1])
        efm[:, 0] = eflm[m:2*LMax+1-m, m]
        for n in range(LMax-m):
            for l in range(n+1, 2*(LMax-m)-n):
                anm = get_anm(n+m, m)
                alm = get_anm(l+m, m)
                almm = get_anm(l-1+m, m)
                anmm = get_anm(n-1+m, m)
                print(l, n+1, l+m, anm, anmm, alm, almm)
                if anmm != 0:
                    print(l, n, m)
                    efm[l, n+1] = (almm*efm[l-1, n] - alm*efm[l+1, n]
                                   + anmm*efm[l, n-1])/anm
                else:
                    efm[l, n+1] = (alm*efm[l+1, n] - almm*efm[l-1, n])/anm
        efms.append(efm)

    for m in range(LMax+1):
        efms[m] = efms[m][:LMax-m+1, :LMax-m+1]
        lm = len(efms[m])
        # Start from m since (E|F)^m matrix starts at (m,m) corner
        nls = (np.arange(m, m+lm))
        enl = np.transpose(efms[m])*np.outer((-1)**(nls), (-1)**(nls))
        efms[m] += enl
        efms[m] -= np.diag(np.diag(enl))

    return eflm, efms


def transl_newt_z_RR_recurs2(LMax, dr):
    """
    Translate coaxially from regular to regular, Gumerov and Duraiswami. In a
    slight contradiction to the paper, our choice of normalization (alpha_n^m)
    does not require all three normalization factors in the translation matrix,
    see eqn 19.

    Inputs
    ------
    l : int
        Order of multipole expansion to output rotation matrix coefficient H

    Reference
    ---------
    "Comparison of the efficiency of translation operators used in the fast
    multipole method for the 3D Laplace equation"

    http://legacydirs.umiacs.umd.edu/~gumerov/PDFs/cs-tr-4701.pdf
    """
    # Descending array of lp from LMax through 0
    lp = np.arange(LMax+1)
    rvals = (-dr)**lp/sp.factorial(lp)
    cols = [rvals[0], *np.zeros(LMax)]
    rrms = []
    fac = alphanm(lp, 0)
    fac2 = np.outer(fac, 1/fac)
    rrms.append((sla.toeplitz(cols, rvals)*fac2).T)
    # Each m has a matrix that is (LMax-|m|)x(LMax-|m|) in size
    for m in range(1, LMax+1):
        fac = alphanm(np.arange(m, LMax+1), m)
        fac2 = np.outer(fac, 1/fac)
        rrm = sla.toeplitz(cols[:-m], rvals[:-m])*fac2
        rrms.append(rrm.T)
    return rrms


def alphanm(n, m):
    anm = (-1)**n*1j**(-np.abs(m))*np.sqrt(4*np.pi/((2*n+1)*sp.factorial(n-m)*sp.factorial(n+m)))
    return anm


def apply_trans_mat(qlm, efms):
    L = len(qlm)-1
    print(L)
    qNew = np.zeros([L+1, 2*L+1], dtype='complex')
    qNew[:, L] = np.dot(efms[0], qlm[:, L])
    for m in range(1, L+1):
        qNew[m:, L+m] = np.dot(efms[m], qlm[m:, L+m])
        qNew[m:, L-m] = np.dot(efms[m], qlm[m:, L-m])
    return qNew


def get_bnm(n, m):
    """
    B-recursion function for translation matrices.
    """
    if np.abs(m) > n:
        bval = 0
    else:
        bval = np.sqrt((n-m-1)*(n-m)/((2*n-1)*(2*n+1)))*np.sign(n)
    return bval


def get_anm(n, m):
    """
    A-recursion function for translation matrices.
    """
    am = abs(m)
    if am > n:
        aval = 0
    else:
        aval = np.sqrt((n+am+1)*(n-am+1)/((2*n+1)*(2*n+3)))
    return aval
